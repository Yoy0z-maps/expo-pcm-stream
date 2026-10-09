package expo.modules.pcmstream

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Refines an MPM pitch to the frequency of the fundamental partial, from the spectrum of the
 * same window.
 *
 * MPM measures the period of the whole waveform. On a stiff string (steel guitar and bass
 * strings) the partials run progressively sharp, f_n = n * f0 * sqrt(1 + B * n^2), so the
 * waveform's period settles between the partials' and reads sharp of the fundamental: ~3 cents
 * on a guitar and ~10-15 cents on a bass. It also wobbles by a few cents as the inharmonic
 * partials drift in relative phase. Tuners and the ear go by the fundamental.
 *
 * This stage measures partials 1..8 from a Blackman-Harris windowed, zero-padded spectrum,
 * fits the string's inharmonicity coefficient B to them (smoothed over time: it is a property
 * of the string, while a single window's estimate is noisy), and combines every partial into one
 * estimate of partial 1, f_n / n * sqrt((1 + B) / (1 + B * n^2)), weighted by n^2 times the
 * partial's power - the weight each partial carries in the autocorrelation. Using all partials
 * keeps it accurate when the fundamental itself is weak, as a phone microphone hears a bass.
 *
 * It also undoes MPM's subharmonic errors: a stiff string's partials can line up better at twice
 * its period than at its period, and a pitch whose odd multiples are empty is really an octave
 * higher.
 *
 * Mirrors FundamentalRefiner.swift. Not thread-safe; all memory is preallocated so the per-hop
 * calls never allocate.
 */
internal class FundamentalRefiner(val windowSize: Int) {
  private val fftSize: Int
  private val half: Int
  private val fft: RealFft
  private val taper = DoubleArray(windowSize) // 4-term Blackman-Harris
  private val buffer: DoubleArray // tapered, zero-padded window
  private val specRe: DoubleArray
  private val specIm: DoubleArray
  private val power: DoubleArray // |X[k]|^2 for k < half (DC zeroed)
  private val partialNumbers = IntArray(MAX_PARTIALS)
  private val partialFrequencies = DoubleArray(MAX_PARTIALS)
  private val partialPowers = DoubleArray(MAX_PARTIALS)
  private var inharmonicity = 0.0
  private var inharmonicityCount = 0
  private var lastPitch = 0.0

  // findPeak's result, kept in fields so a search doesn't allocate.
  private var peakFrequency = 0.0
  private var peakPower = 0.0

  init {
    require(windowSize >= 4) { "windowSize must be >= 4" }
    // 4x zero-padding: on bins this fine the log-parabolic peak interpolation is within
    // ~0.03 cents for a pure tone (2x padding: ~0.2 cents).
    var size = 4
    while (size < 4 * windowSize) size = size shl 1
    fftSize = size
    half = size / 2
    fft = RealFft(size)
    buffer = DoubleArray(size)
    specRe = DoubleArray(half + 1)
    specIm = DoubleArray(half + 1)
    power = DoubleArray(half)

    // Sidelobes at -92 dB: a strong partial can't leak into the measurement of a weak neighbor.
    // The main lobe spans +-4 * sampleRate / W Hz, so cleanly resolved partials take a window of
    // ~4.5 periods of the fundamental (W >= 4.5 * sampleRate / f0). See MIN_PERIODS.
    for (i in 0 until windowSize) {
      val x = 2 * PI * i / windowSize
      taper[i] = 0.35875 - 0.48829 * cos(x) + 0.14128 * cos(2 * x) - 0.01168 * cos(3 * x)
    }
  }

  /**
   * Computes the power spectrum of `window` ([windowSize] samples, already high-passed). Call it
   * before [correctSubharmonic] and [refine] for each new window.
   */
  fun analyze(window: DoubleArray) {
    for (i in 0 until windowSize) buffer[i] = window[i] * taper[i] // the padding stays 0
    fft.forward(buffer, specRe, specIm)
    for (k in 0 until half) power[k] = specRe[k] * specRe[k] + specIm[k] * specIm[k]
    power[0] = 0.0 // DC is never a partial
  }

  /** Returns `frequency`, or an octave (two at most) above it if its odd partials are empty. */
  fun correctSubharmonic(frequency: Double, sampleRate: Double, maxFrequency: Double): Double {
    var pitch = frequency
    if (!resolves(pitch, sampleRate)) return pitch
    val top = min(sampleRate * 0.4, MAX_PARTIAL_FREQUENCY)
    repeat(2) {
      if (2 * pitch > maxFrequency) return pitch
      var odd = 0.0
      var even = 0.0
      var n = 1
      while (n <= 2 * MAX_PARTIALS && n * pitch <= top) {
        if (findPeak(n * pitch, sampleRate)) {
          if (n % 2 == 1) odd += peakPower else even += peakPower
        }
        n++
      }
      if (!(even > 0 && odd < even * SUBHARMONIC_ODD_RATIO)) return pitch
      pitch *= 2
    }
    return pitch
  }

  /**
   * Returns the frequency of the fundamental partial near `pitch` (an MPM estimate), or `pitch`
   * itself when the spectrum doesn't support a refinement.
   */
  fun refine(pitch: Double, sampleRate: Double): Double {
    if (!resolves(pitch, sampleRate)) return pitch
    if (lastPitch == 0.0 || abs(1200 * log2(pitch / lastPitch)) > NOTE_CHANGE_CENTS) {
      inharmonicity = 0.0
      inharmonicityCount = 0
    }
    lastPitch = pitch

    var count = 0
    var strongest = 0.0
    val top = min(sampleRate * 0.4, MAX_PARTIAL_FREQUENCY)
    for (n in 1..MAX_PARTIALS) {
      val target = pitch * n
      if (target > top) break
      if (!findPeak(target, sampleRate)) continue
      partialNumbers[count] = n
      partialFrequencies[count] = peakFrequency
      partialPowers[count] = peakPower
      count++
      strongest = max(strongest, peakPower)
    }
    if (strongest <= 0) return pitch

    // Keep the strong partials, with weights n^2 * relative power.
    var strong = 0
    for (i in 0 until count) {
      if (partialPowers[i] < strongest * STRONG_PARTIAL_RATIO) continue
      val n = partialNumbers[i].toDouble()
      partialNumbers[strong] = partialNumbers[i]
      partialFrequencies[strong] = partialFrequencies[i]
      partialPowers[strong] = n * n * partialPowers[i] / strongest
      strong++
    }

    // Stiff string: (f_n / n)^2 = f0^2 + f0^2 * B * n^2 - a weighted line fit in n^2 gives B.
    if (strong >= 3) {
      var sw = 0.0
      var sx = 0.0
      var sy = 0.0
      var sxx = 0.0
      var sxy = 0.0
      for (i in 0 until strong) {
        val n = partialNumbers[i].toDouble()
        val x = n * n
        val ratio = partialFrequencies[i] / n
        val y = ratio * ratio
        val w = partialPowers[i]
        sw += w
        sx += w * x
        sy += w * y
        sxx += w * x * x
        sxy += w * x * y
      }
      val det = sw * sxx - sx * sx
      if (det > 1e-12 * sw * sxx) {
        val slope = (sw * sxy - sx * sy) / det
        val intercept = (sy - slope * sx) / sw
        if (intercept > 0) {
          inharmonicityCount++
          val alpha = max(INHARMONICITY_SMOOTHING, 1.0 / inharmonicityCount)
          inharmonicity += alpha * (slope / intercept - inharmonicity)
        }
      }
    }

    val b = min(max(inharmonicity, 0.0), MAX_INHARMONICITY)
    var sum = 0.0
    var weights = 0.0
    for (i in 0 until strong) {
      val n = partialNumbers[i].toDouble()
      sum += partialPowers[i] * partialFrequencies[i] / n * sqrt((1 + b) / (1 + b * n * n))
      weights += partialPowers[i]
    }
    val fundamental = sum / weights
    if (abs(1200 * log2(fundamental / pitch)) > MAX_CORRECTION_CENTS) return pitch
    return fundamental
  }

  private fun resolves(pitch: Double, sampleRate: Double): Boolean = pitch * windowSize >= MIN_PERIODS * sampleRate

  /**
   * Finds the interpolated spectral peak within SEARCH_CENTS (at least 2 bins) of `target` Hz
   * into [peakFrequency] and [peakPower]. Returns false if the band's maximum sits on its edge
   * (a neighbor's slope, not a peak of its own).
   */
  private fun findPeak(target: Double, sampleRate: Double): Boolean {
    val center = target * fftSize / sampleRate
    val tolerance = max(center * (2.0.pow(SEARCH_CENTS / 1200) - 1), 2.0)
    val lo = max(2, floor(center - tolerance).toInt())
    val hi = min(half - 3, ceil(center + tolerance).toInt())
    if (hi <= lo) return false
    var k = lo
    for (i in lo + 1..hi) {
      if (power[i] > power[k]) k = i
    }
    if (k == lo || k == hi || power[k] <= 0) return false
    // The Blackman-Harris main lobe is close to a Gaussian, whose log is a parabola.
    val a = ln(power[k - 1] + 1e-300)
    val b = ln(power[k])
    val c = ln(power[k + 1] + 1e-300)
    val denom = a - 2 * b + c
    if (denom >= 0) return false
    val delta = 0.5 * (a - c) / denom
    if (abs(delta) > 1) return false
    peakFrequency = (k + delta) * sampleRate / fftSize
    peakPower = power[k]
    return true
  }

  companion object {
    const val MAX_PARTIALS = 8
    const val MAX_PARTIAL_FREQUENCY = 5000.0

    /** Search half-width around n times the MPM pitch, which can be ~15 cents sharp on a bass. */
    const val SEARCH_CENTS = 50.0

    /** Partials more than 25 dB below the strongest one are mostly noise. */
    const val STRONG_PARTIAL_RATIO = 0.0031622776601683795

    /**
     * The refined pitch may differ from MPM's by at most this much; anything further is a
     * misread partial, so MPM's pitch is kept.
     */
    const val MAX_CORRECTION_CENTS = 60.0
    const val INHARMONICITY_SMOOTHING = 0.15
    const val MAX_INHARMONICITY = 2e-3

    /** A pitch whose odd partials are 20 dB below its even ones is a subharmonic. */
    const val SUBHARMONIC_ODD_RATIO = 0.01

    /** A pitch change larger than this is another note (or string): its B is learned afresh. */
    const val NOTE_CHANGE_CENTS = 50.0

    /**
     * Below this many periods per window the fundamental's main lobe merges with its mirror
     * image at negative frequency, and MPM's pitch is kept as is.
     */
    const val MIN_PERIODS = 2.5
  }
}
