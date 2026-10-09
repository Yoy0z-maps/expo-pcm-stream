import Accelerate
import Foundation

/// Refines an MPM pitch to the frequency of the fundamental partial, from the spectrum of the
/// same window.
///
/// MPM measures the period of the whole waveform. On a stiff string (steel guitar and bass
/// strings) the partials run progressively sharp, f_n = n * f0 * sqrt(1 + B * n^2), so the
/// waveform's period settles between the partials' and reads sharp of the fundamental: ~3 cents
/// on a guitar and ~10-15 cents on a bass. It also wobbles by a few cents as the inharmonic
/// partials drift in relative phase. Tuners and the ear go by the fundamental.
///
/// This stage measures partials 1...8 from a Blackman-Harris windowed, zero-padded spectrum,
/// fits the string's inharmonicity coefficient B to them (smoothed over time: it is a property
/// of the string, while a single window's estimate is noisy), and combines every partial into one
/// estimate of partial 1, f_n / n * sqrt((1 + B) / (1 + B * n^2)), weighted by n^2 times the
/// partial's power - the weight each partial carries in the autocorrelation. Using all partials
/// keeps it accurate when the fundamental itself is weak, as a phone microphone hears a bass.
///
/// It also undoes MPM's subharmonic errors: a stiff string's partials can line up better at twice
/// its period than at its period, and a pitch whose odd multiples are empty is really an octave
/// higher.
///
/// Not thread-safe; all memory is preallocated so the per-hop calls never allocate.
final class FundamentalRefiner {
  static let maxPartials = 8
  static let maxPartialFrequency = 5000.0
  /// Search half-width around n times the MPM pitch, which can be ~15 cents sharp on a bass.
  static let searchCents = 50.0
  /// Partials more than 25 dB below the strongest one are mostly noise.
  static let strongPartialRatio = 0.003_162_277_660_168_379_5
  /// The refined pitch may differ from MPM's by at most this much; anything further is a
  /// misread partial, so MPM's pitch is kept.
  static let maxCorrectionCents = 60.0
  static let inharmonicitySmoothing = 0.15
  static let maxInharmonicity = 2e-3
  /// A pitch whose odd partials are 20 dB below its even ones is a subharmonic.
  static let subharmonicOddRatio = 0.01
  /// A pitch change larger than this is another note (or string): its B is learned afresh.
  static let noteChangeCents = 50.0
  /// Below this many periods per window the fundamental's main lobe merges with its mirror image
  /// at negative frequency, and MPM's pitch is kept as is.
  static let minPeriods = 2.5

  let windowSize: Int
  private let fftSize: Int
  private let halfSize: Int
  private let log2n: vDSP_Length
  private let fftSetup: FFTSetupD
  private let taper: UnsafeMutablePointer<Double> // windowSize: 4-term Blackman-Harris
  private let buffer: UnsafeMutablePointer<Double> // fftSize: tapered, zero-padded window
  private let splitReal: UnsafeMutablePointer<Double> // halfSize
  private let splitImag: UnsafeMutablePointer<Double> // halfSize
  private let power: UnsafeMutablePointer<Double> // halfSize: |X[k]|^2 (DC zeroed)
  private let partialNumbers: UnsafeMutablePointer<Int> // maxPartials
  private let partialFrequencies: UnsafeMutablePointer<Double> // maxPartials
  private let partialPowers: UnsafeMutablePointer<Double> // maxPartials
  private var inharmonicity = 0.0
  private var inharmonicityCount = 0
  private var lastPitch = 0.0

  init?(windowSize: Int) {
    guard windowSize >= 4 else { return nil }
    var size = 1
    var log2 = 0
    // 4x zero-padding: on bins this fine the log-parabolic peak interpolation is within
    // ~0.03 cents for a pure tone (2x padding: ~0.2 cents).
    while size < 4 * windowSize {
      size <<= 1
      log2 += 1
    }
    guard let setup = vDSP_create_fftsetupD(vDSP_Length(log2), FFTRadix(kFFTRadix2)) else { return nil }

    self.windowSize = windowSize
    fftSize = size
    halfSize = size / 2
    log2n = vDSP_Length(log2)
    fftSetup = setup
    taper = .allocate(capacity: windowSize)
    buffer = .allocate(capacity: size)
    splitReal = .allocate(capacity: size / 2)
    splitImag = .allocate(capacity: size / 2)
    power = .allocate(capacity: size / 2)
    partialNumbers = .allocate(capacity: FundamentalRefiner.maxPartials)
    partialFrequencies = .allocate(capacity: FundamentalRefiner.maxPartials)
    partialPowers = .allocate(capacity: FundamentalRefiner.maxPartials)

    // Sidelobes at -92 dB: a strong partial can't leak into the measurement of a weak neighbor.
    // The main lobe spans +-4 * sampleRate / W Hz, so cleanly resolved partials take a window of
    // ~4.5 periods of the fundamental (W >= 4.5 * sampleRate / f0). See `minPeriods`.
    for i in 0..<windowSize {
      let x = 2 * Double.pi * Double(i) / Double(windowSize)
      taper[i] = 0.35875 - 0.48829 * cos(x) + 0.14128 * cos(2 * x) - 0.01168 * cos(3 * x)
    }
    (buffer + windowSize).initialize(repeating: 0, count: size - windowSize)
  }

  deinit {
    vDSP_destroy_fftsetupD(fftSetup)
    taper.deallocate()
    buffer.deallocate()
    splitReal.deallocate()
    splitImag.deallocate()
    power.deallocate()
    partialNumbers.deallocate()
    partialFrequencies.deallocate()
    partialPowers.deallocate()
  }

  /// Computes the power spectrum of `window` (`windowSize` samples, already high-passed). Call it
  /// before `correctSubharmonic` and `refine` for each new window.
  func analyze(_ window: UnsafePointer<Double>) {
    vDSP_vmulD(window, 1, taper, 1, buffer, 1, vDSP_Length(windowSize)) // the padding stays 0
    var split = DSPDoubleSplitComplex(realp: splitReal, imagp: splitImag)
    let n2 = vDSP_Length(halfSize)
    buffer.withMemoryRebound(to: DSPDoubleComplex.self, capacity: halfSize) {
      vDSP_ctozD($0, 2, &split, 1, n2)
    }
    vDSP_fft_zripD(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
    // Packed layout: realp[0] is DC and imagp[0] is Nyquist; neither is ever a partial.
    vDSP_zvmagsD(&split, 1, power, 1, n2)
    power[0] = 0
  }

  /// Returns `frequency`, or an octave (two at most) above it if its odd partials are empty.
  func correctSubharmonic(_ frequency: Double, sampleRate: Double, maxFrequency: Double) -> Double {
    var pitch = frequency
    guard resolves(pitch, sampleRate: sampleRate) else { return pitch }
    let top = min(sampleRate * 0.4, FundamentalRefiner.maxPartialFrequency)
    for _ in 0..<2 {
      guard 2 * pitch <= maxFrequency else { break }
      var odd = 0.0
      var even = 0.0
      var n = 1
      while n <= 2 * FundamentalRefiner.maxPartials && Double(n) * pitch <= top {
        if let partial = peak(near: Double(n) * pitch, sampleRate: sampleRate) {
          if n % 2 == 1 { odd += partial.power } else { even += partial.power }
        }
        n += 1
      }
      guard even > 0, odd < even * FundamentalRefiner.subharmonicOddRatio else { break }
      pitch *= 2
    }
    return pitch
  }

  /// Returns the frequency of the fundamental partial near `pitch` (an MPM estimate), or `pitch`
  /// itself when the spectrum doesn't support a refinement.
  func refine(_ pitch: Double, sampleRate: Double) -> Double {
    guard resolves(pitch, sampleRate: sampleRate) else { return pitch }
    if lastPitch == 0 || abs(1200 * log2(pitch / lastPitch)) > FundamentalRefiner.noteChangeCents {
      inharmonicity = 0
      inharmonicityCount = 0
    }
    lastPitch = pitch

    var count = 0
    var strongest = 0.0
    let top = min(sampleRate * 0.4, FundamentalRefiner.maxPartialFrequency)
    for n in 1...FundamentalRefiner.maxPartials {
      let target = pitch * Double(n)
      if target > top { break }
      guard let partial = peak(near: target, sampleRate: sampleRate) else { continue }
      partialNumbers[count] = n
      partialFrequencies[count] = partial.frequency
      partialPowers[count] = partial.power
      count += 1
      strongest = max(strongest, partial.power)
    }
    guard strongest > 0 else { return pitch }

    // Keep the strong partials, with weights n^2 * relative power.
    var strong = 0
    for i in 0..<count where partialPowers[i] >= strongest * FundamentalRefiner.strongPartialRatio {
      let n = Double(partialNumbers[i])
      partialNumbers[strong] = partialNumbers[i]
      partialFrequencies[strong] = partialFrequencies[i]
      partialPowers[strong] = n * n * partialPowers[i] / strongest
      strong += 1
    }

    // Stiff string: (f_n / n)^2 = f0^2 + f0^2 * B * n^2 - a weighted line fit in n^2 gives B.
    if strong >= 3 {
      var sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0
      for i in 0..<strong {
        let n = Double(partialNumbers[i])
        let x = n * n
        let ratio = partialFrequencies[i] / n
        let y = ratio * ratio
        let w = partialPowers[i]
        sw += w
        sx += w * x
        sy += w * y
        sxx += w * x * x
        sxy += w * x * y
      }
      let det = sw * sxx - sx * sx
      if det > 1e-12 * sw * sxx {
        let slope = (sw * sxy - sx * sy) / det
        let intercept = (sy - slope * sx) / sw
        if intercept > 0 {
          inharmonicityCount += 1
          let alpha = max(FundamentalRefiner.inharmonicitySmoothing, 1 / Double(inharmonicityCount))
          inharmonicity += alpha * (slope / intercept - inharmonicity)
        }
      }
    }

    let b = min(max(inharmonicity, 0), FundamentalRefiner.maxInharmonicity)
    var sum = 0.0
    var weights = 0.0
    for i in 0..<strong {
      let n = Double(partialNumbers[i])
      sum += partialPowers[i] * partialFrequencies[i] / n * ((1 + b) / (1 + b * n * n)).squareRoot()
      weights += partialPowers[i]
    }
    let fundamental = sum / weights
    if abs(1200 * log2(fundamental / pitch)) > FundamentalRefiner.maxCorrectionCents { return pitch }
    return fundamental
  }

  private func resolves(_ pitch: Double, sampleRate: Double) -> Bool {
    return pitch * Double(windowSize) >= FundamentalRefiner.minPeriods * sampleRate
  }

  /// The interpolated spectral peak within `searchCents` (at least 2 bins) of `target` Hz, or nil
  /// if the band's maximum sits on its edge (a neighbor's slope, not a peak of its own).
  private func peak(near target: Double, sampleRate: Double) -> (frequency: Double, power: Double)? {
    let center = target * Double(fftSize) / sampleRate
    let tolerance = max(center * (pow(2, FundamentalRefiner.searchCents / 1200) - 1), 2)
    let lo = max(2, Int(floor(center - tolerance)))
    let hi = min(halfSize - 3, Int(ceil(center + tolerance)))
    guard hi > lo else { return nil }
    var k = lo
    for i in (lo + 1)...hi where power[i] > power[k] { k = i }
    guard k > lo, k < hi, power[k] > 0 else { return nil }
    // The Blackman-Harris main lobe is close to a Gaussian, whose log is a parabola.
    let a = log(power[k - 1] + 1e-300)
    let b = log(power[k])
    let c = log(power[k + 1] + 1e-300)
    let denom = a - 2 * b + c
    guard denom < 0 else { return nil }
    let delta = 0.5 * (a - c) / denom
    guard abs(delta) <= 1 else { return nil }
    return ((Double(k) + delta) * sampleRate / Double(fftSize), power[k])
  }
}
