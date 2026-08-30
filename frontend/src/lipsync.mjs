/**
 * Offline lip-sync analysis for TTS clips.
 *
 * The renderer used to drive the mouth from the live analyser with every
 * viseme weight scaled by loudness — the jaw literally tracked the volume.
 * Real speech doesn't work that way: how far the mouth opens is decided by
 * WHICH sound is being said (an "aa" opens wide, an "ee" stays nearly
 * closed, an "m" closes completely), not by how loud it is. Loudness only
 * gates voicing and adds a whisper of prosodic emphasis.
 *
 * Because a clip is fully known before it plays (the backend hands over a
 * complete WAV), we can analyze it OFFLINE with lookahead: every frame is
 * classified in (F1, F2) formant space into the five VRM vowel shapes,
 * brief energy drops (m/b/p/t/k closures) close the mouth, sibilants get a
 * narrow teeth-together shape, and the result is smoothed symmetrically so
 * the mouth glides into the NEXT shape the way real articulation does —
 * something a causal realtime analyser can never do.
 *
 * Pure functions, no DOM dependency — unit-testable under node.
 */

/* --------------------------------------------------------------- vowels --- */

/** Vowel targets in (F1, F2) formant space, Hz, raised slightly for a
 * feminine/anime voice. */
export const VOWEL_CENTERS = {
  aa: { f1: 750, f2: 1300 }, // open "father"
  ih: { f1: 450, f2: 1900 }, // "bit"
  ee: { f1: 300, f2: 2300 }, // "see"
  oh: { f1: 500, f2: 850 }, // "go"
  ou: { f1: 320, f2: 650 }, // "you"
};

/** Intrinsic jaw opening of each vowel shape (0..1). This is what decouples
 * the mouth from loudness: the opening comes from the vowel identity. */
export const VOWEL_OPEN = { aa: 1.0, oh: 0.85, ih: 0.55, ee: 0.4, ou: 0.35 };

export const VISEME_NAMES = ['aa', 'ih', 'ee', 'oh', 'ou'];

/* ------------------------------------------------------------ wav parse --- */

/** Decode a `data:audio/wav;base64,...` URI (16-bit PCM, what the backend's
 * scipy writer produces) into float samples. Returns null when the URI is
 * not a parseable PCM WAV so the caller can fall back to realtime analysis. */
export function parseWavDataUri(uri) {
  if (typeof uri !== 'string') return null;
  const comma = uri.indexOf(',');
  if (comma < 0 || !/^data:audio\/(?:wav|x-wav|wave);/i.test(uri)) return null;
  const b64 = uri.slice(comma + 1);
  let bin;
  try {
    bin = atob(b64);
  } catch {
    return null;
  }
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

  const view = new DataView(bytes.buffer);
  if (bytes.length < 44 || readTag(bytes, 0) !== 'RIFF' || readTag(bytes, 8) !== 'WAVE') {
    return null;
  }
  let fmt = null;
  let dataOff = -1;
  let dataLen = 0;
  let off = 12;
  while (off + 8 <= bytes.length) {
    const id = readTag(bytes, off);
    const size = view.getUint32(off + 4, true);
    const body = off + 8;
    if (id === 'fmt ') {
      fmt = {
        format: view.getUint16(body, true),
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        bits: view.getUint16(body + 14, true),
      };
    } else if (id === 'data') {
      dataOff = body;
      dataLen = Math.min(size, bytes.length - body);
    }
    off = body + size + (size & 1); // chunks are word-aligned
  }
  if (!fmt || fmt.format !== 1 || fmt.bits !== 16 || dataOff < 0) return null;
  const ch = Math.max(1, fmt.channels);
  const frames = Math.floor(dataLen / (2 * ch));
  const samples = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let c = 0; c < ch; c++) {
      sum += view.getInt16(dataOff + (i * ch + c) * 2, true);
    }
    samples[i] = sum / ch / 32768;
  }
  return { samples, sampleRate: fmt.sampleRate };
}

function readTag(bytes, off) {
  return String.fromCharCode(bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]);
}

/* ------------------------------------------------------------------ fft --- */

const FFT_SIZE = 1024;
const _cosTable = new Float32Array(FFT_SIZE / 2);
const _sinTable = new Float32Array(FFT_SIZE / 2);
for (let i = 0; i < FFT_SIZE / 2; i++) {
  _cosTable[i] = Math.cos((-2 * Math.PI * i) / FFT_SIZE);
  _sinTable[i] = Math.sin((-2 * Math.PI * i) / FFT_SIZE);
}
const _re = new Float32Array(FFT_SIZE);
const _im = new Float32Array(FFT_SIZE);
const _hann = new Float32Array(FFT_SIZE);
for (let i = 0; i < FFT_SIZE; i++) {
  _hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)));
}

/** In-place iterative radix-2 FFT of _re/_im (magnitude written to mag). */
function fftMag(mag) {
  const n = FFT_SIZE;
  // Bit reversal.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = _re[i];
      _re[i] = _re[j];
      _re[j] = t;
      t = _im[i];
      _im[i] = _im[j];
      _im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const idx = k * step;
        const wr = _cosTable[idx];
        const wi = _sinTable[idx];
        const a = i + k;
        const b = a + half;
        const tr = _re[b] * wr - _im[b] * wi;
        const ti = _re[b] * wi + _im[b] * wr;
        _re[b] = _re[a] - tr;
        _im[b] = _im[a] - ti;
        _re[a] += tr;
        _im[a] += ti;
      }
    }
  }
  for (let k = 0; k < n / 2; k++) {
    // 2|X|/N so a full-scale sine reads ~1.0.
    mag[k] = (2 * Math.hypot(_re[k], _im[k])) / n;
  }
}

/* -------------------------------------------------------------- analysis --- */

function bandMean(mag, binHz, f0, f1, nBins) {
  const i0 = Math.max(1, Math.floor(f0 / binHz));
  const i1 = Math.min(nBins - 1, Math.ceil(f1 / binHz));
  if (i1 < i0) return 0;
  let sum = 0;
  for (let i = i0; i <= i1; i++) sum += mag[i];
  return sum / (i1 - i0 + 1);
}

/** All local maxima of the 3-bin-smoothed spectrum in [f0, f1] that rise
 * above `floor`, refined to sub-bin accuracy with a parabolic fit, sorted
 * by frequency. */
function spectralPeaks(mag, binHz, f0, f1, floor) {
  const n = mag.length;
  const i0 = Math.max(2, Math.floor(f0 / binHz));
  const i1 = Math.min(n - 3, Math.ceil(f1 / binHz));
  const peaks = [];
  for (let i = i0; i <= i1; i++) {
    const prev = (mag[i - 2] + mag[i - 1] + mag[i]) / 3;
    const v = (mag[i - 1] + mag[i] + mag[i + 1]) / 3;
    const next = (mag[i] + mag[i + 1] + mag[i + 2]) / 3;
    if (v < floor || v < prev || v < next) continue;
    const denom = prev - 2 * v + next;
    const off = denom !== 0 ? (0.5 * (prev - next)) / denom : 0;
    peaks.push({ f: (i + off) * binHz, v });
  }
  return peaks;
}

/** Classify one frame's vowel from its spectral peaks. Two hypotheses cover
 * the two ways peak-picking goes wrong: (A) the STRONGEST peak below 1.1 kHz
 * is F1 — right for open vowels, where the F1 cluster dominates the
 * spectrum; (B) the LOWEST prominent peak is F1 — right for close vowels
 * (ee/ou), where F1 is weak and the strongest peak is actually F2. The
 * hypothesis that best matches the vowel space wins. `meanLevel` is the
 * band-average magnitude: noise-like frames (sibilants, breath) have no
 * peaks much taller than it and are rejected instead of being read as
 * random vowels. Returns Gaussian scores per vowel, or null. */
function classifyVowel(peaks, meanLevel) {
  if (!peaks.length) return null;
  let tallest = 0;
  for (const p of peaks) tallest = Math.max(tallest, p.v);
  if (tallest < meanLevel * 2.5) return null; // flat spectrum — not harmonic
  const low = peaks.filter((p) => p.f <= 1100);
  if (!low.length) return null;
  const strongest = low.reduce((a, b) => (b.v > a.v ? b : a));
  const lowest = low.find((p) => p.v >= strongest.v * 0.25);
  const f2Of = (f1, gap) => {
    const hi = peaks.filter((p) => p.f >= f1 + gap);
    return hi.length ? hi.reduce((a, b) => (b.v > a.v ? b : a)).f : null;
  };
  const hyps = [];
  const f2A = f2Of(strongest.f, 250);
  if (f2A !== null) hyps.push([strongest.f, f2A]);
  if (lowest && Math.abs(lowest.f - strongest.f) > 100) {
    const f2B = f2Of(lowest.f, 150);
    if (f2B !== null) hyps.push([lowest.f, f2B]);
  }
  let bestScores = null;
  let bestScore = 0.05; // below this the frame matches no vowel — unvoiced mush
  for (const [f1, f2] of hyps) {
    const scores = {};
    let best = 0;
    for (const [name, c] of Object.entries(VOWEL_CENTERS)) {
      const d1 = (f1 - c.f1) / 260;
      const d2 = (f2 - c.f2) / 550;
      const sc = Math.exp(-(d1 * d1 + d2 * d2));
      scores[name] = sc;
      if (sc > best) best = sc;
    }
    if (best > bestScore) {
      bestScore = best;
      bestScores = scores;
    }
  }
  return bestScores;
}

function centeredAverage(src, radius) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    let sum = 0;
    let n = 0;
    for (let k = -radius; k <= radius; k++) {
      const j = i + k;
      if (j >= 0 && j < src.length) {
        sum += src[j];
        n++;
      }
    }
    out[i] = sum / n;
  }
  return out;
}

/**
 * Build the per-frame viseme timeline for a clip.
 *
 * Returns `{ hopSec, duration, jaw, vis }` where `jaw` and each `vis[name]`
 * are Float32Arrays (one value per analysis frame, 0..1), or null when the
 * clip is too short to analyze.
 */
export function buildVisemeTimeline(samples, sampleRate) {
  const hop = Math.max(64, Math.round((sampleRate * 10) / 1000)); // ~10 ms hop
  const nFrames = Math.floor((samples.length - FFT_SIZE) / hop) + 1;
  if (nFrames < 4) return null;

  const binHz = sampleRate / FFT_SIZE;
  const mag = new Float32Array(FFT_SIZE / 2);
  const rms = new Float32Array(nFrames); // hop-sized (fast) — voicing gate
  const rmsWin = new Float32Array(nFrames); // window-sized — spectral floor

  // Pass 1: per-frame RMS + spectrum features.
  const feat = new Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    const start = f * hop;
    let sum = 0;
    let sumWin = 0;
    for (let i = 0; i < FFT_SIZE; i++) {
      const x = samples[start + i];
      sumWin += x * x;
      _re[i] = x * _hann[i];
      _im[i] = 0;
      if (i < hop) sum += x * x;
    }
    rms[f] = Math.sqrt(sum / hop);
    rmsWin[f] = Math.sqrt(sumWin / FFT_SIZE);
    fftMag(mag);
    const eVoice = bandMean(mag, binHz, 70, 1700, mag.length);
    const eSib = bandMean(mag, binHz, 4000, Math.min(8000, sampleRate / 2 - 200), mag.length);
    // Spectral peak floor scales with the frame's own energy so the formant
    // search never locks onto noise in near-silent frames.
    const peakFloor = Math.max(0.004, rmsWin[f] * 0.06);
    const peaks = spectralPeaks(mag, binHz, 120, 3200, peakFloor);
    const meanLevel = bandMean(mag, binHz, 120, 3200, mag.length);
    feat[f] = { eVoice, eSib, scores: classifyVowel(peaks, meanLevel) };
  }

  // Voicing gate: an adaptive floor a hair below the clip's peak, so quiet
  // speech still counts as voiced but true silence (and plosive closures)
  // close the mouth.
  let peak = 0;
  for (let f = 0; f < nFrames; f++) peak = Math.max(peak, rms[f]);
  const floor = Math.max(peak * 0.05, 0.0012);
  const voiced = new Float32Array(nFrames);
  for (let f = 0; f < nFrames; f++) voiced[f] = rms[f] > floor ? 1 : 0;

  // Pass 2: classify each voiced frame in (F1, F2) space. Loudness only
  // saturates a gate — it does NOT scale the opening.
  const raw = {};
  for (const name of VISEME_NAMES) raw[name] = new Float32Array(nFrames);
  const rawJaw = new Float32Array(nFrames);
  let lastW = null; // carry the last classification across unresolvable frames
  for (let f = 0; f < nFrames; f++) {
    if (!voiced[f]) {
      lastW = null;
      continue;
    }
    const { eVoice, eSib, scores } = feat[f];
    // Openness gate: fully open once the frame is a few times above the
    // noise floor — a threshold, not a proportionality.
    const open = Math.min(1, Math.max(0, (rms[f] - floor) / (floor * 2)));
    let w;
    if (eSib > eVoice * 0.55 && eSib > peak * 0.008) {
      // S/sh/f: teeth nearly together, a small narrow opening. Strength is
      // read from the hiss-to-voice RATIO (gain-independent), not the
      // absolute hiss level.
      const s = Math.min(1, eSib / (eVoice + 1e-6) / 4) * 0.6;
      w = { aa: 0, ih: 0.5, ee: 0.42, oh: 0, ou: 0 };
      for (const name of VISEME_NAMES) raw[name][f] = w[name] * s * open;
      rawJaw[f] = 0.14 * s * open;
      lastW = null;
      continue;
    }
    if (scores) {
      // Sharpened softmax so the closest vowel clearly wins.
      let best = 0;
      for (const name of VISEME_NAMES) if (scores[name] > best) best = scores[name];
      let total = 0;
      w = {};
      for (const name of VISEME_NAMES) {
        const sc = Math.pow(scores[name] / (best || 1), 2.5);
        w[name] = sc;
        total += sc;
      }
      for (const name of VISEME_NAMES) w[name] /= total || 1;
      lastW = w;
    } else {
      w = lastW; // mid-plosive or unresolvable frame: hold the last vowel
    }
    if (!w) continue;
    let jaw = 0;
    for (const name of VISEME_NAMES) {
      raw[name][f] = w[name] * open;
      jaw += w[name] * VOWEL_OPEN[name];
    }
    rawJaw[f] = Math.min(1, jaw * 1.15) * open;
  }

  // Pass 3: symmetric (zero-phase) smoothing — the mouth glides INTO the
  // next shape because the filter sees the future. ~50 ms for the viseme
  // mix, a touch shorter for the jaw; the voicing gate is smoothed over a
  // short window so brief plosive closures still snap the mouth shut.
  const gate = centeredAverage(voiced, 1);
  const jaw = centeredAverage(rawJaw, 1);

  // Prosodic emphasis: stressed syllables open a little more, but only by a
  // bounded ±15% — the envelope modulates, it never drives.
  const env = centeredAverage(rms, 4);
  let envPeak = 0;
  const mod = new Float32Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    envPeak = Math.max(env[f], envPeak * 0.995, floor);
    const norm = Math.min(1, Math.pow(env[f] / (envPeak * 0.6), 0.8));
    mod[f] = 0.85 + 0.3 * norm;
  }

  const vis = {};
  for (const name of VISEME_NAMES) {
    const sm = centeredAverage(raw[name], 2);
    const out = new Float32Array(nFrames);
    for (let f = 0; f < nFrames; f++) {
      const v = sm[f] * gate[f] * mod[f];
      out[f] = v < 0.004 ? 0 : Math.min(1, v);
    }
    vis[name] = out;
  }
  const jawOut = new Float32Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    const v = jaw[f] * gate[f] * mod[f];
    jawOut[f] = v < 0.004 ? 0 : Math.min(1, v);
  }

  return {
    hopSec: hop / sampleRate,
    duration: samples.length / sampleRate,
    jaw: jawOut,
    vis,
  };
}

/** Sample the timeline at playback time `t` (seconds) with linear
 * interpolation. Past the end of the clip everything reads closed. */
export function sampleTimeline(tl, t) {
  const out = { jaw: 0, visemes: { aa: 0, ih: 0, ee: 0, oh: 0, ou: 0 } };
  if (!tl || t < 0) return out;
  const pos = t / tl.hopSec;
  const i0 = Math.floor(pos);
  if (i0 >= tl.jaw.length - 1) return out;
  const i = Math.max(0, i0);
  const frac = Math.min(1, Math.max(0, pos - i));
  const j = i + 1;
  out.jaw = tl.jaw[i] + (tl.jaw[j] - tl.jaw[i]) * frac;
  for (const name of VISEME_NAMES) {
    const ch = tl.vis[name];
    out.visemes[name] = ch[i] + (ch[j] - ch[i]) * frac;
  }
  return out;
}
