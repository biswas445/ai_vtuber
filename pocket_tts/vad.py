"""Streaming voice activity detection on top of Silero VAD, plus a
speech-vs-noise verdict so only real human speech ever reaches the STT.

The Silero model is loaded once at startup and kept resident. Mic audio is
fed in fixed 32 ms chunks (512 samples @ 16 kHz) via `feed()`, which scores
every chunk with the neural VAD and does the start/end endpointing right
here (the library's VADIterator is deliberately not used because it hides
the per-chunk speech probability, which the verdict needs). Raw audio is
ring-buffered so the full utterance (including a bit of pre-roll) can be
extracted when speech ends.

Every finished utterance comes with stats: its overall energy, the ambient
noise floor tracked while listening, how much of it the neural model
actually scored as speech, and how much of its energy sits in the human
voice band. `speech_verdict()` turns those numbers into one accept/reject
decision — silence, hum, fans, keyboard bangs and other non-speech get a
rejection reason and never reach the STT.
"""

import collections

import numpy as np
import torch
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32 ms window, required by Silero
CHUNK_MS = 1000 * CHUNK_SAMPLES / SAMPLE_RATE
MIN_PRE_ROLL_CHUNKS = 20  # keep at least ~0.64 s before the speech start
MAX_SPEECH_CHUNKS = 470  # force end-of-utterance after ~15 s of continuous speech

# --- speech verdict ---------------------------------------------------------
# Human voice energy lives almost entirely between these frequencies; a burst
# whose spectrum sits elsewhere (mains hum, fan hiss, keyboard clicks, a music
# thump) is not speech.
SPEECH_BAND_HZ = (80.0, 4000.0)
# At least this fraction of the utterance's power must sit in the voice band.
MIN_SPEECH_BAND_FRACTION = 0.55
# Absolute energy floor (RMS of float32 samples): below this, nothing audible
# was captured at all.
MIN_UTTERANCE_RMS = 0.004
# A real spoken word — even a very short one — keeps the neural model
# convinced for at least a few consecutive chunks; a bang or a click spikes
# for one or two and is gone. ~130 ms of neurally-confirmed voice is the
# floor.
MIN_VOICED_CHUNKS = 4


def speech_band_fraction(audio: np.ndarray) -> float:
    """Fraction (0..1) of the signal's power inside the human voice band."""
    if audio.size < CHUNK_SAMPLES:
        return 0.0
    power = np.abs(np.fft.rfft(audio)) ** 2
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / SAMPLE_RATE)
    lo, hi = SPEECH_BAND_HZ
    return float(power[(freqs >= lo) & (freqs <= hi)].sum()) / total


def speech_verdict(
    stats: dict,
    min_voiced_fraction: float = 0.35,
    noise_margin: float = 3.0,
) -> str | None:
    """None when the captured utterance is genuine human speech, otherwise a
    short rejection reason. Checks, in order:

    1. adaptive energy floor — the burst must clearly rise above the ambient
       noise floor tracked while listening (and above an absolute minimum),
       so a quiet room or a bad mic never counts as speech;
    2. voiced amount — the neural model must have scored enough of the burst
       as speech (both an absolute floor and a fraction of it), which kills
       door slams and keyboard bangs that trip the endpointer;
    3. speech-band gate — the energy must concentrate where a human voice
       lives, which kills hum, hiss and thumps.
    """
    floor = max(MIN_UTTERANCE_RMS, stats["noise_rms"] * noise_margin)
    if stats["rms"] < floor:
        return "below the energy floor"
    if stats["voiced_chunks"] < MIN_VOICED_CHUNKS:
        return f"only {stats['voiced_chunks']} voiced chunks"
    if stats["voiced_fraction"] < min_voiced_fraction:
        return f"voiced fraction {stats['voiced_fraction']:.2f} too low"
    if stats["speech_band_fraction"] < MIN_SPEECH_BAND_FRACTION:
        return f"speech-band fraction {stats['speech_band_fraction']:.2f} too low"
    return None


class StreamingVAD:
    """Streaming endpointer + speech gate. See the module docstring."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 600,
        speech_pad_ms: int = 300,
        device: str = "cpu",
        dtype: str | None = None,
    ):
        from pocket_tts.utils.utils import resolve_dtype

        resolved = resolve_dtype(dtype)
        if resolved is not None and resolved != torch.float32:
            # Silero ships as a TorchScript module whose internal STFT
            # buffers stay fp32 no matter what dtype the module is cast
            # to — anything but fp32 crashes in conv1d on the first chunk.
            raise ValueError(
                f"VAD_DTYPE must be float32: silero's TorchScript model "
                f"cannot run at {resolved}"
            )
        self.device = torch.device(device)
        self.model = load_silero_vad().to(self.device)
        self.threshold = threshold
        # Hysteresis: LEAVING speech needs a clearly low score (same 0.15 gap
        # the library's endpointer uses), so one weak chunk mid-word never
        # cuts the utterance in half.
        self.neg_threshold = max(threshold - 0.15, 0.01)
        self.min_silence_samples = int(SAMPLE_RATE * min_silence_ms / 1000)
        # Pre-roll kept before the detected speech start so soft onsets are
        # never clipped: the configured pad, or ~0.64 s, whichever is more.
        pad_chunks = int(round(speech_pad_ms / CHUNK_MS))
        self.pre_roll_chunks = max(MIN_PRE_ROLL_CHUNKS, pad_chunks)

        self._buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.pre_roll_chunks + MAX_SPEECH_CHUNKS + 8
        )
        self._triggered = False
        self._total_chunks = 0  # every chunk while triggered (for the cap)
        self._speech_chunks = 0  # burst chunks, excl. the silent tail
        self._voiced_chunks = 0  # of those, scored >= threshold
        self._silence_samples = 0  # accumulated while prob < neg_threshold
        self._utterance: np.ndarray | None = None
        self._stats: dict | None = None
        # Ambient noise floor: EMA of the chunk RMS while no speech is going.
        self.noise_rms = 0.0

    def reset(self) -> None:
        """Clear VAD state and buffers (call when re-arming the mic)."""
        self.model.reset_states()
        self._buffer.clear()
        self._triggered = False
        self._total_chunks = 0
        self._speech_chunks = 0
        self._voiced_chunks = 0
        self._silence_samples = 0
        self._utterance = None
        self._stats = None

    @torch.no_grad()
    def feed(self, chunk: np.ndarray) -> str | None:
        """Feed 512 float32 samples. Returns "start", "end", or None."""
        self._buffer.append(chunk)
        prob = float(
            self.model(torch.from_numpy(chunk).to(self.device), SAMPLE_RATE).item()
        )
        rms = float(np.sqrt(np.mean(np.square(chunk))))

        # Track the ambient level from chunks that are not speech: a slow EMA,
        # frozen while a burst is being recorded so the floor reflects the
        # room BEFORE the user spoke.
        if not self._triggered and prob < self.threshold:
            if self.noise_rms <= 0.0:
                self.noise_rms = rms
            else:
                self.noise_rms += (rms - self.noise_rms) * 0.08

        if prob >= self.threshold and not self._triggered:
            self._triggered = True
            self._total_chunks = 1
            # The triggering chunk is voiced by definition.
            self._speech_chunks = 1
            self._voiced_chunks = 1
            self._silence_samples = 0
            # Trim the ring buffer down to just the pre-roll for this burst.
            while len(self._buffer) > self.pre_roll_chunks:
                self._buffer.popleft()
            return "start"

        if self._triggered:
            self._total_chunks += 1
            if prob >= self.threshold:
                # Voiced: counts toward the burst; a pending silence is over.
                self._speech_chunks += 1
                self._voiced_chunks += 1
                self._silence_samples = 0
            elif prob < self.neg_threshold:
                # Clearly not speech: the silence run grows (a run long
                # enough ends the utterance below). These chunks are NOT
                # counted, so the silent tail can't dilute the voiced stats.
                self._silence_samples += CHUNK_SAMPLES
            else:
                # Between the thresholds: hysteresis keeps this chunk inside
                # the burst (it is not the silent tail), but the model did
                # not confirm it as voice — count it toward the burst length
                # only, so voiced_fraction is a real ratio instead of 1.0.
                self._speech_chunks += 1

            forced = self._total_chunks >= MAX_SPEECH_CHUNKS
            if self._silence_samples >= self.min_silence_samples or forced:
                self._triggered = False
                audio = np.concatenate(list(self._buffer))
                self._utterance = audio
                self._stats = {
                    "rms": float(np.sqrt(np.mean(np.square(audio)))),
                    "voiced_chunks": self._voiced_chunks,
                    "voiced_fraction": self._voiced_chunks / max(1, self._speech_chunks),
                    "noise_rms": self.noise_rms,
                    "speech_band_fraction": speech_band_fraction(audio),
                }
                self._buffer.clear()
                return "end"

        return None

    def take_utterance(self) -> tuple[np.ndarray | None, dict | None]:
        """Return (audio, stats) for the finished utterance, or (None, None).

        audio is everything captured since speech start (16 kHz float32
        mono); stats carries the numbers `speech_verdict()` decides from.
        """
        utterance, stats = self._utterance, self._stats
        self._utterance = None
        self._stats = None
        return utterance, stats
