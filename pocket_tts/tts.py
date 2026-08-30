"""Text-to-speech wrapper around Kyutai Pocket TTS with a resident model.

The model and the voice conditioning state are built once at startup and kept
resident (never reloaded per request). `synthesize()` returns the full audio
for a sentence in the cloned voice of the reference clip (see
`get_state_for_audio_prompt` in pocket_tts/models/tts_model.py).

Quality path, in order:
  1. the text is cleaned down to plain spoken language (LLMs leak emoji,
     markdown and stray symbols even when told not to, and the TTS model
     audibly chokes on them);
  2. dead edge silence is trimmed;
  3. click-free raised-cosine fades seal both ends;
  4. loudness is normalized to a fixed RMS target (peak-capped), so every
     sentence plays at the same perceived volume instead of some whispering
     and some blasting.

The voice comes entirely from the reference audio conditioning — the pitch of
the output tracks the reference speaker, so no post-shift is applied.
"""

import re
from pathlib import Path

import numpy as np
import torch

from pocket_tts import TTSModel

TARGET_RMS = 0.22  # consistent perceived loudness across sentences (~-13 dBFS)
OUTPUT_PEAK = 0.95  # hard ceiling — never clip

# Voice conditioning sources the model can consume: a preset name, an hf://
# or http(s):// URL, or a local audio file in any format soundfile reads.
VOICE_AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aiff", ".aif")

# The LLM is told to answer in plain words, but models still leak symbols —
# emoji, markdown, decoration — and the TTS model mispronounces or glitches
# on those. Everything below reduces its output to clean spoken sentences.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictographs, emoji, symbols & supplemental
    "\U00002600-\U000027BF"  # misc symbols / dingbats
    "\U0001F1E6-\U0001F1FF"  # regional flags
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"  # zero-width joiner
    "\U000020E3"
    "]+",
    flags=re.UNICODE,
)
_STAGE_DIR_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # [laughs], (sighs), ...
_MARKDOWN_RE = re.compile(r"[*_`#>|~\[\]]+")
_NON_SPEECH_RE = re.compile(r"[^\w\s.'\"!?…,%+-]", flags=re.UNICODE)


def clean_text(text: str) -> str:
    """Reduce LLM output to clean spoken sentences for the TTS model."""
    text = _EMOJI_RE.sub(" ", text)
    # Stage directions in brackets are never spoken — drop them whole.
    text = _STAGE_DIR_RE.sub(" ", text)
    text = _MARKDOWN_RE.sub(" ", text)
    text = text.replace("&", " and ")
    text = text.replace("%", " percent ")
    # A wall of "!!!" or "..." should be one mark, not acted out per glyph.
    text = re.sub(r"([.!?…]){2,}", r"\1", text)
    text = re.sub(r"\.{3}", "…", text)
    text = _NON_SPEECH_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_fades(audio: np.ndarray, sample_rate: int, fade_ms: float = 8.0) -> np.ndarray:
    """Short raised-cosine fade at both ends so a clip can never click/pop
    at its boundaries when playback starts or stops."""
    n = int(fade_ms * sample_rate / 1000)
    if audio.size <= 2 * n + 1:
        return audio
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


def normalize_loudness(audio: np.ndarray) -> np.ndarray:
    """Normalize to a fixed RMS target so every sentence plays at the same
    perceived volume; the gain is capped by the peak so nothing clips."""
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 1e-6:
        return audio
    gain = TARGET_RMS / rms
    peak = float(np.max(np.abs(audio)))
    if peak * gain > OUTPUT_PEAK:
        gain = OUTPUT_PEAK / max(peak, 1e-9)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def trim_silence(
    audio: np.ndarray, sample_rate: int, threshold: float = 0.01, pad_ms: float = 60.0
) -> np.ndarray:
    """Cut the dead silence a TTS model sometimes leaves at the edges of a
    clip, keeping a small natural pad — tightens the conversational pacing."""
    if audio.size == 0:
        return audio
    loud = np.nonzero(np.abs(audio) > threshold)[0]
    if loud.size == 0:
        return audio
    pad = int(pad_ms * sample_rate / 1000)
    start = max(0, loud[0] - pad)
    end = min(audio.size, loud[-1] + pad + 1)
    return audio[start:end]


def _log_reference_transcript(folder: Path) -> None:
    """Read a companion reference .txt next to the voice audio, for the log.

    The model clones from audio only — there is no transcript input (see
    get_state_for_audio_prompt) — so this is informational: it proves the
    reference pair was found and shows what the clip says.
    """
    txts = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".txt")
    if not txts:
        return
    try:
        transcript = txts[0].read_text(encoding="utf-8", errors="replace").strip()
        preview = " ".join(transcript.split())[:80]
        print(f"[tts] reference transcript: {preview}{'...' if len(transcript) > 80 else ''}")
    except OSError as exc:
        print(f"[tts] could not read reference transcript {txts[0]}: {exc}")


def resolve_voice(voice: str | Path) -> str:
    """Normalize TTS_VOICE into something `get_state_for_audio_prompt` accepts.

    Accepts a preset voice name ("azelma"), an hf:// or http(s):// URL, a
    local audio file (any format soundfile reads: wav/mp3/flac/ogg/...), or a
    local FOLDER holding one reference clip — in that case the first audio
    file inside is used and a companion reference .txt (if present) is read
    for the startup log. The model clones from audio only; the .txt is
    informational (see get_state_for_audio_prompt — there is no
    transcript input).
    """
    voice_str = str(voice)
    if voice_str.startswith(("hf://", "http://", "https://")):
        return voice_str
    path = Path(voice_str)
    if path.is_dir():
        audio_files = sorted(
            f for f in path.iterdir() if f.suffix.lower() in VOICE_AUDIO_SUFFIXES
        )
        if not audio_files:
            raise ValueError(
                f"TTS_VOICE folder {voice_str} has no audio file "
                f"({'/'.join(VOICE_AUDIO_SUFFIXES)} expected)"
            )
        audio = audio_files[0]
        if len(audio_files) > 1:
            print(f"[tts] voice folder has {len(audio_files)} audio files, using {audio.name}")
        _log_reference_transcript(path)
        print(f"[tts] cloning voice from {audio}")
        return str(audio)
    if path.is_file():
        _log_reference_transcript(path.parent)
        return voice_str
    # Not a local path: a preset name the model resolves itself (or an
    # invalid one — let the model raise its clear catalog error).
    return voice_str


class TTS:
    def __init__(
        self,
        voice: str | Path = "azelma",
        device: str = "cuda",
        language: str | None = None,
        dtype: str | None = None,
    ):
        from pocket_tts.utils.utils import resolve_dtype

        self.model = TTSModel.load_model(language=language)
        resolved = resolve_dtype(dtype)
        if resolved is not None:
            # Converts every submodule (flow_lm, mimi, buffers) in one go.
            self.model = self.model.to(resolved)
        self.model.to(torch.device(device))
        self.sample_rate = self.model.sample_rate
        # Voice cloning: `voice` can be a preset name, an hf:// or http(s)://
        # URL, a local audio file (wav/mp3/flac/ogg/...), or a folder holding
        # one reference clip — see resolve_voice. The voice state is built
        # once here and deep-copied per sentence, so the reference is never
        # re-read. Reading non-WAV references needs the soundfile package
        # (the model's `audio` extra).
        self.voice_state = self.model.get_state_for_audio_prompt(resolve_voice(voice))
        self.output_sample_rate = self.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        """Generate the full audio for a text as a single loud float32 array.

        Post-processing for clean, smooth output: the text is reduced to
        spoken language, dead edge silence is trimmed, the clip gets
        click-free fades, and the loudness is RMS-normalized.
        """
        text = clean_text(text)
        if not text:
            return np.zeros(0, dtype=np.float32)
        chunks = []
        for chunk in self.model.generate_audio_stream(self.voice_state, text):
            chunks.append(chunk.squeeze().clamp(-1.0, 1.0).cpu().numpy())
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks).astype(np.float32)
        audio = trim_silence(audio, self.sample_rate)
        audio = apply_fades(audio, self.sample_rate)
        audio = normalize_loudness(audio)
        return audio
