"""Optional installer step: pre-download every model the voice pipeline uses.

Loads the exact same objects assistant.py builds at startup — the Silero VAD,
the faster-whisper STT model and the pocket-tts TTS model with its voice
conditioning — so every HuggingFace download lands in the local cache during
install and the first launch is instant instead of downloading several GB.

Models are loaded on CUDA when available (matching the app's defaults), on
CPU otherwise — this only warms the download cache, so either device works.
Every stage is independent; a failure just warns and the app finishes that
download on first launch.

Run by install.bat, or manually:  venv\\Scripts\\python.exe install_precache.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / "pocket_tts" / ".env")


def env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def have_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


CUDA = have_cuda()


def pick(device: str) -> str:
    """Resolve a configured device for a one-time load: cuda only when it is
    actually available, otherwise CPU (the download is identical either way)."""
    if device == "cuda" and not CUDA:
        print("[warn] CUDA not available - precaching on CPU, downloads are identical")
        return "cpu"
    return device


def stage(name: str, fn) -> bool:
    t0 = time.monotonic()
    try:
        fn()
    except Exception as exc:
        print(f"[warn] {name} failed ({exc}) - it will download on first launch instead")
        return False
    print(f"[ok] {name} cached ({time.monotonic() - t0:.0f} s)")
    return True


def do_vad() -> None:
    from pocket_tts.vad import StreamingVAD

    StreamingVAD(
        threshold=float(env("VAD_THRESHOLD", "0.5")),
        min_silence_ms=int(env("VAD_MIN_SILENCE_MS", "600")),
        speech_pad_ms=int(env("VAD_SPEECH_PAD_MS", "300")),
        device=env("VAD_DEVICE", "cpu"),
        dtype=env("VAD_DTYPE", "float32"),
    )


def do_stt() -> None:
    from pocket_tts.stt import STT

    STT(
        model_size=env("STT_MODEL", "distil-medium.en"),
        device=pick(env("STT_DEVICE", "cuda")),
        compute_type=env("STT_COMPUTE_TYPE", "float32"),
    )


def do_tts() -> None:
    from pocket_tts.tts import TTS

    TTS(
        voice=env("TTS_VOICE", "azelma"),
        device=pick(env("TTS_DEVICE", "cuda")),
        dtype=env("TTS_DTYPE", "float32"),
    )


def main() -> int:
    print(f"[precache] torch CUDA available: {CUDA}")
    print("[precache] loading every voice-pipeline model (one-time, several GB)...")
    ok = stage("Silero VAD", do_vad)
    ok = stage("Whisper STT", do_stt) and ok
    ok = stage("Pocket TTS + voice", do_tts) and ok
    if ok:
        print("[precache] all models cached - first launch will be instant.")
    else:
        print("[precache] some stages were skipped - the app finishes those on first launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
