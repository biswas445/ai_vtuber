"""Speech-to-text wrapper around faster-whisper with a resident GPU model.

The Whisper model is loaded once at startup and kept in VRAM. One dummy
transcription runs right after loading so every CUDA kernel is compiled
then — not in the middle of the user's first words. Each utterance is
transcribed with greedy decoding and no timestamp pass for minimum latency
(endpointing is already handled by the VAD, so no VAD filter is needed
here either). The utterance is handed to Whisper exactly as the microphone
captured it — no filtering, no normalization.
"""

import numpy as np
from faster_whisper import WhisperModel

INPUT_SAMPLE_RATE = 16000


class STT:
    def __init__(
        self,
        model_size: str = "distil-large-v2",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._warmup()

    def _warmup(self) -> None:
        """Transcribe a dummy second of silence so all CUDA kernels compile
        at startup instead of adding seconds to the first real request."""
        try:
            segments, _ = self.model.transcribe(
                np.zeros(INPUT_SAMPLE_RATE, dtype=np.float32),
                language="en",
                beam_size=1,
                without_timestamps=True,
            )
            for _ in segments:
                pass
        except Exception as exc:
            print(f"[stt] warmup failed (first request may be slower): {exc}")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz float32 mono audio to text, unprocessed."""
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
