"""Realtime voice assistant runtime: mic -> VAD -> STT -> LLM -> TTS -> speakers.

All models are loaded once at startup and kept resident (never reloaded per
request). Latency is minimized by pipelining: the LLM streams tokens, complete
sentences are handed to TTS immediately, and each sentence is synthesized
(with the cloned reference voice) and played while the model keeps generating the
rest.

Echo safety: the mic is hard-gated. Frames are only captured while the
assistant is actively listening; while it is thinking or speaking the input
callback drops everything, so the voice coming out of the speakers can never
feed back into the pipeline. In bridge mode the assistant additionally waits
for the overlay's `tts-done` event (fired the instant the character's mouth
actually stops moving) before re-arming the mic, plus a short decay window,
so speaker sound can never leak into the next listening phase. Every burst
the VAD endpointer completes is judged by the speech-verdict gate (voiced
fraction + noise-floor margin, both configurable via pocket_tts/.env), and
known no-speech transcriptions are discarded before the LLM ever sees
them. Nothing is ever written to disk.

Run:  python assistant.py            (standalone, plays through speakers)
Run:  BRIDGE=1 python assistant.py   (spawned by the Electron overlay:
      stdout becomes a JSON-lines protocol of state events and TTS clips
      for the character to lip-sync; logging moves to stderr)
"""

import base64
import io
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from scipy.io import wavfile

from pocket_tts.LLM import LLM
from pocket_tts.stt import STT
from pocket_tts.tts import TTS
from pocket_tts.vad import CHUNK_SAMPLES, SAMPLE_RATE, StreamingVAD, speech_verdict

load_dotenv(Path(__file__).parent / "pocket_tts" / ".env")

# Bridge mode: the Electron overlay spawns us with BRIDGE=1. stdout becomes
# a JSON-lines protocol carrying state changes and TTS clips to the
# character (who lip-syncs them); all logging moves to stderr so it can
# never corrupt the protocol. Without BRIDGE we stay a standalone terminal
# assistant playing through the local speakers.
BRIDGE = os.environ.get("BRIDGE") == "1"

if BRIDGE:
    _BRIDGE_OUT = sys.stdout
    sys.stdout = sys.stderr

# emit() is called from the main loop and from the reply producer thread —
# the lock keeps two JSON events from interleaving on stdout.
_EMIT_LOCK = threading.Lock()


def emit(kind: str, value) -> None:
    """Send one protocol event to the overlay (no-op in standalone mode)."""
    if not BRIDGE:
        return
    try:
        with _EMIT_LOCK:
            _BRIDGE_OUT.write(json.dumps({"type": kind, "value": value}) + "\n")
            _BRIDGE_OUT.flush()
    except Exception:
        # The overlay is gone (broken pipe). Die with it instead of running
        # on as an invisible orphan holding the microphone and GPU memory —
        # nothing would ever read our events or feed us config again.
        os._exit(1)


def wav_data_uri(audio: np.ndarray, sample_rate: int) -> str:
    """Encode float32 audio as a 16-bit WAV data URI for the overlay's <audio>."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, pcm)
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

GREETING = (
    "[smug] Ugh, look who woke me up. I was busy taking over the world. This better be good."
)

SENTENCE_END = re.compile(r'(?<=[.!?…])["\')\]]*\s+|\n+')
MIN_UTTERANCE_S = 0.3  # shorter bursts are treated as noise
MAX_PENDING_CHARS = 300  # flush even without punctuation if a run gets this long
# After playback truly ends (tts-done), wait this long for room echo to die
# away before the mic may re-arm.
SPEAKER_DRAIN_S = 0.4
# Safety margin on top of the clip duration for the tts-done wait, in case
# the overlay's confirmation is lost (reload, crash) — never deadlock.
TTS_DONE_GRACE_S = 5.0

# The LLM prefixes every reply with one of these emotion tags (see the system
# prompt); the tag is stripped before TTS and forwarded to the character so
# her face matches the conversation.
EMOTION_TAG = re.compile(r"^\s*\[([A-Za-z]+)\]\s*")
KNOWN_EMOTIONS = {"happy", "smug", "evil", "angry", "sad", "surprised", "neutral"}

# Whisper's classic no-speech hallucinations — if the mic caught nothing real,
# these show up. They must never reach the LLM. (Kept conservative: real
# one-word greetings like "hi" are NOT filtered.)
HALLUCINATION_RE = re.compile(
    r"^\s*("
    r"thank you\.?|thanks for watching\.?|bye\.?|goodbye\.?|"
    r"you\.?|hmm\.?|mm\.?|ah\.?|"
    r"subtitles? by.*|transcribed by.*|amara\.org.*"
    r")\s*$",
    re.IGNORECASE,
)


def pop_emotion_tag(sentence: str) -> tuple[str | None, str]:
    """Split a leading [emotion] tag off a sentence, if the LLM wrote one."""
    match = EMOTION_TAG.match(sentence)
    if not match:
        return None, sentence
    tag = match.group(1).lower()
    rest = sentence[match.end():].strip()
    return (tag if tag in KNOWN_EMOTIONS else None), rest


def env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def pop_sentence(buffer: str) -> tuple[str | None, str]:
    """Split one speakable sentence off the front of buffer, if one is complete."""
    match = SENTENCE_END.search(buffer)
    if match:
        sentence = buffer[: match.end()].strip()
        rest = buffer[match.end():]
        if sentence:
            return sentence, rest
        return None, rest
    if len(buffer) > MAX_PENDING_CHARS:
        cut = buffer.rfind(" ", 0, MAX_PENDING_CHARS)
        cut = cut if cut > 0 else MAX_PENDING_CHARS
        return buffer[:cut].strip(), buffer[cut:]
    return None, buffer


def resolve_device(kind: str, query: str) -> int | None:
    """Return the index of the first audio device whose name contains `query`.

    Returns None (system default) when `query` is empty or nothing matches.
    """
    if not query:
        return None
    for idx, device in enumerate(sd.query_devices()):
        if query.lower() in device["name"].lower():
            channels = device["max_input_channels" if kind == "input" else "max_output_channels"]
            if channels > 0:
                return idx
    print(f"[warn] no {kind} device matching {query!r}, using system default")
    return None


class Assistant:
    def __init__(self):
        print("Loading VAD...")
        self.vad = StreamingVAD(
            threshold=float(env("VAD_THRESHOLD", "0.5")),
            min_silence_ms=int(env("VAD_MIN_SILENCE_MS", "600")),
            speech_pad_ms=int(env("VAD_SPEECH_PAD_MS", "300")),
            device=env("VAD_DEVICE", "cpu"),
            dtype=env("VAD_DTYPE", "float32"),
        )
        print("Loading STT...")
        self.stt = STT(
            model_size=env("STT_MODEL", "distil-medium.en"),
            device=env("STT_DEVICE", "cuda"),
            compute_type=env("STT_COMPUTE_TYPE", "float32"),
        )
        print("Loading LLM...")
        self.llm = LLM(max_history_turns=int(env("MAX_HISTORY_TURNS", "12")))
        print("Loading TTS...")
        self.tts = TTS(
            voice=env("TTS_VOICE", "azelma"),
            device=env("TTS_DEVICE", "cuda"),
            dtype=env("TTS_DTYPE", "float32"),
        )

        self.listening = False  # hard gate: only queue mic frames while True
        self.voice_enabled = True  # bridge: overlay can mute TTS via stdin
        # Speech-only gate thresholds (see pocket_tts/vad.py speech_verdict):
        # a captured burst only counts as the user speaking when enough of it
        # was neurally scored as voice AND it clearly rose above the ambient
        # noise floor. Values come from .env so the gate can be tuned without
        # touching code.
        self.min_voiced_fraction = float(env("VAD_MIN_VOICED_FRACTION", "0.35"))
        self.noise_margin = float(env("VAD_NOISE_MARGIN", "3.0"))
        # Set by the stdin reader when the overlay reports that the TTS clip
        # has FINISHED PLAYING — speak() waits on it so the mic never re-arms
        # while her own voice is still in the air.
        self.tts_done = threading.Event()
        self.tts_done.set()
        # Clip correlation for the tts-done handshake: every emitted clip
        # gets a monotonic id and its confirmation must carry that same id,
        # so a stale ack (late confirmation for the PREVIOUS clip, or one
        # crossing a backend restart) can never release the gate early while
        # the current clip is still audible.
        self._clip_seq = 0
        self._clip_pending = None
        self.frames: queue.Queue[np.ndarray] = queue.Queue()
        self.mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=resolve_device("input", env("INPUT_DEVICE", "")),
            callback=self._on_mic,
        )
        # In bridge mode the overlay plays the voice (with lip-sync), so no
        # local speaker stream is needed — or opened.
        self.speaker_stream = None
        if not BRIDGE:
            self.speaker_stream = sd.OutputStream(
                samplerate=self.tts.output_sample_rate,
                channels=1,
                dtype="float32",
                device=resolve_device("output", env("OUTPUT_DEVICE", "")),
            )

    def _on_mic(self, indata, frames, time_info, status):
        # Drop everything unless we are actively listening, so the assistant's
        # own speaker output can never leak into the pipeline.
        if self.listening:
            self.frames.put(indata[:, 0].copy())

    def _read_config(self) -> None:
        """Consume config/control lines from the overlay (stdin) while it hosts us.

        Protocol: {"type": "config", "voice": true|false}
                  {"type": "tts-done"}  — the TTS clip finished playing.
        Lines sent before this thread starts are buffered by the pipe and
        picked up anyway.
        """
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "config":
                self.voice_enabled = bool(msg.get("voice", True))
                print(f"[config] voice replies {'on' if self.voice_enabled else 'off'}")
            elif msg.get("type") == "tts-done":
                # The confirmation carries the id of the clip that finished.
                # Only the clip we are waiting on may open the gate — a
                # stale ack (previous clip, previous backend generation)
                # must not re-arm the mic while audio is still in the air.
                # A missing id (legacy overlay) is accepted for safety.
                msg_id = msg.get("id")
                if msg_id is None or msg_id == self._clip_pending:
                    self._clip_pending = None
                    self.tts_done.set()

    def wait_for_utterance(self) -> np.ndarray:
        """Block until the user finishes speaking; return the utterance audio.

        Every burst that trips the endpointer is judged by speech_verdict():
        only genuine human speech (enough voiced chunks, energy clearly above
        the tracked ambient noise floor, energy concentrated in the voice
        band) is returned. Rejected bursts — keyboard bangs, fans, hum — are
        logged and skipped, so noise never reaches the STT/LLM.
        """
        self.vad.reset()
        while not self.frames.empty():
            self.frames.get_nowait()
        self.listening = True
        emit("state", "listening")
        try:
            while True:
                chunk = self.frames.get()
                event = self.vad.feed(chunk)
                if event == "start":
                    print("[...] I hear you...")
                elif event == "end":
                    audio, stats = self.vad.take_utterance()
                    if audio is None or len(audio) < MIN_UTTERANCE_S * SAMPLE_RATE:
                        continue
                    reason = speech_verdict(
                        stats,
                        min_voiced_fraction=self.min_voiced_fraction,
                        noise_margin=self.noise_margin,
                    )
                    if reason is not None:
                        print(f"[vad] burst rejected: {reason}")
                        continue
                    return audio
        finally:
            self.listening = False

    def speak(self, text: str) -> None:
        """Synthesize text and speak it; blocks until the line is delivered.

        Standalone: plays through the local speakers. Bridge: hands the clip
        to the overlay as a data URI (she lip-syncs it) and blocks until the
        overlay confirms playback has ENDED (tts-done), so the mic stays
        hard-gated for the whole time her voice is audible. A timeout based
        on the clip duration keeps a lost confirmation from deadlocking the
        pipeline.
        """
        if BRIDGE and not self.voice_enabled:
            return
        audio = self.tts.synthesize(text)
        if not audio.size:
            return
        if BRIDGE:
            duration = audio.size / self.tts.output_sample_rate
            self._clip_seq += 1
            self._clip_pending = self._clip_seq
            self.tts_done.clear()
            emit(
                "speak",
                {"id": self._clip_pending, "wav": wav_data_uri(audio, self.tts.output_sample_rate)},
            )
            self.tts_done.wait(timeout=duration + TTS_DONE_GRACE_S)
        else:
            self.speaker_stream.write(audio.reshape(-1, 1))

    def reply(self, user_text: str) -> None:
        """Stream an LLM reply and speak it sentence by sentence.

        A producer thread consumes the LLM token stream and queues complete
        sentences; the main path synthesizes and plays them in parallel, so
        speech starts while the model is still generating the rest.
        """
        sentences: queue.Queue[str | None] = queue.Queue()

        def produce():
            buffer = ""
            try:
                for delta in self.llm.stream_reply(user_text):
                    buffer += delta
                    print(delta, end="", flush=True)
                    while True:
                        sentence, buffer = pop_sentence(buffer)
                        if sentence is None:
                            break
                        sentences.put(sentence)
                if buffer.strip():
                    sentences.put(buffer.strip())
            except Exception as exc:
                print(f"\n[llm error] {exc}")
            finally:
                sentences.put(None)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        print("[evil] ", end="", flush=True)
        first = True
        while True:
            sentence = sentences.get()
            if sentence is None:
                break
            if first:
                # The LLM prefixes the reply with an emotion tag — strip it
                # out of the spoken text and tell the character to act it.
                tag, sentence = pop_emotion_tag(sentence)
                if tag and tag != "neutral":
                    emit("emotion", tag)
                if not sentence:
                    continue
                # Same gate the greeting uses: no audio will play while
                # muted, so don't show a speaking state either.
                if not BRIDGE or self.voice_enabled:
                    emit("state", "speaking")
                first = False
            self.speak(sentence)
        producer.join()
        print()

    def run(self) -> None:
        self.mic_stream.start()
        if self.speaker_stream is not None:
            self.speaker_stream.start()
        if BRIDGE:
            threading.Thread(target=self._read_config, daemon=True).start()
        print("\n=== EVIL is awake. Talk to her. Ctrl+C to quit. ===\n")
        tag, greeting = pop_emotion_tag(GREETING)
        if tag and tag != "neutral":
            emit("emotion", tag)
        if not BRIDGE or self.voice_enabled:
            emit("state", "speaking")
        self.speak(greeting)
        emit("state", "idle")
        while True:
            try:
                audio = self.wait_for_utterance()
                t0 = time.monotonic()
                emit("state", "thinking")
                try:
                    text = self.stt.transcribe(audio)
                except Exception as exc:
                    print(f"[stt error] {exc}")
                    emit("state", "idle")
                    continue
                if not text or HALLUCINATION_RE.match(text):
                    # Silence / no-speech hallucination — never answer nothing.
                    emit("state", "idle")
                    continue
                print(f"[you ] {text}  (stt {time.monotonic() - t0:.2f}s)")
                self.reply(text)
                emit("state", "idle")
                # Playback has already been confirmed ended (tts-done); this is
                # the room-echo decay window before the mic may re-arm.
                time.sleep(SPEAKER_DRAIN_S)
            except Exception as exc:
                # A transient VAD/TTS/audio-device failure must not kill the
                # whole pipeline — in bridge mode nothing would restart it,
                # so one glitch would silence her until the app is restarted.
                print(f"[loop error] {exc}")
                self.listening = False
                emit("state", "idle")
                time.sleep(0.5)


def main():
    assistant = Assistant()
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\nFinally, some peace and quiet.")
    finally:
        assistant.mic_stream.close()
        if assistant.speaker_stream is not None:
            assistant.speaker_stream.close()


if __name__ == "__main__":
    main()
