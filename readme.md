# AI VTuber

A realtime AI voice assistant with an anime character overlay on your screen. You talk to it, it listens, thinks, and talks back in a cloned voice while the character lip-syncs and reacts.

## How it works

```
Microphone > VAD (Silero VAD) > STT (faster-whisper) > LLM (Groq API key only) > TTS (Pocket TTS)
```

The voice pipeline runs locally. Only the LLM uses a Groq API key. Everything else is a local model.

## Features

- Realtime voice conversation
- Voice cloning from a reference audio file (wav, mp3, flac, ogg and more)
- Live lip-sync with viseme analysis
- 3D anime character overlay (transparent, always on top)
- Character idle animation, gaze follow and reactions
- Echo safety (it never hears its own voice)
- Fast responses (sentences are spoken while the LLM is still generating)
- Frontend and backend are tightly integrated to get the lowest latency possible

## Requirements

- Windows
- Python 3.10+ (installed by the setup script)
- Node.js (installed by the setup script)
- NVIDIA GPU with CUDA
- A Groq API key (free)

## Hardware requirements

Minimum for realtime use:

- 4 GB system RAM
- CUDA GPU with at least 4 GB VRAM (6 GB is recommended)

## Setup

1. Clone this repo:

   ```
   git clone https://github.com/biswas445/ai_vtuber.git
   ```

2. Move into the project folder:

   ```
   cd ai_vtuber
   ```

3. Run the setup:

   ```
   install.bat
   ```

   This creates the Python virtual environment, installs everything, and downloads the AI models.

## Add your Groq API key

1. Go to https://console.groq.com and sign in or sign up (free).
2. Create an API key at https://console.groq.com/keys.
3. Open `pocket_tts/.env` in the project (a template with the key field removed is already in the repo, nothing sensitive in it).
4. Put your key here:

   ```
   GROQ_API_KEY=your_key_here
   ```

5. Save the file.

## Start

Double-click `Hikasha.bat` to start the project. The character overlay appears on your screen. Just talk to it.

## Voice cloning

Put your own reference audio file (for example a wav or mp3) in `pocket_tts/evil_voice/` and set the path in `pocket_tts/.env`:

```
TTS_VOICE=path\to\your\audio.wav
```

The assistant will speak in the cloned voice of that reference. Use 5 to 30 seconds of clean audio from one speaker.

## Thanks

Thanks to the local model providers only:

- [Silero VAD](https://github.com/snakers4/silero-vad) - voice activity detection
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - speech to text
- [Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts) - text to speech and voice cloning

## License

The Pocket TTS engine is under the [MIT License](pocket_tts/LICENSE), Copyright (c) 2025 Kyutai.

## Purpose

This is a simple foundation for an AI avatar or assistant. You can modify it however you like, extend it, and make it your own.
