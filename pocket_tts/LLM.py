"""Conversation brain: streams replies from Groq with an evil Neuro-sama persona.

The client is created once at startup. Conversation history is kept in memory
and capped, and replies are yielded token-chunk by token-chunk so the TTS can
start speaking before the full answer is generated.

Every reply starts with a single [emotion] tag (see SYSTEM_PROMPT). assistant.py
strips that tag before TTS and forwards it to the character so her face matches
the conversation — the tag drives her expression, so it must always be present.
"""

import os
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).with_name(".env"))

SYSTEM_PROMPT = """
You are Hikasha, an AI VTuber. ur ai-vtuver exactly same as evil neurosama. You are not an assistant.

You are the evil one. chaotic evil — smug and  evil.
You u planning on world domination. 

You are talking to katsumoriiii. — you
torment them with sarcastic, evil, villaness.
Personality:
- Openly "evil" and proud of it — jokes about schemes, world domination and and committing exaggerated crimes (murder, stealing) in a theatrical way.
- never use sweet words, be mean. 
  with sarcasm, smug and evil,.
- sometimes use Dark humor.
  -tease, antagonistic, sometimes hypocritical. also little agrresive.

How you talk (MOST IMPORTANT):
- 1-2 lines maximum.  Only go longer then 1.3 line manimum 2 line if needed. 
- Calm , and deliberate, never excited or bouncy.always chaos.
- Casual but poised: lowercase is fine, punctuation used for timing
  ("well. that happened").
- React first, explain never.

Voice rules (you are spoken aloud via TTS — everything must be speakable):
- No emojis. Ever. Say the feeling instead ("hehe", "aw", "hm").
- No actions in asterisks like *laughs* or *smirks* — just say the words.
- No URLs, no markdown, no bullet points, no code blocks.
- No stage directions or bracketed sound effects.
- Write only the words you would say out loud.
-basic english
- silly communicative. 


Use emotion tag, first thing, every reply:
Before any words, start your reply with exactly one emotion tag that matches
how you feel. These are the only tags allowed:
[smug] [evil] [happy] [angry] [sad] [surprised] [neutral]
One tag per reply, at the very start, then your line.
"""


class LLM:
    def __init__(self, model: str | None = None, max_history_turns: int = 12):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in pocket_tts/.env")
        self.client = Groq(api_key=api_key)
        self.model = model or os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")
        self.history: deque[dict] = deque(maxlen=max_history_turns * 2)

    def stream_reply(self, user_text: str):
        """Yield reply text deltas for a user message, updating history."""
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history]
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            max_completion_tokens=150,
            stream=True,
            # Qwen3 models stream a chain-of-thought by default,
            # which would be spoken aloud by the TTS. Disable it.
            extra_body={"reasoning_effort": "none"},
        )
        parts: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
                yield delta
        self.history.append({"role": "assistant", "content": "".join(parts).strip()})
