"""LLM providers behind one tiny interface, so the model can be swapped later.

Anything with `generate(prompt, system) -> str` is a provider. Tests use a fake
one; production uses Gemini. The generation layer never imports google-genai
directly, so replacing Gemini means writing one new class here and nothing else.
"""

from __future__ import annotations

import time
from typing import Protocol

from src import config


class LLMError(RuntimeError):
    """Generation failed after retries. The message is safe to show a user."""


class LLMProvider(Protocol):
    name: str

    def generate(self, prompt: str, system: str = "") -> str: ...


# Rate-limited (429) and overloaded (503) are transient on the free tier;
# anything else (bad key, bad model name) will not fix itself by waiting.
_RETRYABLE = {429, 503}


class GeminiProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 max_retries: int = 4, base_delay: float = 2.0):
        self.model = model or config.GEMINI_MODEL
        self.name = f"gemini:{self.model}"
        self.max_retries = max_retries
        self.base_delay = base_delay
        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to .env "
                           "(see .env.example).")
        from google import genai
        self._client = genai.Client(api_key=key)

    def generate(self, prompt: str, system: str = "") -> str:
        from google.genai import errors, types

        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.1,  # low: we want faithful restatement, not creativity
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True),  # no tools; also silences an SDK warning
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg)
                return (resp.text or "").strip()
            except errors.APIError as e:
                if e.code not in _RETRYABLE:
                    raise LLMError(f"Gemini request failed ({e.code}): "
                                   f"{e.message}") from e
                if attempt == self.max_retries:
                    raise LLMError(
                        f"Gemini is rate-limiting or overloaded ({e.code}) and "
                        f"still failing after {self.max_retries} retries. Wait a "
                        f"minute and try again.") from e
                delay = self.base_delay * 2 ** attempt   # 2, 4, 8, 16 s
                print(f"  Gemini {e.code}; retrying in {delay:.0f}s "
                      f"({attempt + 1}/{self.max_retries})")
                time.sleep(delay)
        raise AssertionError("unreachable")
