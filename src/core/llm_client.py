from __future__ import annotations

import os
from dataclasses import dataclass

from src.core.config import SETTINGS


@dataclass
class LLMResponse:
    text: str


class LLMClient:
    """
    Stage 1: Gemini via the new Google GenAI SDK (google-genai).
    Docs show: from google import genai; client = genai.Client(); client.models.generate_content(...)
    """

    def __init__(self) -> None:
        provider = (SETTINGS.llm_provider or "").lower()
        if provider != "gemini":
            raise ValueError(f"Stage 1 expects Gemini. Current provider={SETTINGS.llm_provider}")

        # Preferred env var per Gemini docs
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in .env (or GOOGLE_API_KEY as fallback)")

        # New SDK
        from google import genai  # pip install google-genai

        # You can pass api_key explicitly OR rely on GEMINI_API_KEY env var.
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> LLMResponse:
        resp = self._client.models.generate_content(
            model=SETTINGS.llm_model,
            contents=prompt,
            # Keep generation config simple in Stage 1
            config={
                "temperature": float(SETTINGS.llm_temperature),
            },
        )

        # resp.text exists in the new SDK quickstart examples
        text = getattr(resp, "text", "") or ""
        return LLMResponse(text=text.strip())

    def list_models(self) -> list[str]:
        """
        Helpful debug utility when you get 'model not found'.
        """
        models = []
        for m in self._client.models.list():
            name = getattr(m, "name", None)
            if name:
                models.append(name)
        return models
