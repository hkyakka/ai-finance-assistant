from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.langchain_factory import get_chat_model


@dataclass
class LLMResponse:
    text: str


class LLMClient:
    """Thin wrapper kept for backward compatibility.

    Existing agents/tests call:
      - LLMClient().generate(prompt) -> LLMResponse(text=...)
    """

    def __init__(self, *, temperature: float | None = None) -> None:
        self._model: Any = get_chat_model(temperature=temperature)

    def generate(self, prompt: str) -> LLMResponse:
        # LangChain chat models accept a string prompt; they return a BaseMessage.
        msg = self._model.invoke(prompt)
        text = getattr(msg, "content", None)
        if text is None:
            text = str(msg)
        return LLMResponse(text=str(text).strip())

    def list_models(self) -> list[str]:
        # Model listing is provider-specific; keep as a safe no-op.
        return []
