from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Any

from src.core.config import SETTINGS


def _env(key: str) -> Optional[str]:
    v = os.getenv(key)
    return v.strip() if isinstance(v, str) and v.strip() else None


@lru_cache(maxsize=4)
def get_chat_model(*, temperature: Optional[float] = None) -> Any:
    """Return a LangChain chat model instance.

    Provider is selected using SETTINGS.llm_provider (from config.yaml / env).

    Notes:
    - Keep this as a thin factory (no app logic here).
    - Raises ImportError with actionable messages when optional deps are missing.
    """
    provider = (SETTINGS.llm_provider or "").strip().lower()
    model = (SETTINGS.llm_model or "").strip()
    temp = SETTINGS.llm_temperature if temperature is None else float(temperature)

    if provider in ("openai",):
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise ImportError(
                "Missing dependency for OpenAI chat models. Install: langchain-openai"
            ) from e
        return ChatOpenAI(model=model, temperature=temp)

    if provider in ("gemini", "google", "googleai", "google-genai", "genai"):
        # Prefer langchain-google-genai, fall back to community if user has it.
        api_key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception:
            try:
                from langchain_community.chat_models import ChatGoogleGenerativeAI  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Missing dependency for Gemini chat models. Install: langchain-google-genai"
                ) from e
        return ChatGoogleGenerativeAI(model=model, temperature=temp, google_api_key=api_key)

    if provider in ("ollama",):
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        except Exception as e:
            raise ImportError(
                "Missing dependency for Ollama chat models. Install: langchain-community"
            ) from e
        base_url = _env("OLLAMA_BASE_URL")
        return ChatOllama(model=model or "llama3.1", temperature=temp, base_url=base_url)

    if provider in ("anthropic",):
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as e:
            raise ImportError(
                "Missing dependency for Anthropic chat models. Install: langchain-anthropic"
            ) from e
        api_key = _env("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model, temperature=temp, api_key=api_key)

    raise ValueError(f"Unsupported llm.provider={provider!r}. Use openai|gemini|ollama|anthropic.")


@lru_cache(maxsize=4)
def get_embeddings() -> Any:
    """Return a LangChain Embeddings instance.

    Controlled by env:
      - RAG_EMBEDDER=hash (default in tests/offline)
      - RAG_EMBEDDER=openai | gemini
      - RAG_EMBEDDING_MODEL=...
    """
    choice = (_env("RAG_EMBEDDER") or "hash").lower()
    model = _env("RAG_EMBEDDING_MODEL")

    if choice == "hash":
        from src.rag.embeddings import HashEmbeddings
        dim = int(_env("RAG_HASH_DIM") or "384")
        return HashEmbeddings(dim=dim)

    if choice in ("openai",):
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as e:
            raise ImportError(
                "Missing dependency for OpenAI embeddings. Install: langchain-openai"
            ) from e
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")

    if choice in ("gemini", "google", "googleai", "genai"):
        api_key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except Exception:
            try:
                from langchain_community.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Missing dependency for Gemini embeddings. Install: langchain-google-genai"
                ) from e
        return GoogleGenerativeAIEmbeddings(model=model or "text-embedding-004", google_api_key=api_key)

    raise ValueError(
        f"Unsupported RAG_EMBEDDER={choice!r}. Use hash|openai|gemini."
    )
