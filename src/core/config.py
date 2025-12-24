from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    env: str
    log_level: str
    cache_ttl_seconds: int

    kb_manifest: str
    kb_docs_dir: str
    kb_glossary: str

    llm_provider: str
    llm_model: str
    llm_temperature: float

    rag_top_k: int
    rag_use_mmr: bool
    rag_min_score: float

    market_primary: str
    market_fallback: str
    market_retries: int
    market_timeout_seconds: int


def _deep_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_settings(config_path: str = "config.yaml") -> Settings:
    """
    Loads config.yaml + overrides from .env/environment variables.
    """
    load_dotenv()  # loads .env into env vars

    cfg: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Defaults (if config missing)
    env = os.getenv("APP_ENV", _deep_get(cfg, "app.env", "dev"))
    log_level = os.getenv("LOG_LEVEL", _deep_get(cfg, "app.log_level", "INFO"))
    cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", _deep_get(cfg, "app.cache_ttl_seconds", 1800)))

    kb_manifest = os.getenv("KB_MANIFEST", _deep_get(cfg, "paths.kb_manifest", "data/kb/knowledge_base_manifest.csv"))
    kb_docs_dir = os.getenv("KB_DOCS_DIR", _deep_get(cfg, "paths.kb_docs_dir", "data/kb/docs"))
    kb_glossary = os.getenv("KB_GLOSSARY", _deep_get(cfg, "paths.kb_glossary", "data/kb/glossary.csv"))

    llm_provider = os.getenv("LLM_PROVIDER", _deep_get(cfg, "llm.provider", "openai"))
    llm_model = os.getenv("LLM_MODEL", _deep_get(cfg, "llm.model", "gpt-4.1-mini"))
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", _deep_get(cfg, "llm.temperature", 0.2)))

    rag_top_k = int(os.getenv("RAG_TOP_K", _deep_get(cfg, "rag.top_k", 5)))
    rag_use_mmr = str(os.getenv("RAG_USE_MMR", _deep_get(cfg, "rag.use_mmr", True))).lower() in ("1", "true", "yes")
    rag_min_score = float(os.getenv("RAG_MIN_SCORE", _deep_get(cfg, "rag.min_score", 0.2)))

    market_primary = os.getenv("MARKET_PRIMARY", _deep_get(cfg, "market_data.primary", "yfinance"))
    market_fallback = os.getenv("MARKET_FALLBACK", _deep_get(cfg, "market_data.fallback", "alphavantage"))
    market_retries = int(os.getenv("MARKET_RETRIES", _deep_get(cfg, "market_data.retries", 3)))
    market_timeout_seconds = int(os.getenv("MARKET_TIMEOUT_SECONDS", _deep_get(cfg, "market_data.timeout_seconds", 20)))

    return Settings(
        env=env,
        log_level=log_level,
        cache_ttl_seconds=cache_ttl_seconds,
        kb_manifest=kb_manifest,
        kb_docs_dir=kb_docs_dir,
        kb_glossary=kb_glossary,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        rag_top_k=rag_top_k,
        rag_use_mmr=rag_use_mmr,
        rag_min_score=rag_min_score,
        market_primary=market_primary,
        market_fallback=market_fallback,
        market_retries=market_retries,
        market_timeout_seconds=market_timeout_seconds,
    )


# Optional convenience singleton
SETTINGS = load_settings()
