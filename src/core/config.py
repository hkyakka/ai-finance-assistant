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

    # IMPORTANT: treat empty env vars as "not set".
    # On Windows it's easy to end up with LLM_PROVIDER="" in the environment,
    # which would override config.yaml and break provider selection.
    def _env_or_cfg(key: str, cfg_path: str, default):
        v = os.getenv(key)
        if v is None:
            return _deep_get(cfg, cfg_path, default)
        v = v.strip()
        return _deep_get(cfg, cfg_path, default) if v == "" else v

    llm_provider = _env_or_cfg("LLM_PROVIDER", "llm.provider", "openai")
    llm_model = _env_or_cfg("LLM_MODEL", "llm.model", "gpt-4.1-mini")
    llm_temperature = float(_env_or_cfg("LLM_TEMPERATURE", "llm.temperature", 0.2))

    # Normalize common aliases so config and code are consistent.
    if isinstance(llm_provider, str):
        lp = llm_provider.strip().lower()
        if lp in ("google", "googleai", "google-genai", "genai"):
            llm_provider = "gemini"
        else:
            llm_provider = lp

    rag_top_k = int(os.getenv("RAG_TOP_K", _deep_get(cfg, "rag.top_k", 5)))
    rag_use_mmr = str(os.getenv("RAG_USE_MMR", _deep_get(cfg, "rag.use_mmr", True))).lower() in ("1", "true", "yes")
    rag_min_score = float(os.getenv("RAG_MIN_SCORE", _deep_get(cfg, "rag.min_score", 0.2)))

    market_primary = _env_or_cfg("MARKET_PRIMARY", "market_data.primary", "alphavantage")
    market_fallback = _env_or_cfg("MARKET_FALLBACK", "market_data.fallback", "yfinance")
    market_retries = int(_env_or_cfg("MARKET_RETRIES", "market_data.retries", 3))
    market_timeout_seconds = int(_env_or_cfg("MARKET_TIMEOUT_SECONDS", "market_data.timeout_seconds", 20))

    if isinstance(market_primary, str):
        market_primary = market_primary.strip().lower()
    if isinstance(market_fallback, str):
        market_fallback = market_fallback.strip().lower()

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
