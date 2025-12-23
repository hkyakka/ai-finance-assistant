from __future__ import annotations

from datetime import datetime

import pytest

from src.core.schemas import MarketQuote
from src.utils.cache import TTLCache
from src.utils.market_data import MarketDataService, MarketDataError, ProviderUnavailable


def test_quote_cache_hit(monkeypatch):
    cache = TTLCache(default_ttl_seconds=60)
    svc = MarketDataService(cache=cache)

    calls = {"n": 0}

    def fake_quote(provider, symbol, ttl):
        calls["n"] += 1
        return type("R", (), {"quote": MarketQuote(symbol=symbol, price=123.0, as_of=datetime.utcnow(), provider=provider), "ttl_seconds": ttl})

    monkeypatch.setattr(svc, "_quote_via", fake_quote)

    q1 = svc.get_quote("AAPL")
    assert q1.price == 123.0
    assert q1.from_cache is False
    assert calls["n"] == 1

    q2 = svc.get_quote("AAPL")
    assert q2.from_cache is True
    assert calls["n"] == 1  # no extra call


def test_fallback_used(monkeypatch):
    cache = TTLCache(default_ttl_seconds=60)
    svc = MarketDataService(cache=cache)
    svc.primary = "alphavantage"
    svc.fallback = "yfinance"

    def fake_quote_via(provider, symbol, ttl):
        if provider == "alphavantage":
            raise ProviderUnavailable("down")
        return type("R", (), {"quote": MarketQuote(symbol=symbol, price=200.0, as_of=datetime.utcnow(), provider=provider), "ttl_seconds": ttl})

    monkeypatch.setattr(svc, "_quote_via", fake_quote_via)

    q = svc.get_quote("MSFT", force_refresh=True)
    assert q.price == 200.0
    assert q.provider == "yfinance"


def test_all_providers_fail(monkeypatch):
    cache = TTLCache(default_ttl_seconds=60)
    svc = MarketDataService(cache=cache)
    svc.primary = "alphavantage"
    svc.fallback = "yfinance"

    def fake_quote_via(provider, symbol, ttl):
        raise ProviderUnavailable("nope")

    monkeypatch.setattr(svc, "_quote_via", fake_quote_via)

    with pytest.raises(MarketDataError):
        svc.get_quote("TSLA", force_refresh=True)
