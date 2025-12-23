from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import requests

from src.core.config import SETTINGS
from src.core.schemas import MarketQuote, PriceSeries
from src.utils.cache import TTLCache
from src.utils.logging import get_logger

logger = get_logger("market_data")


class MarketDataError(Exception):
    pass


class ProviderUnavailable(MarketDataError):
    pass


class SymbolNotFound(MarketDataError):
    pass


class RateLimited(MarketDataError):
    pass


@dataclass
class _ProviderResult:
    quote: MarketQuote
    ttl_seconds: int


class MarketDataService:
    """
    Stage 5:
    - Quote lookup via primary provider with fallback
    - TTL caching
    - Retries with backoff for transient errors
    """

    def __init__(self, cache: Optional[TTLCache] = None) -> None:
        self.cache = cache or TTLCache(default_ttl_seconds=SETTINGS.cache_ttl_seconds)
        self.session = requests.Session()

        self.primary = (SETTINGS.market_primary or "alphavantage").lower()
        self.fallback = (SETTINGS.market_fallback or "yfinance").lower()
        self.retries = int(SETTINGS.market_retries or 3)
        self.timeout = int(SETTINGS.market_timeout_seconds or 20)

    def get_quote(self, symbol: str, *, force_refresh: bool = False, ttl_seconds: Optional[int] = None) -> MarketQuote:
        sym = symbol.strip().upper()
        if not sym:
            raise ValueError("symbol is empty")

        cache_key = f"quote:{sym}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                val, remaining = cached
                q: MarketQuote = val
                q.from_cache = True
                q.ttl_seconds = remaining
                return q

        ttl = int(ttl_seconds or SETTINGS.cache_ttl_seconds)

        # Try primary then fallback
        last_err: Optional[Exception] = None
        for provider in [self.primary, self.fallback]:
            try:
                res = self._quote_via(provider, sym, ttl)
                # Cache
                res.quote.from_cache = False
                res.quote.ttl_seconds = ttl
                self.cache.set(cache_key, res.quote, ttl_seconds=ttl)
                return res.quote
            except SymbolNotFound as e:
                # symbol issues: don't fallback to avoid misleading
                raise
            except Exception as e:
                last_err = e
                logger.warning(f"provider_failed provider={provider} symbol={sym} err={type(e).__name__}:{e}")

        raise MarketDataError(f"All providers failed for {sym}. Last error: {last_err}")

    def get_history_close(self, symbol: str, *, period: str = "1mo", interval: str = "1d",
                          force_refresh: bool = False, ttl_seconds: Optional[int] = None) -> PriceSeries:
        """
        Simple close-price series for charts.
        Uses same provider selection as quote.
        """
        sym = symbol.strip().upper()
        if not sym:
            raise ValueError("symbol is empty")

        cache_key = f"history:{sym}:{period}:{interval}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                val, _ = cached
                s: PriceSeries = val
                s.from_cache = True
                return s

        ttl = int(ttl_seconds or SETTINGS.cache_ttl_seconds)
        last_err: Optional[Exception] = None
        for provider in [self.primary, self.fallback]:
            try:
                series = self._history_via(provider, sym, period=period, interval=interval)
                series.from_cache = False
                self.cache.set(cache_key, series, ttl_seconds=ttl)
                return series
            except Exception as e:
                last_err = e
                logger.warning(f"provider_failed_history provider={provider} symbol={sym} err={type(e).__name__}:{e}")

        raise MarketDataError(f"All providers failed for history {sym}. Last error: {last_err}")

    # -----------------------------
    # Provider selection
    # -----------------------------
    def _quote_via(self, provider: str, symbol: str, ttl: int) -> _ProviderResult:
        if provider == "alphavantage":
            return self._quote_alphavantage(symbol, ttl)
        if provider == "yfinance":
            return self._quote_yfinance(symbol, ttl)
        if provider == "stooq":
            return self._quote_stooq(symbol, ttl)
        raise ProviderUnavailable(f"Unknown provider: {provider}")

    def _history_via(self, provider: str, symbol: str, period: str, interval: str) -> PriceSeries:
        if provider == "alphavantage":
            # AlphaVantage free tier is limited for history; prefer yfinance/stooq for charts.
            return self._history_yfinance(symbol, period=period, interval=interval)
        if provider == "yfinance":
            return self._history_yfinance(symbol, period=period, interval=interval)
        if provider == "stooq":
            return self._history_stooq(symbol)
        raise ProviderUnavailable(f"Unknown provider: {provider}")

    # -----------------------------
    # Providers: Alpha Vantage
    # -----------------------------
    def _quote_alphavantage(self, symbol: str, ttl: int) -> _ProviderResult:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_KEY")
        if not api_key:
            raise ProviderUnavailable("ALPHAVANTAGE_API_KEY not set")

        url = "https://www.alphavantage.co/query"
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key}

        data = self._request_json_with_retries(url, params=params)

        if "Note" in data:
            # rate limit
            raise RateLimited(data["Note"])
        if "Error Message" in data:
            raise SymbolNotFound(data["Error Message"])

        q = data.get("Global Quote") or {}
        price_str = q.get("05. price") or q.get("05. price".upper())
        if not price_str:
            raise MarketDataError(f"AlphaVantage missing price for {symbol}")

        try:
            price = float(price_str)
        except Exception:
            raise MarketDataError(f"Invalid price from AlphaVantage for {symbol}: {price_str}")

        quote = MarketQuote(
            symbol=symbol,
            price=price,
            currency="USD",
            as_of=datetime.utcnow(),
            provider="alphavantage",
            from_cache=False,
            ttl_seconds=ttl,
        )
        return _ProviderResult(quote=quote, ttl_seconds=ttl)

    # -----------------------------
    # Providers: Yahoo Finance via yfinance
    # -----------------------------
    def _quote_yfinance(self, symbol: str, ttl: int) -> _ProviderResult:
        try:
            import yfinance as yf
        except Exception as e:
            raise ProviderUnavailable("yfinance not installed. pip install yfinance") from e

        # Retry around yfinance calls
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                t = yf.Ticker(symbol)
                # fast_info is light; fallback to history if needed
                info = getattr(t, "fast_info", None) or {}
                price = info.get("last_price") or info.get("lastPrice") or info.get("regularMarketPrice")
                if price is None:
                    hist = t.history(period="1d", interval="1d")
                    if hist is None or hist.empty:
                        raise SymbolNotFound(f"No data for {symbol}")
                    price = float(hist["Close"].iloc[-1])

                quote = MarketQuote(
                    symbol=symbol,
                    price=float(price),
                    currency=info.get("currency") or "USD",
                    as_of=datetime.utcnow(),
                    provider="yfinance",
                    from_cache=False,
                    ttl_seconds=ttl,
                )
                return _ProviderResult(quote=quote, ttl_seconds=ttl)
            except Exception as e:
                last_err = e
                sleep_s = min(8.0, 0.8 * (2 ** (attempt - 1)))
                time.sleep(sleep_s)

        raise MarketDataError(f"yfinance failed for {symbol}: {last_err}")

    def _history_yfinance(self, symbol: str, period: str, interval: str) -> PriceSeries:
        try:
            import yfinance as yf
        except Exception as e:
            raise ProviderUnavailable("yfinance not installed. pip install yfinance") from e

        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                t = yf.Ticker(symbol)
                hist = t.history(period=period, interval=interval)
                if hist is None or hist.empty:
                    raise SymbolNotFound(f"No history for {symbol}")

                dates = [d.to_pydatetime().date().isoformat() for d in hist.index]
                close = [float(x) for x in hist["Close"].tolist()]
                return PriceSeries(
                    symbol=symbol,
                    dates=dates,
                    close=close,
                    as_of=datetime.utcnow(),
                    provider="yfinance",
                    from_cache=False,
                )
            except Exception as e:
                last_err = e
                sleep_s = min(8.0, 0.8 * (2 ** (attempt - 1)))
                time.sleep(sleep_s)

        raise MarketDataError(f"yfinance history failed for {symbol}: {last_err}")

    # -----------------------------
    # Providers: Stooq (free CSV) - backup
    # -----------------------------
    def _quote_stooq(self, symbol: str, ttl: int) -> _ProviderResult:
        # Stooq uses symbols like aapl.us. We'll try to be helpful.
        sym = symbol.lower()
        if "." not in sym:
            sym = f"{sym}.us"
        url = f"https://stooq.com/q/l/?s={sym}&f=sd2t2ohlcv&h&e=csv"

        text = self._request_text_with_retries(url)
        # CSV header: Symbol,Date,Time,Open,High,Low,Close,Volume
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            raise SymbolNotFound(f"Stooq returned no data for {symbol}")

        parts = lines[1].split(",")
        if len(parts) < 8 or parts[0] == "N/D":
            raise SymbolNotFound(f"Stooq no data for {symbol}")

        close = float(parts[6])
        quote = MarketQuote(
            symbol=symbol,
            price=close,
            currency="USD",
            as_of=datetime.utcnow(),
            provider="stooq",
            from_cache=False,
            ttl_seconds=ttl,
        )
        return _ProviderResult(quote=quote, ttl_seconds=ttl)

    def _history_stooq(self, symbol: str) -> PriceSeries:
        # Daily history CSV
        sym = symbol.lower()
        if "." not in sym:
            sym = f"{sym}.us"
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        text = self._request_text_with_retries(url)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            raise SymbolNotFound(f"Stooq returned no history for {symbol}")

        # Header: Date,Open,High,Low,Close,Volume
        dates = []
        close = []
        for ln in lines[1:]:
            p = ln.split(",")
            if len(p) < 5:
                continue
            dates.append(p[0])
            close.append(float(p[4]))

        if not dates:
            raise SymbolNotFound(f"Stooq returned empty history for {symbol}")

        return PriceSeries(
            symbol=symbol,
            dates=dates,
            close=close,
            as_of=datetime.utcnow(),
            provider="stooq",
            from_cache=False,
        )

    # -----------------------------
    # HTTP helpers with retries
    # -----------------------------
    def _request_json_with_retries(self, url: str, *, params: dict) -> dict:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                # AlphaVantage sometimes returns 200 with error JSON
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                last_err = e
                code = getattr(e.response, "status_code", None)
                if code in (429, 500, 502, 503, 504):
                    time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
                    continue
                raise
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
            except Exception as e:
                last_err = e
                time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
        raise MarketDataError(f"HTTP JSON request failed: {last_err}")

    def _request_text_with_retries(self, url: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                return r.text
            except requests.HTTPError as e:
                last_err = e
                code = getattr(e.response, "status_code", None)
                if code in (429, 500, 502, 503, 504):
                    time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
                    continue
                raise
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
            except Exception as e:
                last_err = e
                time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
        raise MarketDataError(f"HTTP text request failed: {last_err}")
