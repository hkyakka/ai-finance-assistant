# Stage 5 — MarketDataService

## What you get
- `MarketDataService.get_quote(symbol)` with:
  - TTL caching
  - retries/backoff
  - primary provider + fallback provider
- Providers implemented:
  - Alpha Vantage (requires `ALPHAVANTAGE_API_KEY`)
  - Yahoo Finance via `yfinance` (optional dependency)
  - Stooq (free CSV fallback)

## Setup
Add to requirements:
- `yfinance` (optional but recommended)
- `requests` already used

Environment variables:
- `ALPHAVANTAGE_API_KEY` (if using Alpha Vantage as primary)

Config:
`config.yaml` -> `market_data` section controls `primary/fallback/retries/timeout_seconds`.

## Quick run
`python scripts/market_smoke.py AAPL`

If `yfinance` is not installed and you don't have AlphaVantage key, set:
`MARKET_PRIMARY=stooq` and `MARKET_FALLBACK=stooq`.
