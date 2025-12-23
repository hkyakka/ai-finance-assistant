"""Simple manual smoke test for Stage 5 MarketDataService.

Usage:
  python scripts/market_smoke.py AAPL
"""
import sys
from src.utils.market_data import MarketDataService

def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    svc = MarketDataService()

    q1 = svc.get_quote(symbol)
    print("Quote1:", q1.model_dump())

    q2 = svc.get_quote(symbol)
    print("Quote2 (cached expected):", q2.model_dump())

    series = svc.get_history_close(symbol, period="1mo", interval="1d")
    print("History provider:", series.provider, "points:", len(series.close))

if __name__ == "__main__":
    main()
