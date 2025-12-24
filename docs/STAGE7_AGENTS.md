# Stage 7 — Implement Core Agents (RAG + Market + Quant)

This patch implements the main agents using the deterministic tools and services built in Stages 4–6.

## Agents added
- `finance_qa_agent.py`  : RAG-backed education Q&A + citations
- `tax_agent.py`         : RAG-backed tax education + strong disclaimer + citations
- `portfolio_agent.py`   : Portfolio analysis via MarketDataService + quant_engine (deterministic)
- `goal_agent.py`        : Goal projections via quant_engine (deterministic)
- `market_agent.py`      : Quote + basic context via MarketDataService
- `news_agent.py`        : Optional NewsAPI integration (graceful fallback if no key)

## Smoke
```bash
python scripts/agents_smoke.py
```

## Tests
```bash
pytest -q
```

> LLM generation is optional: if keys are missing, agents still return deterministic outputs + citations.
