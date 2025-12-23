# Stage 6 — Deterministic Quant Engine + Quant Agent

This patch adds a deterministic numeric engine (no LLM math) and a QuantAgent wrapper.

## What you get
- `src/utils/quant_engine.py`: pure computation for portfolio + goal planning
- `src/utils/quant_models.py`: Pydantic models
- `src/tools/quant_tools.py`: tool-friendly wrappers (return dicts)
- `src/agents/quant_agent.py`: agent wrapper for graph integration
- Unit tests under `tests/`

## Install deps (if missing)
```bash
pip install numpy pandas
```

## Smoke
```bash
python scripts/quant_smoke.py
```

## Tests
```bash
pytest -q
```
