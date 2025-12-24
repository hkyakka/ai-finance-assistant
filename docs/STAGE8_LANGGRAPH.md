# Stage 8 — LangGraph Orchestration

This stage wires the existing agents/tools into a LangGraph state machine with:
- Deterministic router
- Tool nodes (RAG retrieve, Market quote, Quant compute)
- Agent nodes (FinanceQA, Tax, News)
- Deterministic response nodes (Market, Portfolio, Goal) using tool outputs
- Validator + Composer
- Basic memory summary + last-turn buffer (LLM summarized, best-effort)

## Run smoke
```bash
python tests/run_graph_smoke.py
```

## Run tests
```bash
pytest -q
```

## Notes
- External network calls are monkeypatched in `tests/test_graph_integration.py` so CI/dev runs are stable.
- Real execution uses your configured providers + keys.
