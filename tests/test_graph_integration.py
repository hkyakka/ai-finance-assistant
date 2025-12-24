import pytest

from src.workflow.graph import build_graph
from src.core.schemas import UserProfile, PortfolioInput, Holding, GoalInput, RagResult, RagChunk, MarketQuote


@pytest.fixture()
def graph():
    return build_graph()


def test_graph_financeqa(monkeypatch, graph):
    # Patch retriever to avoid needing a built index
    from src.rag import retriever as retriever_mod

    def fake_retrieve(self, query: str, top_k: int, use_mmr: bool, mmr_lambda: float, min_score: float):
        return RagResult(
            query=query,
            chunks=[
                RagChunk(
                    doc_id="doc1",
                    chunk_id="c1",
                    title="ETF basics",
                    url="https://example.com/etf",
                    snippet="ETF is a basket of securities traded on exchanges.",
                    score=0.9,
                )
            ],
        )

    monkeypatch.setattr(retriever_mod.Retriever, "retrieve", fake_retrieve)

    # Patch LLM to be deterministic
    from src.core import llm_client as llm_mod
    monkeypatch.setattr(
        llm_mod.LLMClient,
        "generate",
        lambda self, prompt: llm_mod.LLMResponse(text="An ETF is a basket of securities that trades like a stock."),
    )

    out = graph.invoke({"user_text": "What is an ETF?", "user_profile": UserProfile()})
    assert out["final"].agent_name in ("FinanceQAAgent", "Fallback")
    assert "ETF" in out["final"].answer_md
    tool_names = [t.tool_name for t in out.get("tool_calls", [])]
    assert "RAG_RETRIEVE" in tool_names


def test_graph_market(monkeypatch, graph):
    # Patch market service to avoid network
    from src.utils import market_data as md

    monkeypatch.setattr(
        md.MarketDataService,
        "get_quote",
        lambda self, symbol: MarketQuote(
            symbol=symbol, price=123.45, currency="USD", provider="yfinance", as_of="2025-01-01T00:00:00Z", from_cache=True
        ),
    )
    out = graph.invoke({"user_text": "AAPL price", "user_profile": UserProfile()})
    assert "Market snapshot" in out["final"].answer_md
    tool_names = [t.tool_name for t in out.get("tool_calls", [])]
    assert "MARKET_QUOTE" in tool_names


def test_graph_portfolio(monkeypatch, graph):
    from src.utils import market_data as md
    monkeypatch.setattr(
        md.MarketDataService,
        "get_quote",
        lambda self, symbol: MarketQuote(
            symbol=symbol, price=100.0, currency="USD", provider="yfinance", as_of="2025-01-01T00:00:00Z", from_cache=True
        ),
    )

    pf = PortfolioInput(
        currency="USD",
        holdings=[
            Holding(symbol="AAPL", quantity=2, avg_cost=90),
            Holding(symbol="MSFT", quantity=1, avg_cost=80),
        ],
    )

    out = graph.invoke({"user_text": "Analyze my portfolio risk", "user_profile": UserProfile(), "portfolio": pf})
    assert "Portfolio snapshot" in out["final"].answer_md
    tool_names = [t.tool_name for t in out.get("tool_calls", [])]
    assert "MARKET_QUOTE" in tool_names
    assert "QUANT_COMPUTE" in tool_names


def test_graph_goal(graph):
    goal = GoalInput(
        goal_name="100k",
        target_amount=100000,
        time_horizon_years=10,
        current_savings=10000,
        monthly_contribution=500,
        expected_return_annual=0.08,
        inflation_annual=0.05,
        risk_profile="medium",
        currency="USD",
    )
    out = graph.invoke({"user_text": "Can I reach 100k in 10 years?", "user_profile": UserProfile(), "goal": goal})
    assert "Goal projection" in out["final"].answer_md
    tool_names = [t.tool_name for t in out.get("tool_calls", [])]
    assert "QUANT_COMPUTE" in tool_names
