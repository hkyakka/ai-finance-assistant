from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, cast

from langgraph.graph import StateGraph, END

from src.core.router import Router
from src.core.schemas import (
    AgentResponse,
    RouterDecision,
    UserProfile,
    PortfolioInput,
    GoalInput,
    RagResult,
    QuantResult,
)


# -------------------------
# LangGraph State
# -------------------------

class GraphState(TypedDict, total=False):
    # user input
    user_text: str

    # session / profile
    user_profile: UserProfile
    memory_summary: str

    # routing
    route: RouterDecision
    agent_trace: List[str]

    # domain inputs (optional, filled by UI or previous turns)
    portfolio: PortfolioInput
    goal: GoalInput

    # shared subsystem outputs
    rag_result: RagResult
    quant_result: QuantResult
    market_payload: Dict[str, Any]  # quotes, series, freshness, etc.

    # final response
    responses: List[AgentResponse]
    final: AgentResponse

    # error
    error: Dict[str, Any]


# -------------------------
# Placeholder subsystem nodes
# -------------------------

def _append_trace(state: GraphState, node_name: str) -> None:
    trace = state.get("agent_trace") or []
    trace.append(node_name)
    state["agent_trace"] = trace


def router_node(state: GraphState) -> GraphState:
    _append_trace(state, "RouterNode")
    router = Router()
    decision = router.classify(state.get("user_text", ""))
    state["route"] = decision
    return state


def rag_retrieve_node(state: GraphState) -> GraphState:
    _append_trace(state, "RAGRetrieveNode")
    # TODO: call your real retriever here (FAISS/Chroma/Pinecone)
    # state["rag_result"] = retriever.retrieve(query=state["user_text"], top_k=5, mmr=True)
    state["rag_result"] = RagResult(query=state.get("user_text", ""), chunks=[])
    return state


def market_data_node(state: GraphState) -> GraphState:
    _append_trace(state, "MarketDataNode")
    # TODO: call AlphaVantage/yFinance wrapper + caching here
    state["market_payload"] = {
        "quotes": [],
        "series": [],
        "freshness": {"as_of": None, "from_cache": False, "provider": None},
        "warnings": [],
    }
    return state


def quant_compute_node(state: GraphState) -> GraphState:
    _append_trace(state, "QuantComputeNode")
    # TODO: deterministic computations ONLY (pandas/numpy/SQL)
    # Example:
    #   quant_result = quant_engine.compute_portfolio_metrics(state["portfolio"], quotes=state["market_payload"])
    #   state["quant_result"] = quant_result
    state["quant_result"] = QuantResult(
        metrics={},
        tables={},
        chart_data={},
        warnings=["Quant engine not implemented yet"],
        confidence="low",
    )
    return state


# -------------------------
# Placeholder agent nodes
# -------------------------

def _push_response(state: GraphState, resp: AgentResponse) -> None:
    responses = state.get("responses") or []
    responses.append(resp)
    state["responses"] = responses


def finance_qa_agent_node(state: GraphState) -> GraphState:
    _append_trace(state, "FinanceQANode")
    # TODO: LLM answer using rag_result chunks + citations
    resp = AgentResponse(
        agent_name="FinanceQAAgent",
        answer_md="(TODO) Finance Q&A response here. Use RAG citations.",
        citations=[],
        confidence="low",
    )
    _push_response(state, resp)
    return state


def tax_agent_node(state: GraphState) -> GraphState:
    _append_trace(state, "TaxNode")
    resp = AgentResponse(
        agent_name="TaxEducationAgent",
        answer_md="(TODO) Tax education response here (RAG + disclaimers).",
        confidence="low",
    )
    _push_response(state, resp)
    return state


def market_agent_node(state: GraphState) -> GraphState:
    _append_trace(state, "MarketNode")
    resp = AgentResponse(
        agent_name="MarketAnalysisAgent",
        answer_md="(TODO) Market analysis response here (quotes/trend + freshness).",
        data_freshness=state.get("market_payload", {}).get("freshness"),
        warnings=state.get("market_payload", {}).get("warnings", []),
        confidence="low",
    )
    _push_response(state, resp)
    return state


def portfolio_agent_node(state: GraphState) -> GraphState:
    _append_trace(state, "PortfolioNode")
    resp = AgentResponse(
        agent_name="PortfolioAnalysisAgent",
        answer_md="(TODO) Portfolio analysis response here (interpret quant_result).",
        charts_payload=cast(Optional[Dict[str, Any]], state.get("quant_result", {}).chart_data if state.get("quant_result") else None),
        warnings=(state.get("quant_result").warnings if state.get("quant_result") else []),  # type: ignore[union-attr]
        confidence="low",
    )
    _push_response(state, resp)
    return state


def goal_agent_node(state: GraphState) -> GraphState:
    _append_trace(state, "GoalNode")
    resp = AgentResponse(
        agent_name="GoalPlanningAgent",
        answer_md="(TODO) Goal planning response here (projection from quant_result).",
        charts_payload=cast(Optional[Dict[str, Any]], state.get("quant_result", {}).chart_data if state.get("quant_result") else None),
        warnings=(state.get("quant_result").warnings if state.get("quant_result") else []),  # type: ignore[union-attr]
        confidence="low",
    )
    _push_response(state, resp)
    return state


def news_agent_node(state: GraphState) -> GraphState:
    _append_trace(state, "NewsNode")
    resp = AgentResponse(
        agent_name="NewsSynthesizerAgent",
        answer_md="(TODO) News synthesis response here (summarize + why it matters).",
        confidence="low",
    )
    _push_response(state, resp)
    return state


def validator_node(state: GraphState) -> GraphState:
    _append_trace(state, "ValidatorNode")
    # TODO: enforce:
    # - education-only disclaimer present
    # - citations present for RAG answers
    # - warnings shown if using cached/stale
    return state


def composer_node(state: GraphState) -> GraphState:
    _append_trace(state, "ComposerNode")
    responses = state.get("responses") or []
    if not responses:
        state["final"] = AgentResponse(
            agent_name="System",
            answer_md="I couldn’t generate a response. Please try again.",
            confidence="low",
        )
        return state

    # Simple composition: choose first response as final.
    # Later: merge multiple responses into a single narrative.
    state["final"] = responses[0]
    return state


def fallback_node(state: GraphState) -> GraphState:
    _append_trace(state, "FallbackNode")
    err = state.get("error") or {"code": "UNKNOWN", "message": "Unexpected error"}
    state["final"] = AgentResponse(
        agent_name="System",
        answer_md=f"⚠️ Something went wrong: {err.get('message')}",
        warnings=[str(err)],
        confidence="low",
    )
    return state


# -------------------------
# Conditional routing helpers
# -------------------------

def _route_intent(state: GraphState) -> str:
    decision: RouterDecision = state.get("route")  # type: ignore[assignment]
    if not decision:
        return "CLARIFY"
    return decision.intent


# -------------------------
# Build graph
# -------------------------

def build_graph():
    g = StateGraph(GraphState)

    # Nodes
    g.add_node("router", router_node)
    g.add_node("rag", rag_retrieve_node)
    g.add_node("market_data", market_data_node)
    g.add_node("quant", quant_compute_node)

    g.add_node("finance_qa", finance_qa_agent_node)
    g.add_node("tax", tax_agent_node)
    g.add_node("market", market_agent_node)
    g.add_node("portfolio", portfolio_agent_node)
    g.add_node("goal", goal_agent_node)
    g.add_node("news", news_agent_node)

    g.add_node("validator", validator_node)
    g.add_node("composer", composer_node)
    g.add_node("fallback", fallback_node)

    # Entry
    g.set_entry_point("router")

    # Conditional edges from router
    g.add_conditional_edges(
        "router",
        _route_intent,
        {
            "FINANCE_QA": "rag",
            "TAX": "rag",
            "MARKET": "market_data",
            "PORTFOLIO": "market_data",  # prices often needed
            "GOAL": "quant",             # can compute without prices if only SIP projection
            "NEWS": "news",
            "CLARIFY": "composer",
        },
    )

    # After RAG, go to the right agent
    def _post_rag_next(state: GraphState) -> str:
        intent = state["route"].intent
        return "finance_qa" if intent == "FINANCE_QA" else "tax"

    g.add_conditional_edges("rag", _post_rag_next, {"finance_qa": "finance_qa", "tax": "tax"})

    # After market data:
    # - MARKET -> market agent
    # - PORTFOLIO -> quant -> portfolio agent
    def _post_market_next(state: GraphState) -> str:
        intent = state["route"].intent
        return "market" if intent == "MARKET" else "quant"

    g.add_conditional_edges("market_data", _post_market_next, {"market": "market", "quant": "quant"})

    # After quant:
    # - PORTFOLIO -> portfolio agent
    # - GOAL -> goal agent
    def _post_quant_next(state: GraphState) -> str:
        intent = state["route"].intent
        return "portfolio" if intent == "PORTFOLIO" else "goal"

    g.add_conditional_edges("quant", _post_quant_next, {"portfolio": "portfolio", "goal": "goal"})

    # After any agent node -> validator -> composer -> end
    for node in ["finance_qa", "tax", "market", "portfolio", "goal", "news"]:
        g.add_edge(node, "validator")

    g.add_edge("validator", "composer")
    g.add_edge("composer", END)

    # fallback path if you later wrap nodes with try/except and set state["error"]
    g.add_edge("fallback", END)

    return g.compile()