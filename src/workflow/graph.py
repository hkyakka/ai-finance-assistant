from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from langgraph.graph import StateGraph, END

from src.core.router import Router
from src.core.schemas import (
    AgentResponse,
    AgentTraceEvent,
    ToolCall,
    ErrorEnvelope,
    RouterDecision,
    UserProfile,
    PortfolioInput,
    GoalInput,
    RagResult,
    QuantResult,
)
from src.workflow.state import GraphState
import uuid
import os
from src.core.config import SETTINGS
from src.utils.logging import get_logger, set_log_context, set_agent

logger = get_logger("workflow")

def _append_trace(state: GraphState, node_name: str, agent: str = "-", info: Optional[Dict[str, Any]] = None) -> None:
    """Append node trace (string list + structured events) for demo/debug."""
    trace = state.get("agent_trace") or []
    trace.append(node_name)
    state["agent_trace"] = trace

    events = state.get("trace_events") or []
    events.append(AgentTraceEvent(node=node_name, agent=agent, info=info or {}))
    state["trace_events"] = events

    logger.info(f"trace_node={node_name} agent={agent}")

def _tool_start(state: GraphState, call_id: str, tool_name: str, args: Dict[str, Any]) -> None:
    calls = state.get("tool_calls") or []
    calls.append(ToolCall(call_id=call_id, tool_name=cast(Any, tool_name), args=args))
    state["tool_calls"] = calls

def _tool_end_ok(state: GraphState, call_id: str) -> None:
    calls = state.get("tool_calls") or []
    for c in calls:
        if c.call_id == call_id:
            c.status = "ok"
            c.ended_at = datetime.utcnow()
            break
    state["tool_calls"] = calls

def _tool_end_error(state: GraphState, call_id: str, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
    calls = state.get("tool_calls") or []
    for c in calls:
        if c.call_id == call_id:
            c.status = "error"
            c.ended_at = datetime.utcnow()
            c.error = ErrorEnvelope(code=code, message=message, details=details, retriable=False)
            break
    state["tool_calls"] = calls

def router_node(state: GraphState) -> GraphState:
    if not state.get("request_id"):
        state["request_id"] = str(uuid.uuid4())
    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())
    if not state.get("turn_id"):
        state["turn_id"] = 1

    set_log_context(
        request_id=state["request_id"],
        session_id=state["session_id"],
        turn_id=str(state["turn_id"]),
        agent="RouterNode"
    )

    _append_trace(state, "RouterNode", agent="RouterNode")
    router = Router()
    decision = router.classify(state.get("user_text", ""))
    state["route"] = decision
    return state



def rag_retrieve_node(state: GraphState) -> GraphState:
    """
    Stage 1 placeholder:
    - Logs + traces that the RAG node ran
    - Stores an empty RagResult so downstream agents can rely on the key existing
    Stage 4 will replace this with real retrieval (FAISS/Chroma/Pinecone + citations).
    """
    # Make sure logging context is correct for this node
    set_agent("RAGRetrieveNode")
    _append_trace(state, "RAGRetrieveNode", agent="RAGRetrieveNode")

    user_text = (state.get("user_text") or "").strip()
    call_id = str(uuid.uuid4())
    _tool_start(state, call_id=call_id, tool_name="RAG_RETRIEVE", args={"query": user_text})

    # Defensive: if user_text missing, return empty result + warning
    if not user_text:
        state["rag_result"] = RagResult(query="", chunks=[])
        # Optional: you can store warnings in a common place later
        state["error"] = ErrorEnvelope(code="EMPTY_QUERY", message="No user_text provided to RAG node.")
        return state
    # Stage 4: real retrieval
    try:
        from src.rag.retriever import Retriever
        retriever = Retriever()
        rag = retriever.retrieve(
            query=user_text,
            top_k=SETTINGS.rag_top_k,
            use_mmr=SETTINGS.rag_use_mmr,
            mmr_lambda=float(os.getenv("RAG_MMR_LAMBDA", "0.7")),
            min_score=SETTINGS.rag_min_score,
        )
        state["rag_result"] = rag
        _tool_end_ok(state, call_id)
        return state
    except Exception as e:
        state["rag_result"] = RagResult(query=user_text, chunks=[])
        state["error"] = ErrorEnvelope(code="RAG_RETRIEVE_FAILED", message=str(e))
        _tool_end_error(state, call_id, error_code="RAG_RETRIEVE_FAILED", message=str(e))
        return state





def market_data_node(state: GraphState) -> GraphState:
    _append_trace(state, "MarketDataNode", agent="MarketDataNode")
    call_id = str(uuid.uuid4())
    _tool_start(state, call_id=call_id, tool_name="MARKET_QUOTE", args={"query": state.get("user_text","")})
    # TODO: call AlphaVantage/yFinance wrapper + caching here
    state["market_payload"] = {
        "quotes": [],
        "series": [],
        "freshness": {"as_of": None, "from_cache": False, "provider": None},
        "warnings": [],
    }
    _tool_end_ok(state, call_id)
    return state


def quant_compute_node(state: GraphState) -> GraphState:
    _append_trace(state, "QuantComputeNode", agent="QuantComputeNode")
    call_id = str(uuid.uuid4())
    _tool_start(state, call_id=call_id, tool_name="QUANT_COMPUTE", args={"task": "placeholder"})
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
    _tool_end_ok(state, call_id)
    return state


# -------------------------
# Placeholder agent nodes
# -------------------------

def _push_response(state: GraphState, resp: AgentResponse) -> None:
    responses = state.get("responses") or []
    responses.append(resp)
    state["responses"] = responses


def finance_qa_agent_node(state: GraphState) -> GraphState:
    _append_trace(state, "FinanceQANode", agent="FinanceQANode")
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
    _append_trace(state, "TaxNode", agent="TaxNode")
    resp = AgentResponse(
        agent_name="TaxEducationAgent",
        answer_md="(TODO) Tax education response here (RAG + disclaimers).",
        confidence="low",
    )
    _push_response(state, resp)
    return state


def market_agent_node(state: GraphState) -> GraphState:
    _append_trace(state, "MarketNode", agent="MarketNode")
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
    _append_trace(state, "PortfolioNode", agent="PortfolioNode")
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
    _append_trace(state, "GoalNode", agent="GoalNode")
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
    _append_trace(state, "NewsNode", agent="NewsNode")
    resp = AgentResponse(
        agent_name="NewsSynthesizerAgent",
        answer_md="(TODO) News synthesis response here (summarize + why it matters).",
        confidence="low",
    )
    _push_response(state, resp)
    return state


def validator_node(state: GraphState) -> GraphState:
    _append_trace(state, "ValidatorNode", agent="ValidatorNode")
    # TODO: enforce:
    # - education-only disclaimer present
    # - citations present for RAG answers
    # - warnings shown if using cached/stale
    return state


def composer_node(state: GraphState) -> GraphState:
    _append_trace(state, "ComposerNode", agent="ComposerNode")
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
    _append_trace(state, "FallbackNode", agent="FallbackNode")
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
