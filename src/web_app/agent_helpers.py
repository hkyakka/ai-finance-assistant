from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.goal_agent import GoalAgent
from src.agents.market_agent import MarketAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.core.router import Router
from src.core.schemas import AgentRequest, AgentResponse, ChatMessage, UserProfile

@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str
    meta: Dict[str, Any]


def _to_chat_messages(turns: List[ChatTurn]) -> List[ChatMessage]:
    msgs: List[ChatMessage] = []
    for t in turns:
        msgs.append(ChatMessage(role=t.role, content=t.content))
    return msgs


def _mk_req(
    *,
    user_text: str,
    user_profile: UserProfile,
    messages: List[ChatMessage],
    extra: Optional[Dict[str, Any]] = None,
) -> AgentRequest:
    extra = extra or {}
    st.session_state["turn_id"] += 1
    return AgentRequest(
        request_id=str(uuid.uuid4()),
        session_id=st.session_state["session_id"],
        turn_id=int(st.session_state["turn_id"]),
        user_text=user_text,
        user_profile=user_profile,
        messages=messages,
        **extra,
    )


def _run_chat_agent(user_text: str) -> Tuple[AgentResponse, Dict[str, Any]]:
    router = Router()
    decision = router.classify(user_text)

    # Build multi-turn context
    turns = [ChatTurn(**t) for t in st.session_state["chat"]]
    msgs = _to_chat_messages(turns[-20:])

    trace: List[str] = ["Router", f"intent={decision.intent}"]

    req_extra: Dict[str, Any] = {"route": decision}

    # Choose agent
    if decision.intent == "MARKET":
        trace.append("MarketAgent")
        req = _mk_req(user_text=user_text, user_profile=st.session_state["user_profile"], messages=msgs, extra=req_extra)
        resp = MarketAgent().run(req)
        return resp, {"trace": trace, "route": decision.model_dump()}

    if decision.intent == "PORTFOLIO":
        trace.append("PortfolioAgent")
        req = _mk_req(user_text=user_text, user_profile=st.session_state["user_profile"], messages=msgs, extra=req_extra)
        resp = PortfolioAgent().run(req)
        return resp, {"trace": trace, "route": decision.model_dump()}

    if decision.intent == "GOAL":
        trace.append("GoalAgent")
        req = _mk_req(user_text=user_text, user_profile=st.session_state["user_profile"], messages=msgs, extra=req_extra)
        resp = GoalAgent().run(req)
        return resp, {"trace": trace, "route": decision.model_dump()}

    # default: finance education Q&A
    trace.append("FinanceQAAgent")
    req = _mk_req(user_text=user_text, user_profile=st.session_state["user_profile"], messages=msgs, extra=req_extra)
    resp = FinanceQAAgent().run(req)
    return resp, {"trace": trace, "route": decision.model_dump()}
