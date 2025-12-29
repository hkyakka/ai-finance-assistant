from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from src.core.schemas import AgentRequest, AgentResponse, ChatMessage, UserProfile
from src.workflow.graph import build_graph

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


def _run_chat_agent(user_text: str, source_tab: str) -> Tuple[AgentResponse, Dict[str, Any]]:
    # Build multi-turn context
    turns = [ChatTurn(**t) for t in st.session_state["chat"]]
    msgs = _to_chat_messages(turns[-20:])

    # Choose agent
    graph = build_graph()
    state = graph.invoke({
        "user_text": user_text,
        "user_profile": st.session_state["user_profile"],
        "source_tab": source_tab
    })
    resp = state.get("final")

    trace = state.get("agent_trace", [])
    route = state.get("route", "")

    return resp, {"trace": trace, "route": route}
