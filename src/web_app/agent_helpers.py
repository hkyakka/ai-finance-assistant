from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.core.schemas import AgentRequest, AgentResponse, ChatMessage, UserProfile
from src.workflow.graph import build_graph

from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class ChatTurn:
    role: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def _mk_messages() -> List[ChatMessage]:
    # Convert Streamlit chat history to schema messages
    msgs = []
    for t in st.session_state.get("chat", []):
        role = t.get("role")
        content = t.get("content")
        if role and content:
            msgs.append(ChatMessage(role=role, content=content))
    return msgs


def run_query(
    *,
    user_text: str,
    source_tab: str,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Tuple[AgentResponse, Dict[str, Any]]:
    """Single entry point for all tabs.

    Uses LangGraph orchestration; routing considers source_tab (deterministic for most tabs).
    """
    g = _get_graph()

    state: Dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "session_id": st.session_state.get("session_id") or "local",
        "turn_id": int(st.session_state.get("turn_id") or 0) + 1,
        "user_text": user_text,
        "source_tab": source_tab,
        "user_profile": st.session_state.get("user_profile") or UserProfile(),
        "messages": _mk_messages(),
    }
    if extra_state:
        state.update(extra_state)

    out = g.invoke(state)
    resp: AgentResponse = out["final"]

    meta = {
        "trace": out.get("agent_trace") or [],
        "tool_calls": [t.model_dump() for t in (out.get("tool_calls") or [])],
        "route": out.get("route"),
        "router_decision": (out.get("router_decision").model_dump() if out.get("router_decision") else None),
    }
    return resp, meta


def _run_chat_agent(user_text: str) -> Tuple[AgentResponse, Dict[str, Any]]:
    # Back-compat wrapper for existing chat page.
    return run_query(user_text=user_text, source_tab="chat")
