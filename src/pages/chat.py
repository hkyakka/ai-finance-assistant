import streamlit as st
from src.web_app.agent_helpers import _run_chat_agent, ChatTurn
from src.web_app.ui_helpers import _render_citations, _render_agent_trace

def render():
    left, right = st.columns([0.68, 0.32], gap="large")

    with left:
        st.subheader("Chat")

        # Render history
        for t in st.session_state["chat"]:
            turn = ChatTurn(**t)
            with st.chat_message(turn.role):
                st.markdown(turn.content)
                if turn.role == "assistant":
                    agent = turn.meta.get("agent_name")
                    if agent:
                        st.caption(f"Answered by: {agent}")
                    # Small inline meta
                    if turn.meta.get("warnings"):
                        st.caption("Warnings: " + ", ".join(turn.meta["warnings"]))

        # Input
        user_text = st.chat_input("Ask about markets, goals, portfolios, or concepts")
        if user_text:
            st.session_state["chat"].append({"role": "user", "content": user_text, "meta": {}})

            resp, meta = _run_chat_agent(user_text)

            st.session_state["last_chat_meta"] = {
                "agent_name": resp.agent_name,
                "citations": [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in (resp.citations or [])],
                "trace": meta.get("trace"),
                "route": meta.get("route"),
                "warnings": resp.warnings or [],
                "data_freshness": resp.data_freshness,
                "raw": resp.model_dump() if hasattr(resp, "model_dump") else {},
            }

            st.session_state["chat"].append(
                {
                    "role": "assistant",
                    "content": resp.answer_md,
                    "meta": {
                        "agent_name": resp.agent_name,
                        "citations": st.session_state["last_chat_meta"]["citations"],
                        "trace": meta.get("trace"),
                        "route": meta.get("route"),
                        "warnings": resp.warnings or [],
                        "data_freshness": resp.data_freshness,
                    },
                }
            )
            st.rerun()

    with right:
        st.subheader("Citations")
        meta = st.session_state.get("last_chat_meta") or {}
        _render_citations(meta.get("citations") or [])

        st.divider()
        st.subheader("Agent trace")
        _render_agent_trace(meta.get("trace"), meta.get("route"))
