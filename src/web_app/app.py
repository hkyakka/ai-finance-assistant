import streamlit as st
import uuid
from src.core.config import SETTINGS
from src.utils.logging import setup_logging
from src.core.schemas import UserProfile
from src.pages import chat, portfolio, market, goals, tax, news

# Setup logging
setup_logging(SETTINGS.log_level)

st.set_page_config(page_title="AI Finance Assistant", layout="wide")

# Session initialization
def _init_session() -> None:
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("turn_id", 0)
    st.session_state.setdefault("chat", [])  # list[dict]
    st.session_state.setdefault("last_chat_meta", {})
    st.session_state.setdefault("user_profile", UserProfile())

_init_session()

# Sidebar for user profile
with st.sidebar:
    st.subheader("Profile")
    up: UserProfile = st.session_state["user_profile"]

    up.currency = st.text_input("Currency", value=up.currency)
    up.country = st.text_input("Country (optional)", value=up.country or "") or None
    up.risk_tolerance = st.selectbox("Risk tolerance", ["low", "medium", "high"], index=["low", "medium", "high"].index(up.risk_tolerance))
    up.knowledge_level = st.selectbox(
        "Knowledge level",
        ["beginner", "intermediate", "advanced"],
        index=["beginner", "intermediate", "advanced"].index(up.knowledge_level),
    )
    st.session_state["user_profile"] = up

    st.divider()
    st.caption(f"Session: {st.session_state['session_id']}")
    st.caption(f"Turn: {st.session_state['turn_id']}")

# Main UI
st.title("AI Finance Assistant")

tab_chat, tab_portfolio, tab_market, tab_goals, tab_tax, tab_news = st.tabs(["Chat", "Portfolio", "Market", "Goals", "Tax", "News"])

with tab_chat:
    chat.render()

with tab_portfolio:
    portfolio.render()

with tab_market:
    market.render()

with tab_goals:
    goals.render()

with tab_tax:
    tax.render()

with tab_news:
    news.render()
