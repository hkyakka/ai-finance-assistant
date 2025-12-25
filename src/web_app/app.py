import streamlit as st
import uuid
from src.core.config import SETTINGS
from src.utils.logging import setup_logging
from src.workflow.graph import build_graph
from src.core.schemas import UserProfile
from src.pages import chat, portfolio, market, goals

# Setup logging
setup_logging(SETTINGS.log_level)

# Initialize the LangGraph
st.set_page_config(page_title="AI Finance Assistant", layout="wide")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "graph" not in st.session_state:
    st.session_state["graph"] = build_graph()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Portfolio", "Market", "Goals"])

with tab1:
    chat.render()

with tab2:
    portfolio.render()

with tab3:
    market.render()

with tab4:
    goals.render()
