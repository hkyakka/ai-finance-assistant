import streamlit as st
import uuid
from src.core.config import SETTINGS
from src.utils.logging import setup_logging
from src.workflow.graph import build_graph
from src.core.schemas import UserProfile

# Setup logging
setup_logging(SETTINGS.log_level)

# Initialize the LangGraph
st.set_page_config(page_title="AI Finance Assistant", layout="wide")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "graph" not in st.session_state:
    st.session_state["graph"] = build_graph()

user_text = st.text_input("Ask:", "Analyze my portfolio risk")

if st.button("Run"):
    state = {
        "user_text": user_text,
        "user_profile": UserProfile(),
        "session_id": st.session_state["session_id"],
        "turn_id": 1,
    }
    out = st.session_state["graph"].invoke(state)
    st.markdown(out["final"].answer_md)
    st.write("Trace:", out.get("agent_trace"))
    st.write("Request ID:", out.get("request_id"))
    st.write("Session ID:", out.get("session_id"))
