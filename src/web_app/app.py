import streamlit as st

from src.workflow.graph import build_graph
from src.core.schemas import UserProfile

graph = build_graph()
state = {"user_text": "Analyze my portfolio risk", "user_profile": UserProfile()}
out = graph.invoke(state)
print(out["final"].answer_md)
print(out.get("agent_trace"))

st.set_page_config(page_title="AI Finance Assistant", layout="wide")
st.title("AI Finance Assistant")
st.write("✅ Environment setup complete.")
st.write("✅ Required packages installed.")
st.write("✅ Basic Streamlit app structure in place.")