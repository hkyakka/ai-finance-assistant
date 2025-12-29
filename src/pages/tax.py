import streamlit as st
from src.web_app.agent_helpers import _run_chat_agent

def render():
    st.subheader("Tax Information Agent")
    
    user_text = st.text_input("Ask a tax-related question (e.g., 'What is STCG?')")

    if st.button("Get Tax Info"):
        if user_text:
            with st.spinner("Finding tax information..."):
                resp, _ = _run_chat_agent(user_text, "Tax")
                st.session_state["tax_agent_response"] = resp.answer_md

    if "tax_agent_response" in st.session_state:
        st.markdown(st.session_state["tax_agent_response"])
