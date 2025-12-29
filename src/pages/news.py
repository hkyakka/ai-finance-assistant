import streamlit as st
from src.web_app.agent_helpers import _run_chat_agent

def render():
    st.subheader("News Agent")
    
    user_text = st.text_input("Enter a news topic (e.g., 'Apple earnings')")

    if st.button("Get News"):
        if user_text:
            with st.spinner("Fetching news..."):
                resp, _ = _run_chat_agent(user_text, "News")
                st.session_state["news_agent_response"] = resp.answer_md

    if "news_agent_response" in st.session_state:
        st.markdown(st.session_state["news_agent_response"])
