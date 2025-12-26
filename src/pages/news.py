import streamlit as st
import uuid
from src.agents.news_agent import NewsAgent
from src.core.schemas import AgentRequest

def render():
    st.subheader("News Agent")

    user_text = st.text_input("Enter a news topic (e.g., 'Apple earnings')")

    if st.button("Get News"):
        if user_text:
            with st.spinner("Fetching news..."):
                req = AgentRequest(
                    request_id=str(uuid.uuid4()),
                    session_id=st.session_state["session_id"],
                    turn_id=int(st.session_state["turn_id"]) + 1,
                    user_text=user_text,
                    user_profile=st.session_state["user_profile"],
                )

                news_agent = NewsAgent()
                resp = news_agent.run(req)

                st.session_state["news_agent_response"] = resp.answer_md
                st.session_state["turn_id"] += 1

    if "news_agent_response" in st.session_state:
        st.markdown(st.session_state["news_agent_response"])
