import streamlit as st
import uuid
from src.agents.tax_agent import TaxAgent
from src.core.schemas import AgentRequest

def render():
    st.subheader("Tax Information Agent")

    user_text = st.text_input("Ask a tax-related question (e.g., 'What is STCG?')")

    if st.button("Get Tax Info"):
        if user_text:
            with st.spinner("Finding tax information..."):
                req = AgentRequest(
                    request_id=str(uuid.uuid4()),
                    session_id=st.session_state["session_id"],
                    turn_id=int(st.session_state["turn_id"]) + 1,
                    user_text=user_text,
                    user_profile=st.session_state["user_profile"],
                )

                tax_agent = TaxAgent()
                resp = tax_agent.run(req)

                st.session_state["tax_agent_response"] = resp.answer_md
                st.session_state["turn_id"] += 1

    if "tax_agent_response" in st.session_state:
        st.markdown(st.session_state["tax_agent_response"])
