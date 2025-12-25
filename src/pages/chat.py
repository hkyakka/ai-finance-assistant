import streamlit as st
import uuid
from src.core.schemas import UserProfile

def render():
    st.header("Chat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "answered_by" in message:
                st.markdown(f'---\n*Answered by: {message["answered_by"]}*')
            if "citations" in message:
                with st.expander("Citations"):
                    for citation in message["citations"]:
                        st.markdown(f'- {citation}')


    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            state = {
                "user_text": prompt,
                "user_profile": UserProfile(),
                "session_id": st.session_state["session_id"],
                "turn_id": len(st.session_state.messages),
            }
            out = st.session_state["graph"].invoke(state)

            full_response = out["final"].answer_md
            message_placeholder.markdown(full_response)

            # Answered by and citations
            st.markdown(f'---_Answered by: {out["final"].agent_name}_')
            with st.expander("Citations"):
                for source in out["final"].sources:
                    st.markdown(f'- {source}')

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "answered_by": out["final"].agent_name,
            "citations": out["final"].sources
        })
