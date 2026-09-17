import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from agents import Head_Agent

load_dotenv(Path(__file__).parent / ".env")


def init_agent_once():
    if "agent" not in st.session_state:
        st.session_state.agent = Head_Agent(
            openai_key=os.getenv("OPENAI_API_KEY"),
            pinecone_key=os.getenv("PINECONE_API_KEY"),
        )


def main():
    st.set_page_config(page_title="ML Tutor — Multi-Agent RAG Chatbot", layout="centered")
    st.title("ML Tutor — Multi-Agent RAG Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    init_agent_once()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask a question about the ML document...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, debug = st.session_state.agent.handle_query(
                    st.session_state.messages, prompt
                )

            st.markdown(reply)

            with st.expander("Debug info", expanded=False):
                st.json(debug)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
