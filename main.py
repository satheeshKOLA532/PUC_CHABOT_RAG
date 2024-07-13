import streamlit as st
from llm import run_qa
import os
import numpy as np
import time

# Function to translate roles between llm model and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

def main():
    # Add CSS to change background color and align content to the left
    html_css= """
    <style>
    .main {
        background-color: #f0f2f6; /* Change this to your desired background color */
        display: flex;
        justify-content: flex-start;
    }
    .stChatMessage {
        display: flex;
        justify-content: flex-start;
        background-color: #f0f2f6;
    }
    </style>
    """
    st.markdown(html_css,unsafe_allow_html=True)
    st.title("let's connect with Bio Brain...")

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = {
            "history": []
        }

    # Display the chat history
    for message in st.session_state.chat_session["history"]:
        with st.chat_message(translate_role_for_streamlit(message["role"])):
            st.markdown(message["content"])

    # Text input for user's message at the bottom of the page
    user_message = st.chat_input("Ask RAG...")
    if user_message:
        # Display user message in chat
        st.chat_message("user").markdown(user_message)
        st.session_state.chat_session["history"].append({"role": "user", "content": user_message})
        # Process for English
        response = process_text_message_eng(user_message)
        st.session_state.chat_session["history"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

def process_text_message_eng(message):
    # Get chat history from session state
    chat_history = st.session_state.chat_session["history"]
    response = run_qa(message)
    return response

if __name__ == "__main__":
    main()
