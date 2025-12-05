import streamlit as st
from utils import ask_medbot, load_css, MEDICAL_PROMPT

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")
load_css()

# Sidebar Navigation (Back to Home)
if st.sidebar.button("🏠 Back to Home"):
    st.switch_page("app.py")

st.title("💬 Chat with Doctory")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am Doctory. How can I help you?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ask_medbot(prompt, MEDICAL_PROMPT)
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
