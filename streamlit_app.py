import streamlit as st
from utils import ask_medbot, load_css, MEDICAL_PROMPT

# Page Config
st.set_page_config(page_title="MedBot Pro", page_icon="🩺", layout="wide")
load_css()

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
st.sidebar.title("Dr. AI Assistant")
st.sidebar.info("Navigate to other tests using the sidebar above.")

# --- CHATBOT INTERFACE ---
st.title("💬 Dr. AI Assistant")
st.caption("Ask general medical questions here.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

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
