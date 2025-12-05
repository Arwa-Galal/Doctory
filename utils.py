# utils.py
import requests
import json
import streamlit as st

# --- CONFIGURATION ---
# Your API Key
API_KEY = "AIzaSyBxYCfAwsyhbhiA8EQd6dcn-RdsZQ9xtZ8"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- SHARED CSS ---
def load_css():
    st.markdown("""
        <style>
        /* Medical Blue Theme */
        .css-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border-left: 5px solid #0277BD;
        }
        div.stButton > button {
            background-color: #0277BD; color: white; border: none; border-radius: 8px;
        }
        div.stButton > button:hover {
            background-color: #01579B; color: white;
        }
        /* Hide default header */
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# --- AI FUNCTION ---
def ask_medbot(user_query, system_prompt):
    if not API_KEY: return "⚠️ API Key missing."
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200: return f"Error: {response.text}"
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Connection Error: {e}"

# --- MEDICAL PROMPT ---
MEDICAL_PROMPT = """
You are MedBot, a professional medical AI assistant. 
Answer questions clearly and empathetically. 
If a user asks about a specific medical test result, explain what it means and suggest lifestyle changes.
ALWAYS end with a disclaimer that you are an AI, not a doctor.
"""
