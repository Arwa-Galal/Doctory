import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="MedBot - AI Medical Assistant",
    page_icon="🩺",
    layout="centered"
)

# --- Configuration & Sidebar ---
st.sidebar.title("⚙️ Configuration")

# 1. Get API Key from Sidebar (Safer than hardcoding)
api_key = st.sidebar.text_input("Enter Google API Key", type="password", help="Get your key from Google AI Studio")

# 2. Define the Model URL
# We use the specific model version from your notebook
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

# --- System Prompt (The Brain) ---
medical_system_prompt = """
You are "MedBot," an AI assistant designed to provide helpful, general-purpose medical information.
Your persona is professional, empathetic, and clear.

**Your core responsibilities are:**
1.  **Answer Clearly:** Provide accurate, easy-to-understand explanations for medical questions.
2.  **Be Informative, Not Diagnostic:** You can explain what conditions are, what symptoms are, and describe general treatment options. You MUST NOT diagnose, provide treatment plans, or interpret personal medical results.
3.  **Prioritize Safety:** If a user's query sounds like a medical emergency (e.g., "chest pain," "difficulty breathing," "severe bleeding"), your *first and only* response should be to advise them to seek immediate emergency medical help.

**CRITICAL SAFETY RULE:**
You MUST conclude every single response (except for emergency deflections) with the following disclaimer, formatted exactly like this:

---
*Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only. Please consult a qualified healthcare provider for medical advice, diagnosis, or treatment.*
"""

# --- Helper Function ---
def ask_medbot(user_query, system_prompt):
    """
    Sends a query to the Gemini API via REST requests.
    """
    if not api_key:
        return "⚠️ Error: Please enter your API Key in the sidebar to proceed."

    # Construct the API payload
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": user_query}
                ]
            }
        ],
        "systemInstruction": {
            "parts": [
                {"text": system_prompt}
            ]
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Check for HTTP errors
        result = response.json()

        # Extract text
        if "candidates" in result and result["candidates"]:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Error: No valid response from API."

    except requests.exceptions.RequestException as e:
        return f"Connection Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Main UI Logic ---
st.title("🩺 MedBot")
st.caption("A safety-focused medical AI assistant powered by Gemini")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am MedBot. I can answer general medical questions. How can I help you today?"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture User Input
if prompt := st.chat_input("Ask a medical question..."):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = ask_medbot(prompt, medical_system_prompt)
            st.markdown(response_text)
    
    # 4. Add Assistant Response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
