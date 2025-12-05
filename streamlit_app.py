import streamlit as st
from streamlit_option_menu import option_menu
import requests
import json
import joblib
import numpy as np
from PIL import Image
import os

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="MedBot Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to hide default menu and modernize the look
st.markdown("""
    <style>
    /* Hide default Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide the default sidebar navigation list */
    div[data-testid="stSidebarNav"] {display: none;}
    
    /* Card Styling */
    .css-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #00ADB5;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR & CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100) # Placeholder Logo
    st.title("MedBot Pro")
    
    # Secure API Key Input
    api_key = st.text_input("🔑 Google API Key", type="password", help="Paste your AIza key here")
    
    st.markdown("---")
    
    # MODERN NAVIGATION MENU
    selected = option_menu(
        menu_title="Navigation",
        options=["AI Chatbot", "Diabetes Test", "Pneumonia Check"],
        icons=["chat-dots-fill", "activity", "lungs"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "#00ADB5", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#e1e1e1"},
            "nav-link-selected": {"background-color": "#00ADB5"},
        }
    )
    
    st.markdown("---")
    st.info("⚠️ **Disclaimer:** AI assistance is for educational purposes only.")

# --- 3. HELPER FUNCTIONS (AI & MODELS) ---

# Gemini API URL
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

def ask_medbot(user_query, system_prompt):
    """Call Gemini API via REST"""
    if not api_key: return "⚠️ Please enter your API Key in the sidebar."
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200: return f"Error: {response.text}"
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Connection Error: {e}"

# Load ML Models
@st.cache_resource
def load_models():
    models = {}
    try:
        # Update these paths to match your actual file structure
        models['diabetes'] = joblib.load('models/diabetes_model_package/diabetes_ensemble_model.joblib')
    except:
        pass # Silently fail if file not found (for demo purposes)
    return models

loaded_models = load_models()

# System Prompt for the AI
medical_prompt = """
You are MedBot, a professional medical AI assistant. 
Answer questions clearly and empathetically. 
If a user asks about a specific medical test result (like "Diabetic"), explain what it means and suggest lifestyle changes.
ALWAYS end with a disclaimer that you are an AI, not a doctor.
"""

# --- 4. PAGE CONTENT LOGIC ---

# === SECTION 1: CHATBOT ===
if selected == "AI Chatbot":
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
                reply = ask_medbot(prompt, medical_prompt)
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

# === SECTION 2: DIABETES TEST ===
elif selected == "Diabetes Test":
    st.title("🩸 Diabetes Risk Assessment")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 1, 120, 30)
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose", 0, 500, 100)
    with col2:
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 79)
    with col3:
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Family History (DPF)", 0.0, 3.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Analyze Result", type="primary"):
        if 'diabetes' in loaded_models:
            # 1. Predict
            input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            try:
                prediction = loaded_models['diabetes'].predict(input_data)[0]
                result_str = "Diabetic (High Risk)" if prediction == 1 else "Healthy (Low Risk)"
                color = "red" if prediction == 1 else "green"
                
                # 2. Ask AI to explain
                analysis_prompt = f"The patient has tested: {result_str}. Glucose: {glucose}, BMI: {bmi}. Explain this result briefly."
                ai_explanation = ask_medbot(analysis_prompt, medical_prompt)
                
                # 3. Display
                st.markdown(f"### Result: :{color}[{result_str}]")
                st.info(f"👨‍⚕️ **Dr. AI Analysis:**\n\n{ai_explanation}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.warning("⚠️ Model file not found. Please ensure 'models/diabetes_model_package/' exists.")

# === SECTION 3: PNEUMONIA CHECK ===
elif selected == "Pneumonia Check":
    st.title("🫁 Pneumonia X-Ray Check")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        st.image(uploaded_file, width=300)
        if st.button("Analyze Image"):
            st.info("Image analysis model would run here.")
            # Placeholder for ONNX logic
            # Once you have the result, you can pass it to 'ask_medbot' like we did in Diabetes.
