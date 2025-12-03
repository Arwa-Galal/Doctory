import streamlit as st
from utils import load_all_models, MODELS # Import shared utilities
import streamlit as st
import google.generativeai as genai
import os

# 1. إعداد الشكل (لازم يكون أول سطر)
st.set_page_config(page_title="Doctory", page_icon="🩺", layout="centered")

# 2. إعداد Gemini (حطي مفتاحك هنا)
genai.configure(api_key="AIzaSyAg-7Wu_mCF-z9P-KEbkjpQEb7B3PB_hxo")
model_ai = genai.GenerativeModel('gemini-pro')

# 3. كود CSS عشان الشكل يبقى حلو (بلاش نغير الـ Layout، هنغير الألوان بس)
st.markdown("""
    <style>
    .stApp {background-color: #f0f8ff;} /* لون خلفية هادي */
    .stButton>button {
        background-color: #00ADB5; color: white; border-radius: 10px; width: 100%;
    }
    .result-box {
        padding: 20px; background-color: white; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #00ADB5;
    }
    </style>
""", unsafe_allow_html=True)


# Configure the main application page settings
st.set_page_config(
    page_title="Doctory AI Medical Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Call the model loader once, this will display an error if models/ is missing
if MODELS is None:
    st.error("Application failed to initialize. See console for model loading errors.")
    st.stop() 

# --- HOME PAGE CONTENT ---
st.title("AI Medical Prediction Dashboard")
st.header("Welcome to Doctory AI 🩺")

st.markdown("""
### Use the Sidebar to Select a Specialized Module:
* **AI Chatbot:** Connect to your custom fine-tuned model (Gemma/Gemini) for Q&A.
* **Prediction Modules:** Run local machine learning models for diagnosis based on images or biometric data.
---

### Disclaimer:
**This tool is for informational and educational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
""")
