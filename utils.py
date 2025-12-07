import streamlit as st
import requests
import json
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image
import io
import os
from streamlit_option_menu import option_menu 

# --- 1. CONFIGURATION ---
API_KEY = "AIzaSyBxYCfAwsyhbhiA8EQd6dcn-RdsZQ9xtZ8"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

MEDICAL_PROMPT = """
You are MedBot, a professional medical AI assistant. 
Answer questions clearly and empathetically. 
If a user asks about a specific medical test result, explain what it means.
ALWAYS end with a disclaimer that you are an AI, not a doctor.
"""

# --- 2. STYLING (CSS) ---
def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        /* 1. Background */
        .stApp {
            background: linear-gradient(135deg, #bbdefb 0%, #90caf9 50%, #64b5f6 100%);
            background-attachment: fixed;
        }

        /* 2. Remove Extra Spacing at Top */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 95% !important;
        }

        /* 3. Hide Default Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stSidebarNav"] {display: none;}

        /* 4. Magic Card Styling (using Container Border) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-radius: 20px !important;
            border: 1px solid rgba(13, 71, 161, 0.1) !important;
            border-left: 8px solid #0277BD !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            padding: 20px !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        }
        
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-5px) !important;
            box-shadow: 0 15px 30px rgba(0,0,0,0.15) !important;
        }
        
        /* 5. Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #0277BD 0%, #01579B 100%) !important;
            color: white !important;
            border-radius: 10px;
            border: none;
            padding: 10px;
            font-weight: bold;
            width: 100%;
        }
        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 5px 15px rgba(2, 119, 189, 0.4);
        }
        
        /* 6. Text Alignment */
        h1, h2, h3 { color: #01579B !important; font-weight: 800; text-align: center;}
        p { color: #0277BD !important; font-weight: 500; text-align: center; }
        
        /* Center Images */
        div[data-testid="stImage"] { display: block; margin-left: auto; margin-right: auto; }
        div[data-testid="stImage"] > img { display: block; margin-left: auto; margin-right: auto; }
        </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION (UPDATED NAMES) ---
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #0277BD;'>Doctory AI</h2>", unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            # القائمة التي ستظهر للمستخدم
            options=["Home", "AI Chat", "Pneumonia", "Malaria", "Diabetes"],
            icons=["house-fill", "chat-dots-fill", "lungs-fill", "virus", "droplet-fill"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#0277BD", "font-size": "18px"}, 
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px", "--hover-color": "#E1F5FE"},
                "nav-link-selected": {"background-color": "#0277BD", "color": "white"},
            }
        )
        
        # ربط القائمة بأسماء الملفات الحقيقية
        if selected == "Home": 
            st.switch_page("streamlit_app.py")
        if selected == "AI Chat": 
            st.switch_page("pages/1_AI_Chatbot.py")
        if selected == "Pneumonia": 
            st.switch_page("pages/2_Pneumonia_X_Ray.py")
        if selected == "Malaria": 
            st.switch_page("pages/3_Malaria_Blood_Smear.py")
        if selected == "Diabetes": 
            st.switch_page("pages/4_Diabetes_Risk.py")

# --- MODEL LOADING ---
@st.cache_resource
def load_all_models():
    MODEL_DIR = "models/"
    if not os.path.isdir(MODEL_DIR): return None
    try:
        try: pneumonia = ort.InferenceSession(os.path.join(MODEL_DIR, "best.onnx"))
        except: pneumonia = None
        try: malaria = ort.InferenceSession(os.path.join(MODEL_DIR, "malaria_model.onnx"))
        except: malaria = None
        try: 
            diabetes = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_ensemble_model.joblib"))
            d_scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_scaler.joblib"))
        except: diabetes, d_scaler = None, None
        try:
            # Heart model kept in case you add page 5 later
            heart = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_model.joblib"))
            h_scaler = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_scaler.joblib"))
        except: heart, h_scaler = None, None
        
        return {
            "pneumonia_sess": pneumonia, "malaria_sess": malaria,
            "diabetes_model": diabetes, "diabetes_scaler": d_scaler,
            "heart_model": heart, "heart_scaler": h_scaler,
            "pneu_in": pneumonia.get_inputs()[0].name if pneumonia else None,
            "pneu_out": pneumonia.get_outputs()[0].name if pneumonia else None,
            "mal_in": malaria.get_inputs()[0].name if malaria else None,
            "mal_out": malaria.get_outputs()[0].name if malaria else None
        }
    except Exception: return None

MODELS = load_all_models()

# --- HELPERS ---
def ask_medbot(user_query, system_prompt):
    if not API_KEY: return "⚠️ API Key missing."
    try:
        payload = {"contents": [{"parts": [{"text": user_query}]}], "systemInstruction": {"parts": [{"text": system_prompt}]}}
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except: return "Connection Error"

def process_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(target_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_np.transpose(2, 0, 1), axis=0)

def prepare_diabetes_features(data, scaler):
    features = pd.DataFrame([[data['Pregnancies'], data['Glucose'], data['BP'], 29.0, 125.0, data['BMI'], 0.3725, data['Age']]], 
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    return scaler.transform(features)

def calculate_bmi(height, weight):
    return weight / ((height/100)**2) if height > 0 else 0
