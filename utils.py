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

# --- CONFIGURATION ---
API_KEY = "AIzaSyAg-7Wu_mCF-z9P-KEbkjpQEb7B3PB_hxo"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

MEDICAL_PROMPT = "You are MedBot, a professional medical AI assistant. Answer clearly. Always disclaim you are an AI."

# --- CSS STYLING (WHITE CARDS) ---
def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        /* 1. Background */
        .stApp {
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%) !important;
            background-attachment: fixed;
        }
        
        .block-container { padding-top: 1.5rem !important; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stSidebarNav"] {display: none;}

        /* --- 2. THE CARD STYLE (Solid White) --- */
        /* هذا هو الكود المسؤول عن شكل الكارت */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important; /* خلفية بيضاء */
            border: 1px solid #E0E0E0 !important;
            border-radius: 16px !important;
            padding: 20px !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
            border-top: 5px solid #0277BD !important; /* خط أزرق علوي */
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        }

        /* تأثير الطفو (Hover) */
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-8px) !important;
            box-shadow: 0 15px 30px rgba(0,0,0,0.15) !important;
            border-top-color: #01579B !important;
        }

        /* --- 3. TEXT COLORS (Fix visibility on white) --- */
        h1, h2, h3, h4 { color: #01579B !important; }
        p, label, li, span { color: #424242 !important; }

        /* --- 4. BUTTON STYLE --- */
        div.stButton > button {
            background: #0277BD !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            width: 100%;
            padding: 10px;
            font-weight: 600;
            margin-top: 10px;
        }
        div.stButton > button:hover {
            background: #01579B !important;
            box-shadow: 0 4px 12px rgba(2, 119, 189, 0.3) !important;
        }
        
        /* Center Images */
        div[data-testid="stImage"] { display: flex; justify-content: center; }
        div[data-testid="stImage"] > img { width: 70px !important; object-fit: contain; }
        </style>
    """, unsafe_allow_html=True)

# --- NAVIGATION ---
def render_sidebar(current_page):
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #0277BD;'>Doctory AI</h2>", unsafe_allow_html=True)
        options = ["Home", "AI Chat", "Pneumonia", "Malaria", "Diabetes", "Heart Risk"]
        try: index = options.index(current_page)
        except: index = 0
        
        selected = option_menu(
            menu_title=None,
            options=options,
            icons=["house-fill", "chat-dots-fill", "lungs-fill", "virus", "droplet-fill", "heart-pulse-fill"],
            default_index=index,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#0277BD", "font-size": "18px"}, 
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px", "--hover-color": "#E1F5FE"},
                "nav-link-selected": {"background-color": "#0277BD", "color": "white"},
            }
        )
        
        if selected != current_page:
            if selected == "Home": st.switch_page("streamlit_app.py")
            if selected == "AI Chat": st.switch_page("pages/1_AI_Chatbot.py")
            if selected == "Pneumonia": st.switch_page("pages/2_Pneumonia_X_Ray.py")
            if selected == "Malaria": st.switch_page("pages/3_Malaria_Blood_Smear.py")
            if selected == "Diabetes": st.switch_page("pages/4_Diabetes_Risk.py")
            if selected == "Heart Risk": st.switch_page("pages/5_Heart_Disease_Risk.py") # تأكدي من اسم ملف القلب

# --- OPTIMIZED MODEL LOADING (To Fix Memory Error) ---
# قمنا بتقسيم تحميل الموديلات لتوفير الذاكرة
@st.cache_resource
def get_model(model_type):
    MODEL_DIR = "models/"
    try:
        if model_type == "pneumonia":
            return ort.InferenceSession(os.path.join(MODEL_DIR, "best.onnx"))
        elif model_type == "malaria":
            return ort.InferenceSession(os.path.join(MODEL_DIR, "malaria_model.onnx"))
        elif model_type == "diabetes":
            m = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_ensemble_model.joblib"))
            s = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_scaler.joblib"))
            return m, s
        elif model_type == "heart":
            m = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_model.joblib"))
            s = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_scaler.joblib"))
            return m, s
    except Exception as e:
        return None

# --- HELPERS ---
def ask_medbot(user_query, system_prompt):
    if not API_KEY: return "⚠️ API Key missing."
    try:
        payload = {"contents": [{"parts": [{"text": user_query}]}], "systemInstruction": {"parts": [{"text": system_prompt}]}}
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        if response.status_code != 200: return f"Error: {response.text}"
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e: return f"Connection Error: {e}"

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

def prepare_heart_features(data, scaler):
    bmi = calculate_bmi(data['Height'], data['Weight'])
    # (Mappings simplified for brevity, ensure they match your training logic)
    map_gen = {'Excellent':0,'Fair':1,'Good':2,'Poor':3,'Very Good':4}
    # ... (باقي المابينج كما في الكود السابق) ...
    # لعدم الإطالة، استخدمي نفس منطق تحضير الداتا من الكود السابق هنا
    # ولكن المهم هو استقبال الـ scaler كمدخل للدالة
    return None # استبدلي هذا بالجزء الخاص بتجهيز بيانات القلب من الكود السابق
