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
API_KEY = "AIzaSyBxYCfAwsyhbhiA8EQd6dcn-RdsZQ9xtZ8"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

MEDICAL_PROMPT = """
You are MedBot, a professional medical AI assistant. 
Answer questions clearly and empathetically. 
If a user asks about a specific medical test result, explain what it means.
ALWAYS end with a disclaimer that you are an AI, not a doctor.
"""

# --- CSS STYLING ---
def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        .stApp {
            background: linear-gradient(135deg, #bbdefb 0%, #90caf9 50%, #64b5f6 100%);
            background-attachment: fixed;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 95% !important;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stSidebarNav"] {display: none;}

/* تنسيق الكارت الجديد (أبيض ويطير) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #f2f2ff !important; /* لون أبيض صريح */
            border-radius: 20px !important;
            border: 1px solid #e0e0e0 !important;
            border-left: 8px solid #0277BD !important;
            box-shadow: 0 5px 15px rgba(0,0,4,0.08) !important;
            padding: 25px !important;
            /* كود الحركة الناعمة (Bounce effect) */
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        }

        /* حركة الطفو عند مرور الماوس */
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-12px) scale(1.02) !important; /* يرفع الكارت ويكبره قليلاً */
            box-shadow: 0 30px 60px rgba(0,0,0,0.15) !important; /* ظل قوي للعمق */
            border-color: #0277BD !important;
        }
        
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
        
        h1, h2, h3 { color: #01579B !important; font-weight: 800; text-align: center;}
        p { color: #0277BD !important; font-weight: 500; text-align: center; }
        
        div[data-testid="stImage"] { display: block; margin-left: auto; margin-right: auto; }
        div[data-testid="stImage"] > img { display: block; margin-left: auto; margin-right: auto; }
        </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION (FIXED) ---
def render_sidebar(current_page):
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #0277BD;'>Doctory AI</h2>", unsafe_allow_html=True)
        
        options = ["Home", "AI Chat", "Pneumonia", "Malaria", "Diabetes", "Heart Risk"]
        
        # تحديد الأيقونة النشطة بناءً على الصفحة الحالية لتجنب الأخطاء
        try:
            index = options.index(current_page)
        except ValueError:
            index = 0

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
        
        # المنطق الجديد: الانتقال فقط إذا كانت الصفحة المختارة مختلفة عن الحالية
        if selected != current_page:
            if selected == "Home": st.switch_page("streamlit_app.py")
            if selected == "AI Chat": st.switch_page("pages/1_AI_Chatbot.py")
            if selected == "Pneumonia": st.switch_page("pages/2_Pneumonia_X_Ray.py")
            if selected == "Malaria": st.switch_page("pages/3_Malaria_Blood_Smear.py")
            if selected == "Diabetes": st.switch_page("pages/4_Diabetes_Risk.py")
            if selected == "Heart Risk": st.switch_page("pages/5_❤️_Heart_Risk.py")

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

def prepare_heart_features(data):
    scaler = MODELS['heart_scaler']
    bmi = calculate_bmi(data['Height'], data['Weight'])
    
    # Simple Mappers
    map_gen = {'Excellent':0,'Fair':1,'Good':2,'Poor':3,'Very Good':4}
    map_check = {'More than 5 years':0,'Never':1,'Past 1 year':2,'Past 2 years':3,'Past 5 years':4}
    map_diab = {'No':0,'No Pre Diabetes':1,'Only during pregnancy':2,'Yes':3}
    map_age = {'Adult':0,'Elderly':1,'Mid-Aged':2,'Senior-Adult':3,'Young':4}
    map_bmi = {'Normal weight':0,'Obese I':1,'Obese II':2,'Overweight':3,'Underweight':4}
    
    age = data['Age']
    if 18<=age<=24: ac='Young'
    elif 25<=age<=39: ac='Adult'
    elif 40<=age<=54: ac='Mid-Aged'
    elif 55<=age<=64: ac='Senior-Adult'
    else: ac='Elderly'

    try: bmi_grp = pd.cut([bmi], bins=[0, 18.5, 25, 30, 35, 100], labels=['Underweight','Normal weight','Overweight','Obese I','Obese II'])[0]
    except: bmi_grp = 'Normal weight'

    f_dict = {
        'general_health': map_gen.get(data['General_Health']),
        'checkup': map_check.get(data['Checkup']),
        'exercise': 1 if data['Exercise']=='Yes' else 0,
        'skin_cancer': 1 if data['Skin_Cancer']=='Yes' else 0,
        'other_cancer': 1 if data['Other_Cancer']=='Yes' else 0,
        'depression': 1 if data['Depression']=='Yes' else 0,
        'diabetes': map_diab.get(data['Diabetes']),
        'arthritis': 1 if data['Arthritis']=='Yes' else 0,
        'age_category': map_age.get(ac),
        'height': data['Height'], 'weight': data['Weight'], 'bmi': bmi,
        'bmi_group': map_bmi.get(bmi_grp, 0),
        'alcohol_consumption': 0, 'fruit_consumption': 0, 'vegetables_consumption': 0, 'potato_consumption': 0,
        'sex_Female': 1 if data['Sex']=='Female' else 0,
        'sex_Male': 1 if data['Sex']=='Male' else 0,
        'smoking_history_No': 1 if data['Smoking_History']=='Never' else 0,
        'smoking_history_Yes': 1 if data['Smoking_History']!='Never' else 0
    }
    cols = ['general_health', 'checkup', 'exercise', 'skin_cancer', 'other_cancer', 'depression', 'diabetes', 'arthritis', 'age_category', 'height', 'weight', 'bmi', 'alcohol_consumption', 'fruit_consumption', 'vegetables_consumption', 'potato_consumption', 'bmi_group', 'sex_Female', 'sex_Male', 'smoking_history_No', 'smoking_history_Yes']
    return scaler.transform(pd.DataFrame([f_dict], columns=cols))
