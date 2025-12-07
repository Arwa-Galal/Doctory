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
# تأكدي من وجود هذه المكتبة في requirements.txt
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
        
        .stApp {
            background: linear-gradient(135deg, #bbdefb 0%, #90caf9 50%, #64b5f6 100%);
            background-attachment: fixed;
        }
        
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stSidebarNav"] {display: none;}

        /* تنسيق الكارت */
        .css-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 15px; /* مسافة صغيرة بين الكارت والزر */
            border-left: 8px solid #0277BD;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center; /* توسيط المحتوى */
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .css-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }

        /* تنسيق الصور داخل الكارت */
        .css-card img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 60px;
            margin-bottom: 15px;
        }

        /* تنسيق النصوص داخل الكارت */
        .css-card h3 { color: #01579B; margin: 0; font-size: 1.3rem; font-weight: 800; }
        .css-card p { color: #555; margin: 5px 0 0 0; font-size: 0.9rem; }

        /* الأزرار */
        div.stButton > button {
            background: linear-gradient(135deg, #0277BD 0%, #01579B 100%) !important;
            color: white !important;
            border-radius: 12px;
            border: none;
            padding: 10px;
            font-weight: bold;
            width: 100%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(2, 119, 189, 0.3);
        }
        
        h1, h2 { color: #01579B !important; font-weight: 800; text-align: center;}
        </style>
    """, unsafe_allow_html=True)
# --- 3. CUSTOM SIDEBAR NAVIGATION (THE BLUE MENU) ---
def render_sidebar():
    with st.sidebar:
        # Logo or Title
        st.markdown("<h2 style='text-align: center; color: #0277BD;'>Doctory AI</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # القائمة الزرقاء الاحترافية
        # This menu handles navigation without relying on file emojis
        selected = option_menu(
            menu_title=None,
            options=["Home", "AI Chat", "Diabetes", "Pneumonia", "Malaria", "Heart Risk"],
            # هذه الأيقونات من مكتبة Bootstrap وهي زرقاء ونظيفة
            icons=["house-fill", "chat-dots-fill", "droplet-fill", "lungs-fill", "virus", "heart-pulse-fill"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#0277BD", "font-size": "18px"}, # اللون الأزرق للأيقونات
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin":"5px", 
                    "--hover-color": "#E3F2FD",
                    "color": "#333333"
                },
                "nav-link-selected": {"background-color": "#0277BD", "color": "white"},
            }
        )
        
        # Navigation Logic (Switch Pages based on selection)
        # Note: We assume current page context to prevent reloading the same page loop
        # But st.switch_page requires the script to be run from main. 
        
        if selected == "Home":
            st.switch_page("app.py")
        elif selected == "AI Chat":
            st.switch_page("pages/1_💬_Chat_With_Doctory.py")
        elif selected == "Diabetes":
            st.switch_page("pages/2_🩸_Diabetes_Test.py")
        elif selected == "Pneumonia":
            st.switch_page("pages/3_🫁_Pneumonia_Check.py")
        elif selected == "Malaria":
            st.switch_page("pages/4_🦟_Malaria_Check.py")
        elif selected == "Heart Risk":
            st.switch_page("pages/5_❤️_Heart_Risk.py")
            
        st.markdown("---")
        st.caption("© 2024 Doctory AI Project")

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_all_models():
    MODEL_DIR = "models/"
    if not os.path.isdir(MODEL_DIR): return None
    
    try:
        # Image Models
        try: pneumonia = ort.InferenceSession(os.path.join(MODEL_DIR, "best.onnx"))
        except: pneumonia = None
        
        try: malaria = ort.InferenceSession(os.path.join(MODEL_DIR, "malaria_model.onnx"))
        except: malaria = None

        # Tabular Models
        try: 
            diabetes = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_ensemble_model.joblib"))
            d_scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_scaler.joblib"))
        except: diabetes, d_scaler = None, None

        try:
            heart = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_model.joblib"))
            h_scaler = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_scaler.joblib"))
        except: heart, h_scaler = None, None
        
        return {
            "pneumonia_sess": pneumonia,
            "malaria_sess": malaria,
            "diabetes_model": diabetes,
            "diabetes_scaler": d_scaler,
            "heart_model": heart,
            "heart_scaler": h_scaler,
            "pneu_in": pneumonia.get_inputs()[0].name if pneumonia else None,
            "pneu_out": pneumonia.get_outputs()[0].name if pneumonia else None,
            "mal_in": malaria.get_inputs()[0].name if malaria else None,
            "mal_out": malaria.get_outputs()[0].name if malaria else None
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

MODELS = load_all_models()

# --- 5. AI & HELPERS ---
def ask_medbot(user_query, system_prompt):
    if not API_KEY: return "⚠️ API Key missing."
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    try:
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        if response.status_code != 200: return f"Error: {response.text}"
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e: return f"Error: {e}"

def process_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def calculate_bmi(height, weight):
    return weight / ((height/100)**2) if height > 0 else 0

def prepare_diabetes_features(data, scaler):
    features = pd.DataFrame([[data['Pregnancies'], data['Glucose'], data['BP'], 29.0, 125.0, data['BMI'], 0.3725, data['Age']]], 
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    return scaler.transform(features)

def get_age_category(age):
    age = int(age)
    if 18 <= age <= 24: return 'Young'
    if 25 <= age <= 39: return 'Adult'
    if 40 <= age <= 54: return 'Mid-Aged'
    if 55 <= age <= 64: return 'Senior-Adult'
    if age >= 65: return 'Elderly'
    return 'Adult'

def prepare_heart_features(data):
    scaler = MODELS['heart_scaler']
    bmi = calculate_bmi(data['Height'], data['Weight'])
    age_cat = get_age_category(data['Age'])
    
    # Simple Mappers
    map_gen = {'Excellent':0,'Fair':1,'Good':2,'Poor':3,'Very Good':4}
    map_check = {'More than 5 years':0,'Never':1,'Past 1 year':2,'Past 2 years':3,'Past 5 years':4}
    map_diab = {'No':0,'No Pre Diabetes':1,'Only during pregnancy':2,'Yes':3}
    map_age = {'Adult':0,'Elderly':1,'Mid-Aged':2,'Senior-Adult':3,'Young':4}
    map_bmi = {'Normal weight':0,'Obese I':1,'Obese II':2,'Overweight':3,'Underweight':4}
    
    # BMI Group
    try: bmi_str = pd.cut([bmi], bins=[0, 18.5, 25, 30, 35, 100], labels=['Underweight','Normal weight','Overweight','Obese I','Obese II'])[0]
    except: bmi_str = 'Normal weight'

    # Construct
    f_dict = {
        'general_health': map_gen.get(data['General_Health']),
        'checkup': map_check.get(data['Checkup']),
        'exercise': 1 if data['Exercise'] == 'Yes' else 0,
        'skin_cancer': 1 if data['Skin_Cancer'] == 'Yes' else 0,
        'other_cancer': 1 if data['Other_Cancer'] == 'Yes' else 0,
        'depression': 1 if data['Depression'] == 'Yes' else 0,
        'diabetes': map_diab.get(data['Diabetes']),
        'arthritis': 1 if data['Arthritis'] == 'Yes' else 0,
        'age_category': map_age.get(age_cat),
        'height': data['Height'], 'weight': data['Weight'], 'bmi': bmi,
        'bmi_group': map_bmi.get(bmi_str, 0),
        'alcohol_consumption': 0, 'fruit_consumption': 0, 'vegetables_consumption': 0, 'potato_consumption': 0, # Simplified
        'sex_Female': 1 if data['Sex']=='Female' else 0,
        'sex_Male': 1 if data['Sex']=='Male' else 0,
        'smoking_history_No': 1 if data['Smoking_History']=='Never' else 0,
        'smoking_history_Yes': 1 if data['Smoking_History']!='Never' else 0
    }
    
    # Fill missing columns with 0 to match scaler expectation (Quick fix for demo)
    # Important: In production, map all fields correctly as per previous code
    final_cols = ['general_health', 'checkup', 'exercise', 'skin_cancer', 'other_cancer', 'depression', 'diabetes', 'arthritis', 'age_category', 'height', 'weight', 'bmi', 'alcohol_consumption', 'fruit_consumption', 'vegetables_consumption', 'potato_consumption', 'bmi_group', 'sex_Female', 'sex_Male', 'smoking_history_No', 'smoking_history_Yes']
    
    return scaler.transform(pd.DataFrame([f_dict], columns=final_cols))
