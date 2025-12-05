import streamlit as st
import requests
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
import onnxruntime as ort

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
        /* Hide Default Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stSidebarNav"] {display: none;} /* Hide default sidebar list */
        
        /* Medical Blue Theme */
        .stApp {
            background-color: #FAFAFA;
        }
        
        /* Card Styling */
        .css-card {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border-left: 6px solid #0277BD;
            text-align: center;
        }
        
        /* Buttons */
        div.stButton > button {
            background-color: #0277BD;
            color: white;
            border-radius: 8px;
            border: none;
            width: 100%;
            padding: 12px;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #01579B;
            color: white;
        }

        /* Custom Title */
        h1, h2, h3 {
            color: #0277BD;
            font-family: 'Arial', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 3. AI CHAT FUNCTION ---
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

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_all_models():
    MODEL_DIR = "models/"
    if not os.path.isdir(MODEL_DIR): return None
    
    try:
        # Image Models
        pneumonia = ort.InferenceSession(os.path.join(MODEL_DIR, "best.onnx"))
        malaria = ort.InferenceSession(os.path.join(MODEL_DIR, "malaria_model.onnx"))
        
        # Tabular Models
        diabetes = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_ensemble_model.joblib"))
        d_scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_scaler.joblib"))
        heart = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_model.joblib"))
        h_scaler = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_scaler.joblib"))
        
        return {
            "pneumonia_sess": pneumonia,
            "malaria_sess": malaria,
            "diabetes_model": diabetes,
            "diabetes_scaler": d_scaler,
            "heart_model": heart,
            "heart_scaler": h_scaler,
            "pneu_in": pneumonia.get_inputs()[0].name,
            "pneu_out": pneumonia.get_outputs()[0].name,
            "mal_in": malaria.get_inputs()[0].name,
            "mal_out": malaria.get_outputs()[0].name
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# --- 5. PREPROCESSING FUNCTIONS ---
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
    features = pd.DataFrame([[
        data['Pregnancies'], data['Glucose'], data['BP'], 29.0, 125.0, 
        data['BMI'], 0.3725, data['Age']
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    return scaler.transform(features)

# (Add prepare_heart_features here using the same logic from previous code if needed)
