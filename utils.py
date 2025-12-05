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

# --- 1. CONFIGURATION ---
API_KEY = "AIzaSyCHBjZJiBE4rtDS4daTaq32yY4gBWmR7rA"
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
       /* footer {visibility: hidden;}*/
       /* header {visibility: hidden;}*/
        div[data-testid="stSidebarNav"] {display: none;}
        
        /* Medical Blue Theme */
        .stApp { background-color: #FAFAFA; }
        
        /* Card Styling */
        .css-card {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border-left: 6px solid #0277BD;
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
        
        /* Titles */
        h1, h2, h3 { color: #0277BD; font-family: 'Arial', sans-serif; }
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
    """Loads all models, scalers, and ONNX sessions."""
    MODEL_DIR = "models/"
    if not os.path.isdir(MODEL_DIR): return None
    
    try:
        # Image Models (ONNX)
        try: pneumonia = ort.InferenceSession(os.path.join(MODEL_DIR, "best.onnx"))
        except: pneumonia = None
        
        try: malaria = ort.InferenceSession(os.path.join(MODEL_DIR, "malaria_model.onnx"))
        except: malaria = None

        # Tabular Models (Joblib)
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
            # Helper keys for ONNX input/output names if models exist
            "pneu_in": pneumonia.get_inputs()[0].name if pneumonia else None,
            "pneu_out": pneumonia.get_outputs()[0].name if pneumonia else None,
            "mal_in": malaria.get_inputs()[0].name if malaria else None,
            "mal_out": malaria.get_outputs()[0].name if malaria else None
        }
    except Exception as e:
        st.error(f"Critical Error loading models: {e}")
        return None

MODELS = load_all_models()

# --- 5. PREPROCESSING FUNCTIONS ---

def process_image(image_bytes, target_size=(224, 224)):
    """Generic image preprocessing for ONNX models (Resize -> Norm -> Transpose -> Batch)"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1) # CHW format
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def calculate_bmi(height, weight):
    return weight / ((height/100)**2) if height > 0 else 0

def get_age_category(age):
    age = int(age)
    if 18 <= age <= 24: return 'Young'
    if 25 <= age <= 39: return 'Adult'
    if 40 <= age <= 54: return 'Mid-Aged'
    if 55 <= age <= 64: return 'Senior-Adult'
    if age >= 65: return 'Elderly'
    return 'Adult'

# --- Feature Preparation: Diabetes ---
def prepare_diabetes_features(data, scaler):
    features = pd.DataFrame([[
        data['Pregnancies'], data['Glucose'], data['BP'], 29.0, 125.0, 
        data['BMI'], 0.3725, data['Age']
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    return scaler.transform(features)

# --- Feature Preparation: Heart ---
def prepare_heart_features(data):
    scaler = MODELS['heart_scaler']
    
    height = data.get('Height')
    weight = data.get('Weight')
    age = data.get('Age')
    bmi = calculate_bmi(height, weight)
    
    # Mappings
    general_health_map = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Poor': 3, 'Very Good': 4}
    checkup_map = {'More than 5 years': 0, 'Never': 1, 'Past 1 year': 2, 'Past 2 years': 3, 'Past 5 years': 4}
    binary_map = {'No': 0, 'Yes': 1} 
    diabetes_map = {'No': 0, 'No Pre Diabetes': 1, 'Only during pregnancy': 2, 'Yes': 3}
    age_category_map = {'Adult': 0, 'Elderly': 1, 'Mid-Aged': 2, 'Senior-Adult': 3, 'Young': 4}
    bmi_group_map = {'Normal weight': 0, 'Obese I': 1, 'Obese II': 2, 'Overweight': 3, 'Underweight': 4}

    # BMI Group Calculation
    bmi_bins = [12.02, 18.3, 26.85, 31.58, 37.8, 100]
    bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese I', 'Obese II']
    try:
        bmi_group_str = pd.cut([bmi], bins=bmi_bins, labels=bmi_labels, right=False)[0]
    except:
        bmi_group_str = 'Normal weight'

    # Lifestyle Mappers
    def map_smoking(val): return 1 if val in ['Former', 'Current'] else 0 
    def map_alcohol(val):
        if val == 'Never': return 0
        if val == 'Occasionally': return 4
        if val == 'Weekly': return 8
        if val == 'Daily': return 30
        return 0
    def map_consumption(val):
        if val == '0': return 0
        if val == '1–2': return 12 
        if val == '3–5': return 20 
        if val == '6–7': return 30 
        return 0
    def map_fried(val):
        if val == 'Rarely': return 2
        if val == 'Weekly': return 4
        if val == 'Several times per week': return 8
        return 0
        
    age_cat_str = get_age_category(age)

    feature_dict = {
        'general_health': general_health_map.get(data.get('General_Health')),
        'checkup': checkup_map.get(data.get('Checkup')),
        'exercise': binary_map.get(data.get('Exercise')),
        'skin_cancer': binary_map.get(data.get('Skin_Cancer')),
        'other_cancer': binary_map.get(data.get('Other_Cancer')),
        'depression': binary_map.get(data.get('Depression')),
        'diabetes': diabetes_map.get(data.get('Diabetes')),
        'arthritis': binary_map.get(data.get('Arthritis')),
        'age_category': age_category_map.get(age_cat_str),
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'bmi_group': bmi_group_map.get(bmi_group_str, 0), 
        'alcohol_consumption': map_alcohol(data.get('Alcohol_Consumption')),
        'fruit_consumption': map_consumption(data.get('Fruit_Consumption')),
        'vegetables_consumption': map_consumption(data.get('Vegetables_Consumption')),
        'potato_consumption': map_fried(data.get('FriedPotato_Consumption')),
        'sex_Female': 1 if data.get('Sex') == 'Female' else 0,
        'sex_Male': 1 if data.get('Sex') == 'Male' else 0,
        'smoking_history_No': 1 if map_smoking(data.get('Smoking_History')) == 0 else 0,
        'smoking_history_Yes': 1 if map_smoking(data.get('Smoking_History')) == 1 else 0,
    }

    final_feature_order = [
        'general_health', 'checkup', 'exercise', 'skin_cancer', 'other_cancer',
        'depression', 'diabetes', 'arthritis', 'age_category', 'height', 'weight',
        'bmi', 'bmi_group', 'alcohol_consumption', 'fruit_consumption', 'vegetables_consumption',
        'potato_consumption', 'sex_Female', 'sex_Male',
        'smoking_history_No', 'smoking_history_Yes'
    ]
    
    features = pd.DataFrame([feature_dict], columns=final_feature_order)
    return scaler.transform(features)
