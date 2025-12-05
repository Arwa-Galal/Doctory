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

# Custom CSS for Medical Blue & White Theme
st.markdown("""
    <style>
    /* 1. Hide default Streamlit menu and footer */
   /*#MainMenu {visibility: hidden;}*/
   /* footer {visibility: hidden;}*/
   /* header {visibility: hidden;}*/
    
    /* 2. Hide the default sidebar navigation list */
    div[data-testid="stSidebarNav"] {display: none;}
    
    /* 3. Medical Blue Theme Setup */
    /* Main Background is White by default */
    
    /* Sidebar Background color adjustment (Optional - keeps it clean light grey) */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }

    /* Card Styling - Changed border to Medical Blue */
    .css-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #0277BD; /* Medical Blue */
    }
    
    /* Button Styling Override */
    div.stButton > button {
        background-color: #0277BD;
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #01579B; /* Darker Blue on Hover */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR & CONFIGURATION ---
with st.sidebar:
    # You can replace this URL with your own logo file if you have one
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) 
    st.markdown("<h1 style='text-align: center; color: #0277BD;'>MedBot Pro</h1>", unsafe_allow_html=True)
    
    # API Key (As requested, keeping it hardcoded for now)
    api_key = "AIzaSyBxYCfAwsyhbhiA8EQd6dcn-RdsZQ9xtZ8"
    
    st.write("") # Spacer
    
    # MODERN NAVIGATION MENU (Blue Theme)
    selected = option_menu(
        menu_title=None, # Hides the title "Navigation" for a cleaner look
        options=["AI Chatbot", "Diabetes Test", "Pneumonia Check"],
        icons=["chat-dots-fill", "activity", "lungs"], # Bootstrap Icons
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#0277BD", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin":"5px", 
                "--hover-color": "#E1F5FE", # Light blue hover
                "color": "#333333"
            },
            "nav-link-selected": {"background-color": "#0277BD", "color": "white"}, # Active Blue
        }
    )
    
    st.markdown("---")
    st.info("⚠️ **Disclaimer:** AI assistance is for educational purposes only. Always consult a doctor.")

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
def load_all_models():
    """Loads all models, scalers, and ONNX sessions from the models/ directory."""
    MODEL_DIR = "models/"
    if not os.path.isdir(MODEL_DIR):
        return None
    
    try:
        # --- Image Models (ONNX) ---
        pneumonia_session = ort.InferenceSession(os.path.join(MODEL_DIR, "best.onnx"))
        malaria_session = ort.InferenceSession(os.path.join(MODEL_DIR, "malaria_model.onnx"))

        # --- Tabular Models (Joblib) ---
        diabetes_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_ensemble_model.joblib"))
        diabetes_scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_model_package/diabetes_scaler.joblib"))
        heart_model = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_model.joblib"))
        heart_scaler = joblib.load(os.path.join(MODEL_DIR, "HeartRisk_model_package/HeartRisk_scaler.joblib"))
        
        return {
            "pneumonia_session": pneumonia_session,
            "malaria_session": malaria_session,
            "diabetes_model": diabetes_model,
            "diabetes_scaler": diabetes_scaler,
            "heart_model": heart_model,
            "heart_scaler": heart_scaler,
            "pneumonia_input_name": pneumonia_session.get_inputs()[0].name,
            "pneumonia_output_name": pneumonia_session.get_outputs()[0].name,
            "malaria_input_name": malaria_session.get_inputs()[0].name,
            "malaria_output_name": malaria_session.get_outputs()[0].name,
            "pneumonia_classes": ["Normal", "Pneumonia_bacteria", "Pneumonia_virus"]
        }
    except Exception as e:
        st.error(f"Error loading local models. Details: {e}")
        return None

MODELS = load_all_models()

# --------------------------------------------------------------------
# 2. HELPER FUNCTIONS 
# --------------------------------------------------------------------

def process_image_yolo(image_bytes, target_size=(224, 224)):
    """Preprocesses image for the Pneumonia (YOLO-based ONNX) model."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def process_image_keras(image_bytes, target_size=(224, 224)):
    """Preprocesses image for the Malaria (Keras-based ONNX) model."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def calculate_bmi(height_cm, weight_kg):
    if height_cm == 0 or height_cm < 50:
        return 0
    return weight_kg / ((height_cm / 100) ** 2)

def get_age_category(age):
    age = int(age)
    if 18 <= age <= 24: return 'Young'
    if 25 <= age <= 39: return 'Adult'
    if 40 <= age <= 54: return 'Mid-Aged'
    if 55 <= age <= 64: return 'Senior-Adult'
    if age >= 65: return 'Elderly'
    return 'Adult'

# Feature preparation functions remain here, accessible to all pages:

def prepare_diabetes_features(data):
    # Uses the global scaler loaded in MODELS
    scaler = MODELS['diabetes_scaler']
    age = data.get('Age')
    weight = data.get('Weight')
    height = data.get('Height')
    bp = data.get('BP')
    glucose = data.get('Glucose')
    pregnancies = data.get('Pregnancies', 0)

    bmi = calculate_bmi(height, weight)
    skin_thickness_default = 29.0
    insulin_default = 125.0 
    dpf_default = 0.3725

    feature_order = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    features = pd.DataFrame([[
        pregnancies, glucose, bp, skin_thickness_default,
        insulin_default, bmi, dpf_default, age
    ]], columns=feature_order)

    return scaler.transform(features)

def prepare_heart_features(data):
    # Uses the global scaler loaded in MODELS
    scaler = MODELS['heart_scaler']
    height = data.get('Height')
    weight = data.get('Weight')
    age = data.get('Age')
    bmi = calculate_bmi(height, weight)
    
    # Mappings (unchanged)
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
    except ValueError:
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

    # Build feature dictionary in the correct final order
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
        'alcohol_consumption': map_alcohol(data.get('Alcohol_Consumption')),
        'fruit_consumption': map_consumption(data.get('Fruit_Consumption')),
        'vegetables_consumption': map_consumption(data.get('Vegetables_Consumption')),
        'potato_consumption': map_fried(data.get('FriedPotato_Consumption')),
        'bmi_group': bmi_group_map.get(bmi_group_str),
        'sex_Female': 1 if data.get('Sex') == 'Female' else 0,
        'sex_Male': 1 if data.get('Sex') == 'Male' else 0,
        'smoking_history_No': 1 if map_smoking(data.get('Smoking_History')) == 0 else 0,
        'smoking_history_Yes': 1 if map_smoking(data.get('Smoking_History')) == 1 else 0,
    }

    final_feature_order = [
        'general_health', 'checkup', 'exercise', 'skin_cancer', 'other_cancer',
        'depression', 'diabetes', 'arthritis', 'age_category', 'height', 'weight',
        'bmi', 'alcohol_consumption', 'fruit_consumption', 'vegetables_consumption',
        'potato_consumption', 'bmi_group', 'sex_Female', 'sex_Male',
        'smoking_history_No', 'smoking_history_Yes'
    ]

    features = pd.DataFrame([feature_dict], columns=final_feature_order)
    return scaler.transform(features)
