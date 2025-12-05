import streamlit as st
from streamlit_option_menu import option_menu
import requests
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import os
import io
import onnxruntime as ort

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="MedBot Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Medical Blue Theme & Hiding Default Navigation
st.markdown("""
    <style>
    /* 1. Hide default Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 2. Hide the default sidebar navigation list (CRITICAL) */
    div[data-testid="stSidebarNav"] {display: none;}
    
    /* 3. Medical Blue Theme Setup */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }

    /* Card Styling */
    .css-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #0277BD; /* Medical Blue */
    }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #0277BD;
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
        padding: 10px;
    }
    div.stButton > button:hover {
        background-color: #01579B;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR & CUSTOM NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) 
    st.markdown("<h2 style='text-align: center; color: #0277BD;'>MedBot Pro</h2>", unsafe_allow_html=True)
    
    # API Key
    api_key = "AIzaSyBxYCfAwsyhbhiA8EQd6dcn-RdsZQ9xtZ8"
    
    st.write("") 
    
    # CUSTOM MODERN NAVIGATION MENU
    selected = option_menu(
        menu_title=None,
        options=["AI Chatbot", "Diabetes Test", "Heart Disease Risk", "Pneumonia Check", "Malaria Check"],
        icons=["chat-dots-fill", "droplet", "heart-pulse", "lungs", "bug"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#0277BD", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin":"5px", 
                "--hover-color": "#E1F5FE",
                "color": "#333333"
            },
            "nav-link-selected": {"background-color": "#0277BD", "color": "white"},
        }
    )
    
    st.markdown("---")
    st.info("⚠️ **Disclaimer:** AI assistance is for educational purposes only. Always consult a doctor.")

# --- 3. HELPER FUNCTIONS & MODEL LOADING (YOUR PROVIDED CODE) ---

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
# DATA PROCESSING HELPER FUNCTIONS 
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

# --- 4. CHATBOT FUNCTION ---

# Gemini API URL
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

def ask_medbot(user_query, system_prompt):
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

medical_prompt = """
You are MedBot, a professional medical AI assistant. 
Answer questions clearly and empathetically. 
If a user asks about a specific medical test result, explain what it means.
ALWAYS end with a disclaimer that you are an AI, not a doctor.
"""

# --- 5. MAIN APP LOGIC (PAGES) ---

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

elif selected == "Diabetes Test":
    st.title("🩸 Diabetes Risk Assessment")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 30)
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose", 0, 500, 100)
    with col2:
        weight = st.number_input("Weight (kg)", 10, 300, 70)
        height = st.number_input("Height (cm)", 50, 250, 170)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
    with col3:
        st.info("BMI is calculated automatically.")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Analyze Result"):
        if MODELS and 'diabetes_model' in MODELS:
            try:
                input_data = {'Age': age, 'Weight': weight, 'Height': height, 'BP': bp, 'Glucose': glucose, 'Pregnancies': pregnancies}
                final_features = prepare_diabetes_features(input_data)
                prediction = MODELS['diabetes_model'].predict(final_features)[0]
                
                result_str = "Diabetic (High Risk)" if prediction == 1 else "Healthy (Low Risk)"
                color = "#D32F2F" if prediction == 1 else "#388E3C"
                
                st.markdown(f"### Result: <span style='color:{color}'>{result_str}</span>", unsafe_allow_html=True)
                ai_exp = ask_medbot(f"Patient Result: {result_str}. Glucose: {glucose}, Age: {age}. Explain.", medical_prompt)
                st.info(f"👨‍⚕️ **Dr. AI Analysis:**\n\n{ai_exp}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Models not loaded.")

elif selected == "Heart Disease Risk":
    st.title("❤️ Heart Disease Risk Assessment")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age_h = st.number_input("Age", 18, 100, 50, key="h_age")
        sex = st.selectbox("Sex", ["Male", "Female"])
        weight_h = st.number_input("Weight (kg)", 30, 200, 80, key="h_w")
        height_h = st.number_input("Height (cm)", 100, 250, 175, key="h_h")
        gen_health = st.selectbox("General Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
        checkup = st.selectbox("Last Checkup", ["Past 1 year", "Past 2 years", "Past 5 years", "More than 5 years", "Never"])
        
    with col2:
        smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Weekly", "Daily"])
        diabetes_status = st.selectbox("Diabetes History", ["No", "No Pre Diabetes", "Yes", "Only during pregnancy"])
        exercise = st.selectbox("Exercise", ["Yes", "No"])
        # Simplified for UI brevity, ideally add all fields
        
    st.info("Note: Using simplified inputs for demo. Ensure all fields map to `prepare_heart_features`.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Analyze Heart Risk"):
        if MODELS and 'heart_model' in MODELS:
            try:
                # Construct data dictionary (Needs all fields from prepare_heart_features)
                # Filling missing ones with defaults for stability
                input_data = {
                    'Age': age_h, 'Sex': sex, 'Weight': weight_h, 'Height': height_h,
                    'General_Health': gen_health, 'Checkup': checkup, 'Smoking_History': smoking,
                    'Alcohol_Consumption': alcohol, 'Diabetes': diabetes_status, 'Exercise': exercise,
                    'Fruit_Consumption': '1–2', 'Vegetables_Consumption': '1–2', # Defaults
                    'FriedPotato_Consumption': 'Rarely', 'Depression': 'No', 
                    'Skin_Cancer': 'No', 'Other_Cancer': 'No', 'Arthritis': 'No'
                }
                
                final_features = prepare_heart_features(input_data)
                prediction = MODELS['heart_model'].predict(final_features)[0]
                
                result_str = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
                color = "#D32F2F" if prediction == 1 else "#388E3C"
                
                st.markdown(f"### Result: <span style='color:{color}'>{result_str}</span>", unsafe_allow_html=True)
                ai_exp = ask_medbot(f"Patient Result: {result_str}. Age: {age_h}, Smoking: {smoking}. Explain.", medical_prompt)
                st.info(f"👨‍⚕️ **Dr. AI Analysis:**\n\n{ai_exp}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Models not loaded.")

elif selected == "Pneumonia Check":
    st.title("🫁 Pneumonia X-Ray Check")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"], key="pneu")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and MODELS:
        st.image(uploaded_file, width=300)
        if st.button("Analyze Image"):
            try:
                img_bytes = uploaded_file.read()
                img_input = process_image_yolo(img_bytes)
                
                input_name = MODELS['pneumonia_input_name']
                output_name = MODELS['pneumonia_output_name']
                result = MODELS['pneumonia_session'].run([output_name], {input_name: img_input})
                
                idx = np.argmax(result[0])
                label = MODELS['pneumonia_classes'][idx] if idx < 3 else "Unknown"
                
                st.markdown(f"### Result: **{label}**")
                ai_exp = ask_medbot(f"X-Ray shows: {label}. Explain.", medical_prompt)
                st.info(f"👨‍⚕️ **Dr. AI Analysis:**\n\n{ai_exp}")
            except Exception as e:
                st.error(f"Error: {e}")

elif selected == "Malaria Check":
    st.title("🦟 Malaria Cell Check")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Cell Image", type=["jpg", "png", "jpeg"], key="mal")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and MODELS:
        st.image(uploaded_file, width=300)
        if st.button("Analyze Cell"):
            try:
                img_bytes = uploaded_file.read()
                img_input = process_image_keras(img_bytes)
                
                input_name = MODELS['malaria_input_name']
                output_name = MODELS['malaria_output_name']
                result = MODELS['malaria_session'].run([output_name], {input_name: img_input})
                
                # Assuming simple binary classification for Malaria
                prediction = result[0][0][0] # Adjust index based on specific model output shape
                label = "Parasitized (Infected)" if prediction < 0.5 else "Uninfected" # Adjust threshold based on model training
                color = "#D32F2F" if label.startswith("Parasitized") else "#388E3C"

                st.markdown(f"### Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                ai_exp = ask_medbot(f"Cell analysis shows: {label}. Explain.", medical_prompt)
                st.info(f"👨‍⚕️ **Dr. AI Analysis:**\n\n{ai_exp}")
            except Exception as e:
                st.error(f"Error: {e}")
