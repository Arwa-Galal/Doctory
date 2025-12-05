import streamlit as st
from streamlit_option_menu import option_menu
import requests
import json
import joblib
import numpy as np
from PIL import Image
import os
import io

# --- IMPORTS FOR MODELS ---
# These are the fixes for your errors
import pandas as pd
import onnxruntime as ort 

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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }

    .css-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #0277BD;
    }
    
    div.stButton > button {
        background-color: #0277BD;
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #01579B;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR & CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) 
    st.markdown("<h1 style='text-align: center; color: #0277BD;'>MedBot Pro</h1>", unsafe_allow_html=True)
    
    # API Key
    api_key = "AIzaSyBxYCfAwsyhbhiA8EQd6dcn-RdsZQ9xtZ8"
    
    st.write("") 
    
    selected = option_menu(
        menu_title=None,
        options=["AI Chatbot", "Diabetes Test", "Pneumonia Check"],
        icons=["chat-dots-fill", "activity", "lungs"],
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
        # Ensure 'import onnxruntime as ort' is at the top of the file
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
# 4. DATA PROCESSING FUNCTIONS 
# --------------------------------------------------------------------

def process_image_yolo(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def calculate_bmi(height_cm, weight_kg):
    if height_cm == 0 or height_cm < 50:
        return 0
    return weight_kg / ((height_cm / 100) ** 2)

def prepare_diabetes_features(data):
    scaler = MODELS['diabetes_scaler']
    age = data.get('Age')
    weight = data.get('Weight')
    height = data.get('Height')
    bp = data.get('BP')
    glucose = data.get('Glucose')
    pregnancies = data.get('Pregnancies', 0)

    bmi = calculate_bmi(height, weight)
    # Default values for missing fields
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

# System Prompt for the AI
medical_prompt = """
You are MedBot, a professional medical AI assistant. 
Answer questions clearly and empathetically. 
If a user asks about a specific medical test result (like "Diabetic"), explain what it means and suggest lifestyle changes.
ALWAYS end with a disclaimer that you are an AI, not a doctor.
"""

# --- 5. PAGE CONTENT LOGIC ---

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
        weight = st.number_input("Weight (kg)", 10, 300, 70)
        height = st.number_input("Height (cm)", 50, 250, 170)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
    with col3:
        # BMI is calculated automatically in background
        st.info("BMI will be calculated from Weight and Height.")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Analyze Result", type="primary"):
        if MODELS and 'diabetes_model' in MODELS:
            try:
                # Prepare data dictionary
                input_dict = {
                    'Age': age, 'Weight': weight, 'Height': height,
                    'BP': bp, 'Glucose': glucose, 'Pregnancies': pregnancies
                }
                
                # Preprocess
                final_features = prepare_diabetes_features(input_dict)
                
                # Predict
                prediction = MODELS['diabetes_model'].predict(final_features)[0]
                result_str = "Diabetic (High Risk)" if prediction == 1 else "Healthy (Low Risk)"
                color = "#D32F2F" if prediction == 1 else "#388E3C"
                
                # Display
                st.markdown(f"### Result: <span style='color:{color}'>{result_str}</span>", unsafe_allow_html=True)
                
                # AI Explanation
                bmi = calculate_bmi(height, weight)
                analysis_prompt = f"The patient has tested: {result_str}. Glucose: {glucose}, BMI: {bmi:.1f}, Age: {age}. Explain this result briefly."
                ai_explanation = ask_medbot(analysis_prompt, medical_prompt)
                st.info(f"👨‍⚕️ **Dr. AI Analysis:**\n\n{ai_explanation}")
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.warning("⚠️ Models not loaded. Check 'models/' folder.")

# === SECTION 3: PNEUMONIA CHECK ===
elif selected == "Pneumonia Check":
    st.title("🫁 Pneumonia X-Ray Check")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and MODELS:
        st.image(uploaded_file, width=300)
        if st.button("Analyze Image"):
            try:
                # Read file
                image_bytes = uploaded_file.read()
                
                # Preprocess for ONNX
                img_input = process_image_yolo(image_bytes)
                
                # Run Inference
                input_name = MODELS['pneumonia_input_name']
                output_name = MODELS['pneumonia_output_name']
                result = MODELS['pneumonia_session'].run([output_name], {input_name: img_input})
                
                # Process Result (Assuming classification output)
                # Note: You might need to adjust this depending on exact YOLO output format
                # This is a generic handler for classification
                prediction_idx = np.argmax(result[0])
                classes = MODELS['pneumonia_classes']
                final_result = classes[prediction_idx] if prediction_idx < len(classes) else "Unknown"
                
                st.markdown(f"### Result: **{final_result}**")
                
                # AI Explanation
                ai_explanation = ask_medbot(f"X-Ray result shows: {final_result}. Explain what this means.", medical_prompt)
                st.info(f"👨‍⚕️ **Dr. AI Analysis:**\n\n{ai_explanation}")

            except Exception as e:
                st.error(f"Image Analysis Error: {e}")
