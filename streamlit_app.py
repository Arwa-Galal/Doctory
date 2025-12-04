import streamlit as st
import google.generativeai as genai
import joblib
import numpy as np
import os
from PIL import Image

# --- 1. CONFIGURATION (SETUP) ---
st.set_page_config(
    page_title="Doctory AI",
    page_icon="🩺",
    layout="wide"
)

# *** PUT YOUR GOOGLE API KEY HERE ***
# Replace the text inside quotes with your actual key starting with AIza...
GOOGE_API_KEY = "AIzaSyCGlprvtIdX7vTQCPBGi7dv4FcQ4usEpdI"

# Configure Gemini
try:
    genai.configure(api_key=GOOGE_API_KEY)
    model_ai = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"API Key Error: {e}")

# --- 2. LOAD MODELS (With Error Handling) ---
@st.cache_resource
def load_models():
    models = {}
    try:
        # Load Diabetes Model (Adjust path if needed)
        # Note: We import xgboost globally so joblib can find it
        import xgboost
        models['diabetes'] = joblib.load('models/diabetes_model_package/diabetes_ensemble_model.joblib')
        # models['heart'] = joblib.load('models/HeartRisk_model_package/HeartRisk_model.joblib')
    except Exception as e:
        print(f"Model loading warning: {e}")
    return models

loaded_models = load_models()

# --- 3. SIDEBAR (NAVIGATION) ---
with st.sidebar:
    st.title("🩺 Doctory Menu")
    choice = st.radio(
        "Choose an action:", 
        ["💬 Chat with AI Doctor", "🩸 Diabetes Test", "🫁 Pneumonia Check", "🦠 Malaria Check"]
    )
    st.markdown("---")
    st.info("This is an AI Assistant. Please consult a real doctor for medical decisions.")

# --- 4. MAIN PAGES ---

# === PAGE 1: CHATBOT (The User wants this first) ===
if choice == "💬 Chat with AI Doctor":
    st.title("💬 Chat with Dr. AI")
    st.caption("Ask me anything about your symptoms or health...")

    # Initialize chat history if empty
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am Doctory. How can I help you today?"}]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Type your symptoms here..."):
        # 1. Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 2. Get AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Context for the AI to act like a doctor
                    full_prompt = f"Act as a professional and empathetic doctor. Answer this patient query: {prompt}"
                    response = model_ai.generate_content(full_prompt)
                    ai_text = response.text
                    st.write(ai_text)
                    # 3. Add AI response to history
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                except Exception as e:
                    st.error("Connection Error. Please check your API Key.")

# === PAGE 2: DIABETES TEST ===
elif choice == "🩸 Diabetes Test":
    st.title("🩸 Diabetes Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 30)
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose Level", 0, 300, 100)
    with col2:
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 79)
    with col3:
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    if st.button("Analyze Result"):
        if 'diabetes' in loaded_models:
            # Prepare data
            input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            
            try:
                prediction = loaded_models['diabetes'].predict(input_data)[0]
                
                # Get Explanation from AI
                result_str = "Diabetic" if prediction == 1 else "Healthy"
                prompt_analysis = f"Patient Data: Glucose {glucose}, BMI {bmi}, Age {age}. Model Result: {result_str}. Explain this briefly to the patient."
                explanation = model_ai.generate_content(prompt_analysis).text

                st.success(f"Prediction: **{result_str}**")
                st.info(f"👨‍⚕️ Dr. AI Note: {explanation}")
                
            except Exception as e:
                st.error(f"Calculation Error: {e}")
        else:
            st.warning("Diabetes model file not found in 'models/diabetes_model_package/'. Please check file path.")

# === PAGE 3: PNEUMONIA CHECK ===
elif choice == "🫁 Pneumonia Check":
    st.title("🫁 Pneumonia X-Ray Check")
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, width=300)
        if st.button("Check Image"):
            st.info("To enable Image Analysis, you need to ensure the ONNX model is loaded correctly.")
            # Code for ONNX image processing goes here.
            # For now, let's ask the AI what it thinks generally (Simulator)
            st.write("Processing image...")

# === PAGE 4: MALARIA CHECK ===
elif choice == "🦠 Malaria Check":
    st.title("🦠 Malaria Cell Check")
    st.write("Upload cell image here...")
    # Add malaria upload logic here
