import streamlit as st
from PIL import Image
import io
import numpy as np
from utils import process_image, load_all_models

# --- Custom CSS with Enhanced Styling from HTML ---
MODELS = load_all_models()

if st.sidebar.button("🏠 Back to Home"):
    st.switch_page("app.py")
def local_css():
    st.markdown(
        """
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        /* Global Body Styling */
        .stApp {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #bbdefb 0%, #90caf9 50%, #64b5f6 100%);
        }

        /* Remove default Streamlit padding */
        .main .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            max-width: 100%;
        }

        /* Hero Section */
        .hero-section {
            background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            position: relative;
            overflow: hidden;
            margin: -5rem -5rem 0 -5rem;
            padding: 2rem;
        }

        .hero-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.2) 0%, transparent 70%);
            animation: pulse 10s infinite ease-in-out;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.5; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.1); }
        }

        .hero-content {
            position: relative;
            z-index: 2;
            animation: fadeInUp 1.5s ease-out;
        }

        .hero-title {
            font-weight: 700;
            font-size: 4.5rem;
            color: white !important;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
            animation: glow 2s infinite alternate;
            margin-bottom: 2rem;
        }

        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
            to { text-shadow: 0 0 30px rgba(255, 255, 255, 0.8); }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(50px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Content Section */
        .content-section {
            padding: 120px 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Glassmorphism Card Effect */
        .glass-card {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(13, 71, 161, 0.3);
            border-radius: 25px;
            box-shadow: 0 25px 50px rgba(13, 71, 161, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.5);
            padding: 50px;
            margin-bottom: 80px;
            position: relative;
            transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(13, 71, 161, 0.1) 0%, transparent 100%);
            border-radius: 25px;
            z-index: -1;
        }

        .glass-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 35px 70px rgba(13, 71, 161, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6);
        }

        /* Headers */
        h1, h2, h3 {
            color: #0d47a1 !important;
            font-weight: 700;
            text-align: center;
        }

        h2 {
            margin-bottom: 2rem;
        }

        /* File Uploader Styling */
        [data-testid='stFileUploader'] {
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid rgba(13, 71, 161, 0.3);
            border-radius: 20px;
            padding: 20px;
            transition: all 0.4s ease;
        }
        
        [data-testid='stFileUploader']:hover {
            border-color: #0d47a1;
            transform: scale(1.02);
            box-shadow: 0 10px 30px rgba(13, 71, 161, 0.2);
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 35px !important;
            padding: 18px 50px !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
            box-shadow: 0 10px 30px rgba(13, 71, 161, 0.3) !important;
            transition: all 0.4s ease !important;
            width: 100%;
            position: relative;
            overflow: hidden;
        }

        .stButton > button:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(13, 71, 161, 0.5) !important;
        }

        /* Image Styling */
        img {
            border: 4px solid #0d47a1;
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(13, 71, 161, 0.3);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            max-width: 400px;
            margin: 0 auto;
            display: block;
        }

        img:hover {
            transform: scale(1.05);
            box-shadow: 0 20px 50px rgba(13, 71, 161, 0.5);
        }

        /* Result Box Animations */
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-30px) scale(0.9); }
            to { opacity: 1; transform: translateX(0) scale(1); }
        }

        .result-box {
            animation: slideIn 0.6s ease-out;
            box-shadow: 0 10px 30px rgba(13, 71, 161, 0.2);
        }
        
        /* Text Colors */
        p, label {
            color: #0d47a1;
            font-size: 1.1rem;
            text-align: center;
        }

        /* Spinner */
        .stSpinner > div {
            border-color: #0d47a1 !important;
        }

        /* Info Section */
        .info-text {
            line-height: 1.8;
            font-size: 1.1rem;
        }

        /* Footer */
        .footer {
            background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
            color: white;
            text-align: center;
            padding: 50px 0;
            margin-top: 120px;
        }

        .footer p {
            color: white !important;
            margin: 0;
        }

        /* Icon styling */
        .icon {
            margin-right: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Ensure models are loaded before proceeding
if MODELS is None:
    st.error("Model initialization failed. Check your 'models/' folder structure.")
    st.stop()

# Get specific models/names from the global dict
pneumonia_session = MODELS['pneumonia_session']
pneumonia_input_name = MODELS['pneumonia_input_name']
pneumonia_output_name = MODELS['pneumonia_output_name']
pneumonia_classes = MODELS['pneumonia_classes']

def pneumonia_predictor_page():
    # Apply custom CSS
    local_css()

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-content">
                <h1 class="hero-title">🫁 Chest X-ray Detection</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Main Content Section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    # Model Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h2>🧠 Try the Model</h2>", unsafe_allow_html=True)
    st.markdown("<p>Upload a chest X-ray image, and the model will predict the condition.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a Chest X-Ray Image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Center image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_column_width=True)

        # Spacing
        st.write("")

        if st.button("🔍 Analyze X-Ray"):
            with st.spinner('Analyzing image...'):
                try:
                    # Preprocessing
                    img_np = process_image_yolo(image_bytes, target_size=(224, 224))
                    
                    # Inference
                    outputs = pneumonia_session.run([pneumonia_output_name], {pneumonia_input_name: img_np})[0]
                    
                    probs = outputs[0]
                    top_index = np.argmax(probs)
                    confidence = float(probs[top_index])
                    prediction_label = pneumonia_classes[top_index]
                    
                    # Result Display
                    if prediction_label == "Normal":
                        st.markdown(
                            f"""
                            <div class="result-box" style="background-color: rgba(56, 142, 60, 0.2); border: 1px solid #388e3c; color: #388e3c; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px;">
                                <h3 style="color: #388e3c !important; margin: 0; font-size: 1.8rem;">Result: Normal</h3>
                                <p style="color: #388e3c; margin: 10px 0 0 0; font-size: 1.3rem;">Confidence: {confidence:.2%}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.balloons()
                    else:
                        # Determine color based on condition
                        if "COVID" in prediction_label:
                            color = "#d32f2f"
                            bg_color = "rgba(211, 47, 47, 0.2)"
                        else:
                            color = "#f57c00"
                            bg_color = "rgba(245, 124, 0, 0.2)"
                        
                        st.markdown(
                            f"""
                            <div class="result-box" style="background-color: {bg_color}; border: 1px solid {color}; color: {color}; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px;">
                                <h3 style="color: {color} !important; margin: 0; font-size: 1.8rem;">Result: {prediction_label}</h3>
                                <p style="color: {color}; margin: 10px 0; font-size: 1.3rem;">Confidence: {confidence:.2%}</p>
                                <p style="margin-top: 15px; font-size: 1.1rem;"><strong>⚠️ Warning:</strong> The image indicates a condition. Please consult a medical professional.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card

    # Explanation Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h2>ℹ️ How Does the Model Work?</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p class="info-text">This is a convolutional neural network (CNN) model trained on a dataset of chest X-ray images. 
        It analyzes the uploaded image and classifies it into categories like Normal, Pneumonia, or COVID based on learned patterns.</p>
        <p class="info-text"><strong>Note:</strong> This model is for educational purposes. 
        Always consult qualified medical professionals for actual diagnosis.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card

    st.markdown('</div>', unsafe_allow_html=True)  # Close content-section

    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2023 Chest X-ray Detection Page. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

pneumonia_predictor_page()
