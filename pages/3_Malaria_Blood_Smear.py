import streamlit as st
from PIL import Image
import io
import numpy as np
from utils import MODELS, process_image_keras

# --- Custom CSS with Enhanced Styling ---
def local_css():
    st.markdown(
        """
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        /* Global Body Styling */
        .stApp {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 50%, #e57373 100%);
        }

        /* Remove default Streamlit padding */
        .main .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            max-width: 100%;
        }

        /* Hero Section */
        .hero-section {
            background: linear-gradient(135deg, #c62828 0%, #d32f2f 50%, #e53935 100%);
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
            border: 1px solid rgba(198, 40, 40, 0.3);
            border-radius: 25px;
            box-shadow: 0 25px 50px rgba(198, 40, 40, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.5);
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
            background: linear-gradient(135deg, rgba(198, 40, 40, 0.1) 0%, transparent 100%);
            border-radius: 25px;
            z-index: -1;
        }

        .glass-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 35px 70px rgba(198, 40, 40, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6);
        }

        /* Headers */
        h1, h2, h3 {
            color: #c62828 !important;
            font-weight: 700;
            text-align: center;
        }

        h2 {
            margin-bottom: 2rem;
        }

        /* File Uploader Styling */
        [data-testid='stFileUploader'] {
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid rgba(198, 40, 40, 0.3);
            border-radius: 20px;
            padding: 20px;
            transition: all 0.4s ease;
        }
        
        [data-testid='stFileUploader']:hover {
            border-color: #c62828;
            transform: scale(1.02);
            box-shadow: 0 10px 30px rgba(198, 40, 40, 0.2);
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #c62828 0%, #d32f2f 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 35px !important;
            padding: 18px 50px !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
            box-shadow: 0 10px 30px rgba(198, 40, 40, 0.3) !important;
            transition: all 0.4s ease !important;
            width: 100%;
            position: relative;
            overflow: hidden;
        }

        .stButton > button:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(198, 40, 40, 0.5) !important;
        }

        /* Image Styling */
        img {
            border: 4px solid #c62828;
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(198, 40, 40, 0.3);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            max-width: 400px;
            margin: 0 auto;
            display: block;
        }

        img:hover {
            transform: scale(1.05);
            box-shadow: 0 20px 50px rgba(198, 40, 40, 0.5);
        }

        /* Result Box Animations */
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-30px) scale(0.9); }
            to { opacity: 1; transform: translateX(0) scale(1); }
        }

        .result-box {
            animation: slideIn 0.6s ease-out;
            box-shadow: 0 10px 30px rgba(198, 40, 40, 0.2);
        }
        
        /* Text Colors */
        p, label {
            color: #c62828;
            font-size: 1.1rem;
            text-align: center;
        }

        /* Spinner */
        .stSpinner > div {
            border-color: #c62828 !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #c62828;
        }

        [data-testid="stMetricLabel"] {
            font-size: 1.2rem;
            color: #c62828;
        }

        /* Info Section */
        .info-text {
            line-height: 1.8;
            font-size: 1.1rem;
        }

        /* Footer */
        .footer {
            background: linear-gradient(135deg, #c62828 0%, #d32f2f 100%);
            color: white;
            text-align: center;
            padding: 50px 0;
            margin-top: 120px;
        }

        .footer p {
            color: white !important;
            margin: 0;
        }

        /* Custom Alert Boxes */
        .stAlert {
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Ensure models are loaded
if MODELS is None:
    st.error("Model initialization failed. Check your 'models/' folder structure.")
    st.stop()

malaria_session = MODELS['malaria_session']
malaria_input_name = MODELS['malaria_input_name']
malaria_output_name = MODELS['malaria_output_name']

def malaria_predictor_page():
    # Apply custom CSS
    local_css()

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-content">
                <h1 class="hero-title">🩸 Malaria Blood Cell Analysis</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Main Content Section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    # Model Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h2>🔬 Try the Model</h2>", unsafe_allow_html=True)
    st.markdown("<p>Upload a blood smear image of a single cell to classify it as <strong>Parasitized</strong> or <strong>Uninfected</strong>.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a Blood Smear Image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Center image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_column_width=True)

        # Spacing
        st.write("")

        if st.button("🔍 Analyze Blood Smear"):
            with st.spinner('Running AI analysis...'):
                try:
                    # Preprocessing
                    img_np = process_image_keras(image_bytes, target_size=(224, 224))
                    
                    # Inference
                    outputs = malaria_session.run([malaria_output_name], {malaria_input_name: img_np})[0]
                    
                    prob_uninfected = float(outputs[0][0])
                    
                    if prob_uninfected > 0.5:
                        label = "Uninfected"
                        confidence = prob_uninfected
                    else:
                        label = "Parasitized"
                        confidence = 1.0 - prob_uninfected

                    # Result Display
                    if label == "Uninfected":
                        st.markdown(
                            f"""
                            <div class="result-box" style="background-color: rgba(56, 142, 60, 0.2); border: 1px solid #388e3c; color: #388e3c; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px;">
                                <h3 style="color: #388e3c !important; margin: 0; font-size: 1.8rem;">Result: Uninfected</h3>
                                <p style="color: #388e3c; margin: 10px 0 0 0; font-size: 1.3rem;">Confidence: {confidence:.2%}</p>
                                <p style="margin-top: 15px; font-size: 1.1rem;">✓ The cell appears to be healthy and uninfected.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.balloons()
                    else:
                        st.markdown(
                            f"""
                            <div class="result-box" style="background-color: rgba(211, 47, 47, 0.2); border: 1px solid #d32f2f; color: #d32f2f; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px;">
                                <h3 style="color: #d32f2f !important; margin: 0; font-size: 1.8rem;">Result: Parasitized</h3>
                                <p style="color: #d32f2f; margin: 10px 0; font-size: 1.3rem;">Confidence: {confidence:.2%}</p>
                                <p style="margin-top: 15px; font-size: 1.1rem;"><strong>⚠️ Warning:</strong> The cell is likely parasitized. Seek urgent medical advice.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card

    # Explanation Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h2>ℹ️ About Malaria Detection</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p class="info-text">This deep learning model analyzes blood smear images to detect the presence of malaria parasites in blood cells. 
        The model has been trained to distinguish between parasitized cells (infected with Plasmodium parasites) and uninfected healthy cells.</p>
        <p class="info-text"><strong>Important:</strong> This tool is designed for educational and screening purposes only. 
        It should not replace professional medical diagnosis. Always consult qualified healthcare professionals for proper malaria testing and treatment.</p>
        <p class="info-text"><strong>How it works:</strong> The AI analyzes cellular features such as texture, color patterns, and structural abnormalities 
        that are characteristic of malaria infection to make its prediction.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card

    st.markdown('</div>', unsafe_allow_html=True)  # Close content-section

    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2023 Malaria Blood Cell Analysis. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

malaria_predictor_page()
