import streamlit as st
from utils import MODELS, prepare_diabetes_features, calculate_bmi

# --- Custom CSS with Enhanced Styling (Blue Theme from X-ray) ---
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

        /* Form Styling */
        .stNumberInput > div > div > input {
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid rgba(13, 71, 161, 0.3);
            border-radius: 15px;
            padding: 12px;
            font-size: 1.1rem;
            transition: all 0.4s ease;
            color: #0d47a1;
        }

        .stNumberInput > div > div > input:focus {
            border-color: #0d47a1;
            box-shadow: 0 0 0 0.4rem rgba(13, 71, 161, 0.15);
            transform: scale(1.02);
        }

        /* Label Styling */
        label {
            color: #0d47a1 !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
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

        /* Form Submit Button */
        .stForm button[kind="primary"] {
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
        }

        .stForm button[kind="primary"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(13, 71, 161, 0.5) !important;
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
        p {
            color: #0d47a1;
            font-size: 1.1rem;
            text-align: center;
        }

        /* Spinner */
        .stSpinner > div {
            border-color: #0d47a1 !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
            font-weight: 700;
            color: #0d47a1;
        }

        [data-testid="stMetricLabel"] {
            font-size: 1.2rem;
            color: #0d47a1;
            font-weight: 600;
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

        /* Custom Alert Boxes */
        .stAlert {
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
        }

        /* Column spacing */
        [data-testid="column"] {
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Ensure models are loaded
if MODELS is None:
    st.error("Model initialization failed. Check your 'models/' folder structure.")
    st.stop()

diabetes_model = MODELS['diabetes_model']

def diabetes_predictor_page():
    # Apply custom CSS
    local_css()

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-content">
                <h1 class="hero-title">🩺 Diabetes Risk Prediction</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Main Content Section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    # Model Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h2>📊 Check Your Risk</h2>", unsafe_allow_html=True)
    st.markdown("<p>Enter the required biometric data to estimate the risk of Type II Diabetes.</p>", unsafe_allow_html=True)

    # Input form 
    with st.form("diabetes_form"):
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (Years)", min_value=18, max_value=120, value=30)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.1f")
            bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=120, value=70)
        
        with col2:
            pregnancies = st.number_input("Pregnancies (0 for males/default)", min_value=0, max_value=15, value=0)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0, format="%.1f")
            glucose = st.number_input("Plasma Glucose (mg/dL)", min_value=50, max_value=250, value=100)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Calculate Risk")

    if submitted:
        # Check for invalid inputs
        if height == 0 or weight == 0:
            st.error("Height and Weight must be greater than zero.")
            return

        data = {
            'Age': age, 'Weight': weight, 'Height': height, 'BP': bp,
            'Glucose': glucose, 'Pregnancies': pregnancies
        }
        
        with st.spinner('Calculating risk...'):
            try:
                # Use helper function from utils.py
                features = prepare_diabetes_features(data)
                prob = diabetes_model.predict_proba(features)[0][1]
                
                risk_percent = prob * 100
                prediction_label = "Diabetic" if prob > 0.5 else "Non-Diabetic"
                bmi = calculate_bmi(height, weight)
                
                # Result Display
                if prob <= 0.4:
                    # Low Risk - Green
                    st.markdown(
                        f"""
                        <div class="result-box" style="background-color: rgba(56, 142, 60, 0.2); border: 1px solid #388e3c; color: #388e3c; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px;">
                            <h3 style="color: #388e3c !important; margin: 0; font-size: 1.8rem;">Result: {prediction_label}</h3>
                            <p style="color: #388e3c; margin: 10px 0; font-size: 1.5rem; font-weight: 700;">Risk Probability: {risk_percent:.1f}%</p>
                            <p style="color: #388e3c; margin: 10px 0; font-size: 1.2rem;">BMI: {bmi:.1f}</p>
                            <p style="margin-top: 15px; font-size: 1.1rem;">✓ Risk appears low, but maintaining a healthy lifestyle is key.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.balloons()
                elif prob > 0.6:
                    # High Risk - Red/Orange (using COVID color from HTML)
                    st.markdown(
                        f"""
                        <div class="result-box" style="background-color: rgba(211, 47, 47, 0.2); border: 1px solid #d32f2f; color: #d32f2f; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px;">
                            <h3 style="color: #d32f2f !important; margin: 0; font-size: 1.8rem;">Result: {prediction_label}</h3>
                            <p style="color: #d32f2f; margin: 10px 0; font-size: 1.5rem; font-weight: 700;">Risk Probability: {risk_percent:.1f}%</p>
                            <p style="color: #d32f2f; margin: 10px 0; font-size: 1.2rem;">BMI: {bmi:.1f}</p>
                            <p style="margin-top: 15px; font-size: 1.1rem;"><strong>⚠️ Warning:</strong> High risk detected. Consult a physician for testing.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    # Moderate Risk - Orange (pneumonia color from HTML)
                    st.markdown(
                        f"""
                        <div class="result-box" style="background-color: rgba(245, 124, 0, 0.2); border: 1px solid #f57c00; color: #f57c00; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px;">
                            <h3 style="color: #f57c00 !important; margin: 0; font-size: 1.8rem;">Result: {prediction_label}</h3>
                            <p style="color: #f57c00; margin: 10px 0; font-size: 1.5rem; font-weight: 700;">Risk Probability: {risk_percent:.1f}%</p>
                            <p style="color: #f57c00; margin: 10px 0; font-size: 1.2rem;">BMI: {bmi:.1f}</p>
                            <p style="margin-top: 15px; font-size: 1.1rem;">⚡ Moderate risk. Consider lifestyle modifications and regular monitoring.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card

    # Explanation Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h2>ℹ️ About Diabetes Risk Assessment</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p class="info-text">This machine learning model assesses Type II Diabetes risk based on key biometric indicators including age, BMI, blood pressure, glucose levels, and pregnancy history.</p>
        <p class="info-text"><strong>Risk Categories:</strong></p>
        <ul style="text-align: left; max-width: 700px; margin: 20px auto; line-height: 1.8; font-size: 1.1rem; color: #0d47a1;">
            <li><strong>Low Risk (< 40%):</strong> Continue healthy lifestyle habits</li>
            <li><strong>Moderate Risk (40-60%):</strong> Consider lifestyle modifications</li>
            <li><strong>High Risk (> 60%):</strong> Seek medical consultation for testing</li>
        </ul>
        <p class="info-text"><strong>Important:</strong> This tool provides risk estimates for educational purposes only. 
        It cannot diagnose diabetes. Always consult healthcare professionals for proper diagnosis and treatment.</p>
        <p class="info-text"><strong>Prevention Tips:</strong> Maintain a healthy weight, exercise regularly, eat a balanced diet, 
        monitor blood sugar levels, and get regular check-ups.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card

    st.markdown('</div>', unsafe_allow_html=True)  # Close content-section

    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2023 Diabetes Risk Prediction. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

diabetes_predictor_page()
