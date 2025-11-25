import streamlit as st
from utils import MODELS, prepare_heart_features

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
            min-height: 60vh;
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
            font-size: 3.5rem;
            color: white !important;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
            animation: glow 2s infinite alternate;
            margin-bottom: 1rem;
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
            padding: 60px 2rem;
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
            padding: 40px;
            margin-bottom: 40px;
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
            transform: translateY(-5px);
            box-shadow: 0 35px 70px rgba(13, 71, 161, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6);
        }

        /* Headers */
        h2, h3 {
            color: #0d47a1 !important;
            font-weight: 700;
            text-align: center;
        }
        
        /* Form Inputs */
        .stNumberInput > div > div > input, .stSelectbox > div > div {
             background: rgba(255, 255, 255, 0.9);
            border: 2px solid rgba(13, 71, 161, 0.3);
            border-radius: 15px;
            color: #0d47a1;
        }

        /* Labels */
        label {
            color: #0d47a1 !important;
            font-weight: 600 !important;
        }

        /* Submit Button */
        .stForm button[kind="primary"] {
            background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 35px !important;
            padding: 15px 40px !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            box-shadow: 0 10px 30px rgba(13, 71, 161, 0.3) !important;
            transition: all 0.4s ease !important;
            width: 100%;
        }

        .stForm button[kind="primary"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(13, 71, 161, 0.5) !important;
        }

        /* Result Box */
        .result-box {
            animation: slideIn 0.6s ease-out;
            box-shadow: 0 10px 30px rgba(13, 71, 161, 0.2);
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-30px) scale(0.9); }
            to { opacity: 1; transform: translateX(0) scale(1); }
        }
        
        /* Spinner */
        .stSpinner > div {
            border-color: #0d47a1 !important;
        }

        /* Footer */
        .footer {
            background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
            color: white;
            text-align: center;
            padding: 30px 0;
            margin-top: 60px;
        }
        .footer p {
             color: white !important;
             margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Ensure models are loaded
if MODELS is None:
    st.error("Model initialization failed. Check your 'models/' folder structure.")
    st.stop()

heart_model = MODELS['heart_model']

def heart_predictor_page():
    # Apply CSS
    local_css()

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-content">
                <h1 class="hero-title">❤️ Heart Disease Risk</h1>
                <h3 style="color: white !important;">AI-Powered Assessment Tool</h3>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Main Content
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    # Form Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h2>🩺 Enter Health Metrics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#0d47a1;'>Provide your lifestyle and health inputs for a 10-year risk assessment.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Input form 
    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        
        # COLUMN 1: Basic Biometrics
        with col1:
            st.markdown("### 👤 Biometrics")
            age = st.number_input("Age (Years)", min_value=18, max_value=120, value=45)
            sex = st.selectbox("Sex", ['Female', 'Male'])
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, format="%.1f")
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=80.0, format="%.1f")

        # COLUMN 2: Medical History
        with col2:
            st.markdown("### 🏥 Health History")
            general_health = st.selectbox("General Health Status", ['Very Good', 'Good', 'Fair', 'Poor', 'Excellent'])
            checkup = st.selectbox("Last Health Checkup", ['Past 1 year', 'Past 2 years', 'Past 5 years', 'More than 5 years', 'Never'])
            diabetes = st.selectbox("Diabetes Status", ['No', 'No Pre Diabetes', 'Only during pregnancy', 'Yes'])
            arthritis = st.selectbox("Have Arthritis?", ['No', 'Yes'])
            depression = st.selectbox("Have Depression?", ['No', 'Yes'])

        # COLUMN 3: Lifestyle
        with col3:
            st.markdown("### 🥗 Lifestyle")
            smoking = st.selectbox("Smoking History", ['Never', 'Former', 'Current'])
            exercise = st.selectbox("Any physical exercise in past 30 days?", ['No', 'Yes'])
            alcohol = st.selectbox("Avg. Alcoholic drinks per day", ['Never', 'Occasionally', 'Weekly', 'Daily'], index=0)
            fruit = st.selectbox("Fruit servings per day", ['0', '1–2', '3–5', '6–7'])
            vegetables = st.selectbox("Vegetable servings per day", ['0', '1–2', '3–5', '6–7'])
            fried_potato = st.selectbox("Fried Potato consumption", ['Rarely', 'Weekly', 'Several times per week'])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Predict Heart Risk")

    if submitted:
        if height == 0 or weight == 0:
            st.error("Height and Weight must be greater than zero.")
            return

        data = {
            'Age': age, 'Sex': sex, 'Height': height, 'Weight': weight,
            'General_Health': general_health, 'Checkup': checkup, 'Diabetes': diabetes,
            'Arthritis': arthritis, 'Depression': depression,
            'Smoking_History': smoking, 'Exercise': exercise,
            'Alcohol_Consumption': alcohol, 'Fruit_Consumption': fruit,
            'Vegetables_Consumption': vegetables, 'FriedPotato_Consumption': fried_potato,
        }
        
        with st.spinner('Analyzing cardiovascular health...'):
            try:
                # Use helper function from utils.py
                features = prepare_heart_features(data)
                prob = MODELS['heart_model'].predict_proba(features)[0][1]
                
                risk_percent = prob * 100
                prediction_label = "High Risk" if prob > 0.5 else "Low Risk"
                
                # Result Display Logic
                if prob > 0.5:
                     # High Risk - Red
                    st.markdown(
                        f"""
                        <div class="result-box" style="background-color: rgba(211, 47, 47, 0.2); border: 1px solid #d32f2f; color: #d32f2f; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px;">
                            <h3 style="color: #d32f2f !important; margin: 0; font-size: 1.8rem;">Result: {prediction_label} of Heart Disease</h3>
                            <p style="color: #d32f2f; margin: 10px 0; font-size: 1.5rem; font-weight: 700;">10-Year Risk Probability: {risk_percent:.1f}%</p>
                            <p style="margin-top: 15px; font-size: 1.1rem;"><strong>⚠️ High risk detected.</strong> Please consult a cardiologist for a complete assessment.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    # Low Risk - Green
                    st.markdown(
                        f"""
                        <div class="result-box" style="background-color: rgba(56, 142, 60, 0.2); border: 1px solid #388e3c; color: #388e3c; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px;">
                            <h3 style="color: #388e3c !important; margin: 0; font-size: 1.8rem;">Result: {prediction_label} of Heart Disease</h3>
                            <p style="color: #388e3c; margin: 10px 0; font-size: 1.5rem; font-weight: 700;">10-Year Risk Probability: {risk_percent:.1f}%</p>
                            <p style="margin-top: 15px; font-size: 1.1rem;">✓ Risk appears manageable. Maintain healthy habits and regular checkups.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.balloons()
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True) # Close glass-card
    st.markdown('</div>', unsafe_allow_html=True) # Close content-section
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2023 Heart Risk Prediction. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

heart_predictor_page()
