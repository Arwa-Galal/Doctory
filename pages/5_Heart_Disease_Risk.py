import streamlit as st
import pandas as pd
from utils import MODELS

# --- Helper Functions for Feature Preparation ---
def calculate_bmi(height_cm, weight_kg):
    if height_cm == 0: return 0
    return weight_kg / ((height_cm / 100) ** 2)

def get_age_category(age):
    age = int(age)
    if 18 <= age <= 24: return 'Young'
    if 25 <= age <= 39: return 'Adult'
    if 40 <= age <= 54: return 'Mid-Aged'
    if 55 <= age <= 64: return 'Senior-Adult'
    if age >= 65: return 'Elderly'
    return 'Adult'

def prepare_heart_features(data):
    # Scaler
    scaler = MODELS['heart_scaler']
    
    # Inputs
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
    except (ValueError, IndexError):
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

    # Corrected Feature Order 
    final_feature_order = [
        'general_health', 'checkup', 'exercise', 'skin_cancer', 'other_cancer',
        'depression', 'diabetes', 'arthritis', 'age_category', 'height', 'weight',
        'bmi', 'bmi_group', 'alcohol_consumption', 'fruit_consumption', 'vegetables_consumption',
        'potato_consumption', 'sex_Female', 'sex_Male',
        'smoking_history_No', 'smoking_history_Yes'
    ]
    
    features = pd.DataFrame([feature_dict], columns=final_feature_order)
    return scaler.transform(features)

# --- Custom CSS with Enhanced Styling (Blue Theme) ---
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
                <h1 style="color: white !important; font-size: 3.5rem; text-shadow: 0 0 20px rgba(255,255,255,0.5);">❤️ Heart Disease Risk</h1>
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
                # Use local helper function with corrected feature order
                features = prepare_heart_features(data)
                prob = MODELS['heart_model'].predict_proba(features)[0][1]
                
                risk_percent = prob * 100
                prediction_label = "High Risk" if prob > 0.5 else "Low Risk"
                
                # Result Display Logic
                if prob > 0.5:
                     # High Risk - Red
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(211, 47, 47, 0.2); border: 1px solid #d32f2f; color: #d32f2f; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px; box-shadow: 0 10px 30px rgba(13, 71, 161, 0.2);">
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
                        <div style="background-color: rgba(56, 142, 60, 0.2); border: 1px solid #388e3c; color: #388e3c; padding: 30px; border-radius: 20px; text-align: center; margin-top: 30px; box-shadow: 0 10px 30px rgba(13, 71, 161, 0.2);">
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
