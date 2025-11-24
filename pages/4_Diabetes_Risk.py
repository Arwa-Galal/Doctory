import streamlit as st
from utils import MODELS, prepare_diabetes_features, calculate_bmi

if MODELS is None:
    st.error("Model initialization failed. Check your 'models/' folder structure.")
    st.stop()

diabetes_model = MODELS['diabetes_model']

def diabetes_predictor_page():
    st.header("Diabetes Risk Prediction")
    st.markdown("Enter the required biometric data to estimate the risk of Type II Diabetes.")

    # Input form 
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (Years)", min_value=18, max_value=120, value=30)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.1f")
            bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=120, value=70)
        
        with col2:
            pregnancies = st.number_input("Pregnancies (0 for males/default)", min_value=0, max_value=15, value=0)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0, format="%.1f")
            glucose = st.number_input("Plasma Glucose (mg/dL)", min_value=50, max_value=250, value=100)
            
        submitted = st.form_submit_button("Calculate Risk")

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
                
                st.success(f"Prediction: {prediction_label}")
                st.metric("Risk Probability", f"{risk_percent:.1f}%", help=f"Your current BMI is {calculate_bmi(height, weight):.1f}")

                if prob > 0.6:
                    st.warning("High risk detected. Consult a physician for testing.")
                elif prob < 0.4:
                    st.info("Risk appears low, but maintaining a healthy lifestyle is key.")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

diabetes_predictor_page()