import streamlit as st
from utils import MODELS, prepare_heart_features

if MODELS is None:
    st.error("Model initialization failed. Check your 'models/' folder structure.")
    st.stop()

heart_model = MODELS['heart_model']

def heart_predictor_page():
    st.header("Heart Disease Risk Prediction")
    st.markdown("Provide lifestyle and health inputs to assess your 10-year risk of heart disease.")

    # Input form 
    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        
        # COLUMN 1: Basic Biometrics
        with col1:
            st.subheader("Biometrics")
            age = st.number_input("Age (Years)", min_value=18, max_value=120, value=45)
            sex = st.selectbox("Sex", ['Female', 'Male'])
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, format="%.1f")
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=80.0, format="%.1f")

        # COLUMN 2: Medical History
        with col2:
            st.subheader("Health History")
            general_health = st.selectbox("General Health Status", ['Very Good', 'Good', 'Fair', 'Poor', 'Excellent'])
            checkup = st.selectbox("Last Health Checkup", ['Past 1 year', 'Past 2 years', 'Past 5 years', 'More than 5 years', 'Never'])
            diabetes = st.selectbox("Diabetes Status", ['No', 'No Pre Diabetes', 'Only during pregnancy', 'Yes'])
            arthritis = st.selectbox("Have Arthritis?", ['No', 'Yes'])
            depression = st.selectbox("Have Depression?", ['No', 'Yes'])

        # COLUMN 3: Lifestyle
        with col3:
            st.subheader("Lifestyle")
            smoking = st.selectbox("Smoking History", ['Never', 'Former', 'Current'])
            exercise = st.selectbox("Any physical exercise in past 30 days?", ['No', 'Yes'])
            alcohol = st.selectbox("Avg. Alcoholic drinks per day", ['Never', 'Occasionally', 'Weekly', 'Daily'], index=0)
            fruit = st.selectbox("Fruit servings per day", ['0', '1–2', '3–5', '6–7'])
            vegetables = st.selectbox("Vegetable servings per day", ['0', '1–2', '3–5', '6–7'])
            fried_potato = st.selectbox("Fried Potato consumption", ['Rarely', 'Weekly', 'Several times per week'])


        submitted = st.form_submit_button("Predict Heart Risk")

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
        
        with st.spinner('Calculating risk...'):
            try:
                # Use helper function from utils.py
                features = prepare_heart_features(data)
                prob = MODELS['heart_model'].predict_proba(features)[0][1]
                
                risk_percent = prob * 100
                prediction_label = "High Risk of Heart Disease" if prob > 0.5 else "Low Risk of Heart Disease"
                
                st.success(f"Prediction: {prediction_label}")
                st.metric("10-Year Risk Probability", f"{risk_percent:.1f}%")

                if prob > 0.5:
                    st.error("High risk detected. Consult a cardiologist for a complete assessment.")
                else:
                    st.info("Risk appears manageable. Maintain healthy habits and regular checkups.")
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")

heart_predictor_page()