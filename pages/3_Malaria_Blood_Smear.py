import streamlit as st
from PIL import Image
import io
import numpy as np
from utils import MODELS, process_image_keras

if MODELS is None:
    st.error("Model initialization failed. Check your 'models/' folder structure.")
    st.stop()

malaria_session = MODELS['malaria_session']
malaria_input_name = MODELS['malaria_input_name']
malaria_output_name = MODELS['malaria_output_name']

def malaria_predictor_page():
    st.header("Malaria Blood Cell Analysis 🩸")
    st.markdown("Upload a blood smear image of a single cell to classify it as **Parasitized** or **Uninfected**.")

    uploaded_file = st.file_uploader("Choose a Blood Smear Image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption='Uploaded Cell Image.', use_column_width=True)

        if st.button("Analyze Blood Smear"):
            with st.spinner('Running AI analysis...'):
                try:
                    img_np = process_image_keras(image_bytes, target_size=(224, 224))
                    outputs = malaria_session.run([malaria_output_name], {malaria_input_name: img_np})[0]
                    
                    prob_uninfected = float(outputs[0][0])
                    
                    if prob_uninfected > 0.5:
                        label = "Uninfected"
                        confidence = prob_uninfected
                    else:
                        label = "Parasitized"
                        confidence = 1.0 - prob_uninfected

                    st.success(f"Prediction: {label}")
                    st.metric("Confidence Score", f"{confidence:.2%}")

                    if label == "Parasitized":
                        st.error("The cell is likely Parasitized. Seek urgent medical advice.")
                    else:
                        st.info("The cell appears Uninfected.")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

malaria_predictor_page()