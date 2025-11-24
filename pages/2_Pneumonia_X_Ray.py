import streamlit as st
from PIL import Image
import io
import numpy as np
# Import ALL shared resources and helper functions from the parent directory's utils.py
from utils import MODELS, process_image_yolo 

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
    st.header("Pneumonia X-Ray Classification 🩺")
    st.markdown("Upload a chest X-ray image to classify it as **Normal**, **Bacterial Pneumonia**, or **Viral Pneumonia**.")

    uploaded_file = st.file_uploader("Choose a Chest X-Ray Image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Analyze X-Ray"):
            with st.spinner('Running AI analysis...'):
                try:
                    # Preprocessing using the helper function from utils.py
                    img_np = process_image_yolo(image_bytes, target_size=(224, 224))
                    
                    # Inference
                    outputs = pneumonia_session.run([pneumonia_output_name], {pneumonia_input_name: img_np})[0]
                    
                    probs = outputs[0]
                    top_index = np.argmax(probs)
                    confidence = float(probs[top_index])
                    prediction_label = pneumonia_classes[top_index]
                    
                    st.success(f"Prediction: {prediction_label}")
                    st.metric("Confidence Score", f"{confidence:.2%}")

                    if prediction_label != "Normal":
                        st.warning("The image indicates a type of Pneumonia. Please consult a medical professional immediately.")
                    else:
                        st.balloons()
                        st.info("The image appears normal.")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

pneumonia_predictor_page()