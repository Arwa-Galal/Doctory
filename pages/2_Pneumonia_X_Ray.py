import streamlit as st
from PIL import Image
import io
import numpy as np
from utils import MODELS, process_image_yolo 

# --- Custom CSS Injection (Extracted from xray.html) ---
def local_css():
    st.markdown(
        """
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        /* Global Body Styling */
        .stApp {
            font-family: 'Inter', sans-serif;
            /* Linear gradient background from HTML */
            background: linear-gradient(135deg, #bbdefb 0%, #90caf9 50%, #64b5f6 100%);
        }

        /* Headers (H1, H2, H3) */
        h1, h2, h3 {
            color: #0d47a1 !important;
            font-weight: 700;
            text-align: center;
        }
        
        /* Hero-like Title Styling */
        h1 {
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
            margin-bottom: 1.5rem;
        }

        /* Glassmorphism Card Effect for the Main Container */
        .block-container {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(13, 71, 161, 0.3);
            border-radius: 25px;
            box-shadow: 0 25px 50px rgba(13, 71, 161, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.5);
            padding: 3rem !important;
            max-width: 900px;
            margin-top: 2rem;
        }

        /* File Uploader Styling */
        [data-testid='stFileUploader'] {
            background: rgba(255, 255, 255, 0.8);
            border: 2px solid rgba(13, 71, 161, 0.3);
            border-radius: 20px;
            padding: 15px;
            transition: all 0.4s ease;
        }
        
        [data-testid='stFileUploader']:hover {
            border-color: #0d47a1;
            transform: scale(1.01);
        }

        /* Button Styling (Modern Gradient) */
        .stButton > button {
            background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 35px !important;
            padding: 15px 50px !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
            box-shadow: 0 10px 30px rgba(13, 71, 161, 0.3) !important;
            transition: all 0.4s ease !important;
            width: 100%;
        }

        .stButton > button:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(13, 71, 161, 0.5) !important;
        }

        /* Success/Warning Message Styling to match result classes */
        .stAlert {
            border-radius: 20px;
            font-weight: 600;
        }

        /* Image Border Styling */
        img {
            border: 4px solid #0d47a1;
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(13, 71, 161, 0.3);
        }
        
        /* Text Colors */
        p, label {
            color: #0d47a1;
            font-size: 1.1rem;
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
    # Apply the custom CSS from HTML
    local_css()

    st.title("Chest X-ray Detection") # Changed to match HTML Title
    st.markdown("<p style='text-align: center;'>Upload a chest X-ray image, and the model will predict the condition.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a Chest X-Ray Image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Centering image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Spacing
        st.write("") 
        st.write("")

        if st.button("Analyze X-Ray"):
            with st.spinner('Analyzing image...'):
                try:
                    # Preprocessing using the helper function from utils.py
                    img_np = process_image_yolo(image_bytes, target_size=(224, 224))
                    
                    # Inference
                    outputs = pneumonia_session.run([pneumonia_output_name], {pneumonia_input_name: img_np})[0]
                    
                    probs = outputs[0]
                    top_index = np.argmax(probs)
                    confidence = float(probs[top_index])
                    prediction_label = pneumonia_classes[top_index]
                    
                    # Result Logic with Custom HTML styling for colors
                    if prediction_label == "Normal":
                        st.markdown(
                            f"""
                            <div style="background-color: rgba(56, 142, 60, 0.2); border: 1px solid #388e3c; color: #388e3c; padding: 20px; border-radius: 20px; text-align: center; margin-top: 20px;">
                                <h3 style="color: #388e3c !important; margin: 0;">Result: Normal</h3>
                                <p style="color: #388e3c; margin: 0;">Confidence: {confidence:.2%}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.balloons()
                    else:
                        # Dynamic color for Pneumonia types (Orange/Red)
                        color = "#f57c00" # Pneumonia color from HTML
                        bg_color = "rgba(245, 124, 0, 0.2)"
                        
                        st.markdown(
                            f"""
                            <div style="background-color: {bg_color}; border: 1px solid {color}; color: {color}; padding: 20px; border-radius: 20px; text-align: center; margin-top: 20px;">
                                <h3 style="color: {color} !important; margin: 0;">Result: {prediction_label}</h3>
                                <p style="color: {color}; margin: 0;">Confidence: {confidence:.2%}</p>
                                <p style="margin-top: 10px;"><strong>Warning:</strong> The image indicates a type of Pneumonia. Please consult a medical professional.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

pneumonia_predictor_page()
