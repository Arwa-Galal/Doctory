import streamlit as st
import numpy as np
from PIL import Image
from utils import load_css, load_all_models, process_image as process_image_yolo, ask_medbot, MEDICAL_PROMPT


# --- 1. Page Config ---
st.set_page_config(page_title="Pneumonia Check", page_icon="🫁", layout="wide")
load_css() # Loads the Blue/White Theme

# --- 2. Navigation ---
if st.sidebar.button("🏠 Back to Home"):
    st.switch_page("app.py")

# --- 3. Load Models ---
MODELS = load_all_models()

st.title("🫁 Pneumonia X-Ray Check")
st.markdown("Upload a chest X-Ray image to detect Pneumonia or Normal conditions.")

# --- 4. Input Section (The "Card" Look) ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    st.info("Supported formats: JPEG, PNG")
    
with col2:
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])

st.markdown('</div>', unsafe_allow_html=True)

# --- 5. Analysis Logic ---
if uploaded_file:
    # Show the image centered
    col_img1, col_img2, col_img3 = st.columns([1, 1, 1])
    with col_img2:
        st.image(uploaded_file, caption="Uploaded X-Ray", width=300)

    # Analyze Button
    if st.button("Analyze Image"):
        if MODELS and MODELS.get('pneumonia_sess'):
            try:
                with st.spinner("Analyzing lung patterns..."):
                    # 1. Preprocess
                    image_bytes = uploaded_file.read()
                    img_input = process_image(image_bytes)
                    
                    # 2. Inference (ONNX)
                    session = MODELS['pneumonia_sess']
                    input_name = MODELS['pneu_in']
                    output_name = MODELS['pneu_out']
                    
                    result = session.run([output_name], {input_name: img_input})
                    
                    # 3. Process Result
                    # Assuming classification: [Normal, Bacterial, Viral]
                    probs = result[0][0] # Adjust based on your model output shape
                    idx = np.argmax(probs)
                    classes = ["Normal", "Pneumonia (Bacterial)", "Pneumonia (Viral)"]
                    
                    # Safety check for index
                    if idx < len(classes):
                        final_result = classes[idx]
                    else:
                        final_result = "Unknown"

                    # 4. Color Logic (Green for Normal, Red for Disease)
                    if "Normal" in final_result:
                        color = "#388E3C"
