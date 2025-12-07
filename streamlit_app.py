import streamlit as st
from utils import load_css, render_sidebar,load_all_models, MODELS

st.set_page_config(page_title="Doctory", page_icon="🩺", layout="wide")
load_css()
render_sidebar()

# --- HERO SECTION ---
with st.container(border=True):
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=120)
        st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Doctory AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Your Intelligent Medical Companion</p>", unsafe_allow_html=True)

st.write("") # Spacer
st.markdown("<h3 style='text-align: center;'>Choose a Service</h3>", unsafe_allow_html=True)
st.write("") 

# --- SERVICE CARDS ---

# Row 1: Chat and Diabetes
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=60)
        st.markdown("<h3>AI Doctor</h3>", unsafe_allow_html=True)
        st.markdown("<p>Chat with our smart assistant.</p>", unsafe_allow_html=True)
        if st.button("Start Chat"):
            st.switch_page("pages/1_AI_Chatbot.py")

with col2:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/2865/2865769.png", width=60)
        st.markdown("<h3>Diabetes</h3>", unsafe_allow_html=True)
        st.markdown("<p>Check risk based on vitals.</p>", unsafe_allow_html=True)
        if st.button("Check Risk"):
            st.switch_page("pages/4_Diabetes_Risk.py")

# Row 2: Pneumonia and Malaria
col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=60)
        st.markdown("<h3>Pneumonia</h3>", unsafe_allow_html=True)
        st.markdown("<p>Analyze Chest X-Ray images.</p>", unsafe_allow_html=True)
        if st.button("Check Lungs"):
            st.switch_page("pages/2_Pneumonia_X_Ray.py")

with col4:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=60)
        st.markdown("<h3>Malaria</h3>", unsafe_allow_html=True)
        st.markdown("<p>Analyze cell images.</p>", unsafe_allow_html=True)
        if st.button("Check Cells"):
            st.switch_page("pages/3_Malaria_Blood_Smear.py")
