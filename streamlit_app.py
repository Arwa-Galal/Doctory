import streamlit as st
from utils import load_css, render_sidebar

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

# --- SERVICE CARDS (Using Border Containers) ---

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=60)
        st.markdown("<h3>AI Doctor</h3>", unsafe_allow_html=True)
        st.markdown("<p>Chat with our smart assistant.</p>", unsafe_allow_html=True)
        if st.button("Start Chat"):
            st.switch_page("pages/1_💬_Chat_With_Doctory.py")

with col2:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/2865/2865769.png", width=60)
        st.markdown("<h3>Diabetes</h3>", unsafe_allow_html=True)
        st.markdown("<p>Check risk based on vitals.</p>", unsafe_allow_html=True)
        if st.button("Check Risk"):
            st.switch_page("pages/2_🩸_Diabetes_Test.py")

with col3:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=60)
        st.markdown("<h3>Pneumonia</h3>", unsafe_allow_html=True)
        st.markdown("<p>Analyze Chest X-Ray images.</p>", unsafe_allow_html=True)
        if st.button("Check Lungs"):
            st.switch_page("pages/3_🫁_Pneumonia_Check.py")

# Row 2
col4, col5, col6 = st.columns(3)

with col4:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=60)
        st.markdown("<h3>Heart</h3>", unsafe_allow_html=True)
        st.markdown("<p>Assess cardiovascular risk.</p>", unsafe_allow_html=True)
        if st.button("Check Heart"):
            st.switch_page("pages/5_❤️_Heart_Risk.py")

with col5:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=60)
        st.markdown("<h3>Malaria</h3>", unsafe_allow_html=True)
        st.markdown("<p>Analyze cell images.</p>", unsafe_allow_html=True)
        if st.button("Check Cells"):
            st.switch_page("pages/4_🦟_Malaria_Check.py")

with col6:
    st.write("") # Empty column
