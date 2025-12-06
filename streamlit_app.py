import streamlit as st
from utils import load_css ,load_all_models, MODELS

st.set_page_config(page_title="Doctory", page_icon="🩺", layout="wide")
load_css()

# --- HERO SECTION ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://www.shutterstock.com/image-vector/blue-heart-pulse-monitor-signal-260nw-229134526.jpg", width=120)
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Doctory AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>Your Intelligent Medical Companion</h4>", unsafe_allow_html=True)
    st.markdown("---")

# --- NAVIGATION ICONS ---
st.markdown("<h3 style='text-align: center;'>Choose a Service</h3>", unsafe_allow_html=True)
st.write("") # Spacer

# Row 1: Chat, Diabetes, Pneumonia
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=60)
    st.subheader("AI Doctor")
    st.caption("Chat with our smart assistant.")
    if st.button("Start Chat"):
        # تأكدي أن اسم الملف في فولدر pages هو 1_💬_Chat_With_Doctory.py
        st.switch_page("pages/1_AI_Chatbot.py")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865769.png", width=60)
    st.subheader("Diabetes")
    st.caption("Check risk based on vitals.")
    if st.button("Check Diabetes"):
        # تأكدي أن اسم الملف 2_🩸_Diabetes_Test.py
        st.switch_page("pages/4_Diabetes_Risk.py")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=60)
    st.subheader("Pneumonia")
    st.caption("Analyze Chest X-Ray images.")
    if st.button("Check Lungs"):
        # تأكدي أن اسم الملف 3_🫁_Pneumonia_Check.py
        st.switch_page("pages/2_Pneumonia_X_Ray.py")
    st.markdown('</div>', unsafe_allow_html=True)

# Row 2: Heart, Malaria (تم الإصلاح هنا)
c4, c5, c6 = st.columns([1, 1, 1]) # أضفت العمود الثالث الفارغ

with c4:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=60)
    st.subheader("Heart Disease")
    st.caption("Assess cardiovascular risk.")
    if st.button("Check Heart"):
        # تأكدي أن اسم الملف 5_❤️_Heart_Risk.py
        st.switch_page("pages/5_Heart_Disease_Risk.py")
    st.markdown('</div>', unsafe_allow_html=True)

with c5:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=60)
    st.subheader("Malaria")
    st.caption("Analyze cell images.")
    if st.button("Check Malaria"):
        # تأكدي أن اسم الملف 4_🦟_Malaria_Check.py
        st.switch_page("pages/3_Malaria_Blood_Smear.py")
    st.markdown('</div>', unsafe_allow_html=True)

# العمود الثالث فارغ للتوازن
with c6:
    st.write("")
