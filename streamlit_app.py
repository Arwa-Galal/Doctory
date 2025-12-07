import streamlit as st
from utils import load_css, render_sidebar

st.set_page_config(page_title="Doctory", page_icon="🩺", layout="wide")
load_css()
render_sidebar("Home") # تحديد الصفحة الحالية

# --- HERO SECTION ---
# نضعها داخل كارت أيضاً لتوحيد الشكل
with st.container(border=True):import streamlit as st
from utils import load_css, render_sidebar

st.set_page_config(page_title="Doctory", page_icon="🩺", layout="wide")
load_css()
render_sidebar("Home")

# --- HERO SECTION ---
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=120)
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Doctory AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your Intelligent Medical Companion</p>", unsafe_allow_html=True)

st.write("") 
st.markdown("<h3 style='text-align: center;'>Choose a Service</h3>", unsafe_allow_html=True)
st.write("") 

# --- SERVICE CARDS (Using Buttons as Cards) ---
# ملاحظة: العناوين داخل الزر نستخدم فيها \n للنزول سطر

col1, col2 = st.columns(2)

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=70)
    # الزر الآن هو الكارت الأبيض
    if st.button("AI Doctor\nChat with our smart assistant", key="btn_chat"):
        st.switch_page("pages/1_AI_Chatbot.py")

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865769.png", width=70)
    if st.button("Diabetes\nCheck risk based on vitals", key="btn_dia"):
        st.switch_page("pages/4_Diabetes_Risk.py")

st.write("") # فاصل

col3, col4 = st.columns(2)

with col3:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=70)
    if st.button("Pneumonia\nAnalyze Chest X-Ray images", key="btn_pneu"):
        st.switch_page("pages/2_Pneumonia_X_Ray.py")

with col4:
    st.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=70)
    if st.button("Malaria\nAnalyze cell images", key="btn_mal"):
        st.switch_page("pages/3_Malaria_Blood_Smear.py")

st.write("") # فاصل

col5, col6, col7 = st.columns([1, 2, 1])
with col6:
    st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=70)
    if st.button("Heart Disease\nAssess cardiovascular risk", key="btn_heart"):
        st.switch_page("pages/5_❤️_Heart_Risk.py")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=120)
        st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Doctory AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Your Intelligent Medical Companion</p>", unsafe_allow_html=True)

st.write("") 
st.markdown("<h3 style='text-align: center;'>Choose a Service</h3>", unsafe_allow_html=True)
st.write("") 

# --- SERVICE CARDS ---
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=60)
        st.markdown("<h3>AI Doctor</h3>", unsafe_allow_html=True)
        st.markdown("<p>Chat with our smart assistant.</p>", unsafe_allow_html=True)
        # الزر هو الوسيلة الوحيدة للتفاعل
        if st.button("Start Chat"):
            st.switch_page("pages/1_AI_Chatbot.py")

with col2:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/2865/2865769.png", width=60)
        st.markdown("<h3>Diabetes</h3>", unsafe_allow_html=True)
        st.markdown("<p>Check risk based on vitals.</p>", unsafe_allow_html=True)
        if st.button("Check Risk"):
            st.switch_page("pages/4_Diabetes_Risk.py")

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

# القلب (منفرد)
col5, col6, col7 = st.columns([1, 2, 1])
with col6:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=60)
        st.markdown("<h3>Heart Disease</h3>", unsafe_allow_html=True)
        st.markdown("<p>Assess cardiovascular risk.</p>", unsafe_allow_html=True)
        if st.button("Check Heart"):
            st.switch_page("pages/5_❤️_Heart_Risk.py")
