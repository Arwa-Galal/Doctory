import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import google.generativeai as genai
import requests
import joblib
import numpy as np
from PIL import Image
import os

# --- 1. إعداد الصفحة وتجهيز Gemini ---
st.set_page_config(
    page_title="Doctory AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# مفتاح API (يفضل وضعه في st.secrets عند الرفع)
# os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"] 
# للتجربة المحلية ضعي المفتاح هنا:
genai.configure(api_key="AIzaSyAg-7Wu_mCF-z9P-KEbkjpQEb7B3PB_hxo") 
model_ai = genai.GenerativeModel('gemini-pro')

# --- 2. دوال مساعدة (CSS & Lottie) ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

def local_css():
    st.markdown("""
    <style>
        /* إخفاء عناصر ستريم ليت الافتراضية */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* تنسيق الخطوط */
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Cairo', sans-serif;
        }
        
        /* تنسيق الكروت */
        .css-card {
            border-radius: 15px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-right: 5px solid #00ADB5;
        }
        
        /* تنسيق النتائج */
        .result-title { font-size: 24px; font-weight: bold; color: #222831; }
        .result-val { font-size: 20px; color: #00ADB5; }
        .ai-box { background-color: #e0f7fa; padding: 15px; border-radius: 10px; border: 1px dashed #00ADB5; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. دالة الطبيب الذكي (Gemini) ---
def get_ai_advice(condition, result, patient_info):
    prompt = f"""
    تصرف كطبيب استشاري رحيم. 
    المريض قام بفحص {condition}. النتيجة: {result}.
    بيانات المريض: {patient_info}.
    
    المطلوب:
    1. طمأنة المريض وشرح النتيجة ببساطة (بالعربية).
    2. تقديم 3 نصائح طبية وعادات صحية مناسبة لهذه الحالة.
    3. إذا كانت النتيجة إيجابية، انصحه بالخطوة التالية (تحاليل أو زيارة طبيب).
    """
    try:
        response = model_ai.generate_content(prompt)
        return response.text
    except:
        return "عذراً، خدمة المستشار الطبي غير متاحة حالياً."

# --- 4. تحميل الموديلات (Load Models) ---
# ملاحظة: تأكدي من مسارات الملفات عندك
@st.cache_resource
def load_models():
    models = {}
    try:
        # مثال لتحميل موديل السكر
        models['diabetes'] = joblib.load('models/diabetes_model_package/diabetes_ensemble_model.joblib')
        # models['pneumonia'] = ... (تحميل موديل الـ ONNX هنا)
    except Exception as e:
        st.error(f"خطأ في تحميل الموديلات: {e}")
    return models

models = load_models()

# --- 5. القائمة الجانبية (Sidebar) ---
with st.sidebar:
    # يمكنك وضع اللوجو هنا
    # st.image("assets/logo.png", width=200)
    selected = option_menu(
        "Doctory AI",
        ["الرئيسية", "فحص السكري", "فحص الرئة", "الملاريا", "عن المشروع"],
        icons=['house', 'activity', 'lungs', 'virus', 'info-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "nav-link-selected": {"background-color": "#00ADB5"},
        }
    )

# --- 6. الصفحات ---

# === الصفحة الرئيسية ===
if selected == "الرئيسية":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.title("مرحباً بك في Doctory 👋")
        st.markdown("""
        <div class="css-card">
            <h3>نظام التشخيص الطبي الذكي</h3>
            <p>نستخدم أحدث تقنيات الذكاء الاصطناعي لمساعدتك في الاطمئنان على صحتك.</p>
            <ul>
                <li>تحليل فوري للبيانات.</li>
                <li>تفسير طبي للنتائج.</li>
                <li>خصوصية وأمان لبياناتك.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_5njp3vgg.json"
        lottie_json = load_lottieurl(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=300)

# === فحص السكري ===
elif selected == "فحص السكري":
    st.title("🩸 فحص السكري (Diabetes)")
    
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("العمر", min_value=1, value=30)
            pregnancies = st.number_input("عدد مرات الحمل", min_value=0, value=0)
        with col2:
            glucose = st.number_input("مستوى الجلوكوز", min_value=0, value=100)
            bp = st.number_input("ضغط الدم", min_value=0, value=70)
        with col3:
            bmi = st.number_input("مؤشر كتلة الجسم (BMI)", min_value=0.0, value=25.0)
            pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
            
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("تحليل البيانات 🔍", type="primary", use_container_width=True):
        # 1. تجهيز الداتا
        input_data = np.array([[pregnancies, glucose, bp, 0, 0, bmi, pedigree, age]]) # تأكدي من ترتيب الـ Features حسب تدريب الموديل
        
        # 2. التوقع (Prediction)
        if 'diabetes' in models:
            prediction = models['diabetes'].predict(input_data)[0] # 0 or 1
            result_text = "مصاب بالسكري (Diabetic)" if prediction == 1 else "سليم (Healthy)"
            color = "red" if prediction == 1 else "green"
            
            # 3. استشارة Gemini
            with st.spinner("جاري استشارة الطبيب الذكي..."):
                ai_reply = get_ai_advice(
                    "مرض السكري", 
                    result_text, 
                    f"العمر: {age}, السكر: {glucose}, BMI: {bmi}"
                )
            
            # 4. عرض النتيجة
            st.markdown(f"""
            <div class="css-card" style="border-right: 5px solid {color};">
                <h3 class="result-title">النتيجة: <span style="color:{color}">{result_text}</span></h3>
                <hr>
                <div class="ai-box">
                    <h4>👨‍⚕️ رأي المستشار الطبي:</h4>
                    <p>{ai_reply}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("عذراً، موديل السكري لم يتم تحميله بشكل صحيح.")

# === فحص الرئة (صور) ===
elif selected == "فحص الرئة":
    st.title("🫁 فحص الالتهاب الرئوي (Pneumonia)")
    
    uploaded_file = st.file_uploader("ارفع صورة الأشعة (X-Ray)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="الصورة المرفوعة", width=300)
        
        if st.button("فحص الأشعة 🔍", type="primary"):
            # هنا تضعي كود معالجة الصورة واستخدام موديل الـ ONNX
            # image = process_image(uploaded_file)
            # pred = onnx_session.run(...)
            
            # (سنفترض نتيجة للتجربة)
            fake_result = "Normal (سليم)" 
            
            with st.spinner("جاري تحليل الصورة..."):
                ai_reply = get_ai_advice("التهاب رئوي", fake_result, "لا توجد أعراض أخرى مسجلة")
            
            st.markdown(f"""
            <div class="css-card">
                <h3 class="result-title">النتيجة: {fake_result}</h3>
                <div class="ai-box">
                    <p>{ai_reply}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# === عن المشروع ===
elif selected == "عن المشروع":
    st.markdown("""
    <div class="css-card">
        <h2>عن Doctory</h2>
        <p>مشروع تخرج يهدف إلى تسخير الذكاء الاصطناعي لخدمة القطاع الطبي.</p>
        <p><strong>فريق العمل:</strong> أروى جلال ومجموعتها المتميزة.</p>
    </div>
    """, unsafe_allow_html=True)
