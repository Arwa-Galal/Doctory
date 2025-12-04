import streamlit as st
import google.generativeai as genai
import joblib
import numpy as np
import os
from PIL import Image

# --- 1. إعداد الصفحة ---
st.set_page_config(
    page_title="طبيبي الذكي",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. مفتاح جوجل (ضعي مفتاحك هنا) ---
# تأكدي أن المفتاح يبدأ بـ AIza
GOOGE_API_KEY = "AIzaSyCGlprvtIdX7vTQCPBGi7dv4FcQ4usEpdI" 

# إعداد الاتصال
try:
    genai.configure(api_key=GOOGE_API_KEY)
    # سنستخدم gemini-pro لأنه الأكثر استقراراً حالياً
    model_ai = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"خطأ في إعداد المفتاح: {e}")

# --- 3. تحميل موديلات الأمراض ---
@st.cache_resource
def load_models():
    models = {}
    try:
        import xgboost
        # تأكدي أن مسار الملف صحيح لديك في GitHub
        models['diabetes'] = joblib.load('models/diabetes_model_package/diabetes_ensemble_model.joblib')
    except Exception as e:
        print(f"Error loading models: {e}") 
    return models

loaded_models = load_models()

# --- 4. القائمة الجانبية (عربي) ---
with st.sidebar:
    st.title("🩺 قائمة الخدمات")
    choice = st.radio(
        "اختر الخدمة:", 
        ["💬 التحدث مع الطبيب", "🩸 فحص السكري", "🫁 فحص الرئة"]
    )
    st.markdown("---")
    st.warning("⚠️ تنبيه: هذا تطبيق ذكاء اصطناعي للمساعدة فقط ولا يغني عن الطبيب.")

# --- 5. الصفحات (عربي) ---

# === الصفحة 1: الشات (الصفحة الرئيسية) ===
if choice == "💬 التحدث مع الطبيب":
    st.title("💬 عيادة طبيبي الذكية")
    st.caption("أهلاً بك.. أنا هنا للإجابة على استفساراتك الطبية العامة.")

    # تهيئة سجل المحادثة
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "أهلاً بك. مم تشكو اليوم؟"}]

    # عرض الرسائل السابقة
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # استقبال الرسالة الجديدة
    if prompt := st.chat_input("اكتب سؤالك هنا..."):
        # عرض سؤال المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # الرد من الذكاء الاصطناعي
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                try:
                    # الأمر الموجه للذكاء الاصطناعي ليتحدث بالعربية
                    full_prompt = f"تصرف كطبيب محترف. أجب على هذا السؤال باللغة العربية: {prompt}"
                    response = model_ai.generate_content(full_prompt)
                    ai_text = response.text
                    
                    st.write(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                except Exception as e:
                    st.error("عذراً، حدث خطأ في الاتصال. تأكد من مفتاح جوجل.")
                    st.error(f"تفاصيل الخطأ: {e}")

# === الصفحة 2: فحص السكري ===
elif choice == "🩸 فحص السكري":
    st.title("🩸 تحليل مخاطر السكري")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("العمر", 1, 120, 30)
        pregnancies = st.number_input("عدد مرات الحمل", 0, 20, 0)
        glucose = st.number_input("مستوى السكر (Glucose)", 0, 500, 100)
    with col2:
        bp = st.number_input("ضغط الدم", 0, 200, 70)
        skin = st.number_input("سمك الجلد", 0, 100, 20)
        insulin = st.number_input("الأنسولين", 0, 900, 79)
    with col3:
        bmi = st.number_input("مؤشر كتلة الجسم (BMI)", 0.0, 70.0, 25.0)
        dpf = st.number_input("تاريخ العائلة (DPF)", 0.0, 3.0, 0.5)

    if st.button("تحليل النتيجة"):
        if 'diabetes' in loaded_models:
            input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            try:
                prediction = loaded_models['diabetes'].predict(input_data)[0]
                
                # ترجمة النتيجة
                result_str = "مصاب محتمل (Diabetic)" if prediction == 1 else "سليم (Healthy)"
                color = "red" if prediction == 1 else "green"
                
                # طلب الشرح من الـ AI
                prompt_analysis = f"مريض سكر (جلوكوز: {glucose})، عمره {age}. نتيجة الموديل تقول: {result_str}. اشرح له النتيجة بالعربية وقدم نصيحة."
                explanation = model_ai.generate_content(prompt_analysis).text

                st.markdown(f"### النتيجة: :{color}[{result_str}]")
                st.info(f"👨‍⚕️ رأي المستشار الطبي: {explanation}")
                
            except Exception as e:
                st.error(f"حدث خطأ في الحساب: {e}")
        else:
            st.error("عذراً، ملف موديل السكري غير موجود في المسار الصحيح.")

# === الصفحة 3: فحص الرئة ===
elif choice == "🫁 فحص الرئة":
    st.title("🫁 فحص الأشعة (X-Ray)")
    uploaded_file = st.file_uploader("ارفع صورة الأشعة هنا", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, width=300)
        st.info("الذكاء الاصطناعي جاهز لتحليل الصورة (يحتاج ربط موديل الصور).")
