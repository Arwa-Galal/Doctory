import streamlit as st
import google.generativeai as genai
import joblib
import numpy as np
import os
from PIL import Image

# --- 1. إعداد الصفحة ---
st.set_page_config(
    page_title="Doctory AI",
    page_icon="🩺",
    layout="wide"
)

# --- 2. إعداد مفتاح جوجل ---
# ضعي مفتاحك هنا (تأكدي أنه يبدأ بـ AIza)
GOOGE_API_KEY = "AIzaSyCGlprvtIdX7vTQCPBGi7dv4FcQ4usEpdI" 

# إعداد الاتصال بـ Gemini
try:
    genai.configure(api_key=GOOGE_API_KEY)
    
    # === التعديل هنا: استخدمنا الاسم الجديد للموديل ===
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
    
except Exception as e:
    st.error(f"خطأ في إعداد المفتاح: {e}")

# --- 3. تحميل موديلات الأمراض ---
@st.cache_resource
def load_models():
    models = {}
    try:
        import xgboost # استدعاء المكتبة عشان joblib يشوفها
        # تأكدي من مسار الملف عندك
        models['diabetes'] = joblib.load('models/diabetes_model_package/diabetes_ensemble_model.joblib')
    except Exception as e:
        # لو فيه خطأ مش هنوقف الموقع، بس هنطبع تحذير
        pass 
    return models

loaded_models = load_models()

# --- 4. القائمة الجانبية ---
with st.sidebar:
    st.title("🩺 قائمة دكتوري")
    choice = st.radio(
        "اختر الخدمة:", 
        ["💬 التحدث مع الطبيب الذكي", "🩸 فحص السكري", "🫁 فحص الرئة"]
    )
    st.markdown("---")
    st.warning("⚠️ تنبيه: هذا تطبيق مساعد ولا يغني عن الطبيب الحقيقي.")

# --- 5. الصفحات ---

# === الصفحة 1: الشات (الدكتور الذكي) ===
if choice == "💬 التحدث مع الطبيب الذكي":
    st.title("💬 عيادة دكتوري الذكية")
    st.caption("أنا هنا للإجابة على استفساراتك الطبية العامة...")

    # حفظ المحادثة عشان متتمسحش
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "أهلاً بك! كيف يمكنني مساعدتك صحياً اليوم؟"}]

    # عرض الرسائل القديمة
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # استقبال سؤال جديد
    if prompt := st.chat_input("اكتب شكواك أو سؤالك هنا..."):
        # عرض سؤال المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # استقبال الرد من Gemini
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                try:
                    full_prompt = f"تصرف كطبيب محترف ومتعاطف. أجب على هذا السؤال الطبي باختصار وفائدة: {prompt}"
                    response = model_ai.generate_content(full_prompt)
                    ai_text = response.text
                    
                    st.write(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                except Exception as e:
                    st.error(f"حدث خطأ في الاتصال: {e}")

# === الصفحة 2: فحص السكري ===
elif choice == "🩸 فحص السكري":
    st.title("🩸 تحليل مخاطر السكري")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("العمر (Age)", 1, 120, 30)
        pregnancies = st.number_input("عدد مرات الحمل", 0, 20, 0)
        glucose = st.number_input("مستوى الجلوكوز", 0, 500, 100)
    with col2:
        bp = st.number_input("ضغط الدم (BP)", 0, 200, 70)
        skin = st.number_input("سمك الجلد (Skin)", 0, 100, 20)
        insulin = st.number_input("الأنسولين", 0, 900, 79)
    with col3:
        bmi = st.number_input("مؤشر الكتلة (BMI)", 0.0, 70.0, 25.0)
        dpf = st.number_input("تاريخ العائلة (DPF)", 0.0, 3.0, 0.5)

    if st.button("تحليل النتيجة"):
        if 'diabetes' in loaded_models:
            # تجهيز البيانات
            input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            
            try:
                # 1. الموديل يحسب النتيجة
                prediction = loaded_models['diabetes'].predict(input_data)[0]
                result_str = "Diabetic (مصاب محتمل)" if prediction == 1 else "Healthy (سليم)"
                color = "red" if prediction == 1 else "green"
                
                # 2. الـ AI يشرح النتيجة
                prompt_analysis = f"بيانات المريض: سكر {glucose}، عمر {age}. نتيجة الموديل: {result_str}. اشرح النتيجة للمريض باختصار."
                explanation = model_ai.generate_content(prompt_analysis).text

                # 3. العرض
                st.markdown(f"### النتيجة: :{color}[{result_str}]")
                st.info(f"👨‍⚕️ رأي المستشار الطبي: {explanation}")
                
            except Exception as e:
                st.error(f"خطأ في الحساب: {e}")
        else:
            st
