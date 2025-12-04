import streamlit as st
import google.generativeai as genai
import joblib
import numpy as np
import os

# --- 1. الإعدادات ---
st.set_page_config(page_title="Doctory Debugger", layout="wide")

# ضعي مفتاحك هنا (تأكدي أنه يبدأ بـ AIza)
GOOGE_API_KEY = "AIzaSyCGlprvtIdX7vTQCPBGi7dv4FcQ4usEpdI" 

# --- 2. محاولة الاتصال بـ Gemini (مع كشف السبب الحقيقي) ---
try:
    genai.configure(api_key=GOOGE_API_KEY)
    model_ai = genai.GenerativeModel('gemini-pro')
    
    # تجربة سريعة للتأكد من الاتصال
    response = model_ai.generate_content("Hello")
    st.success("✅ تم الاتصال بـ Google Gemini بنجاح!")
    
except Exception as e:
    # هنا سيظهر السبب الحقيقي للمشكلة
    st.error(f"❌ خطأ في مفتاح جوجل: {e}")
    st.warning("تأكدي أنك نسختي المفتاح بالكامل ولم تتركي مسافات فارغة.")

# --- 3. بقية التطبيق ---
# (لن يعمل إلا إذا نجح الاتصال فوق)

with st.sidebar:
    st.title("Doctory Menu")
    choice = st.radio("Choose:", ["Chatbot", "Diabetes Test"])

if choice == "Chatbot":
    st.title("💬 Chatbot Test")
    user_input = st.text_input("Say something:")
    if user_input:
        try:
            reply = model_ai.generate_content(user_input)
            st.write(reply.text)
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Diabetes Test":
    st.write("Diabetes Model Test Area")
    # ... باقي كود الموديلات القديم ...
