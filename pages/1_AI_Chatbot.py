import streamlit as st
import google.generativeai as genai
import os

# 1. إعداد الشكل (لازم يكون أول سطر)
st.set_page_config(page_title="Doctory", page_icon="🩺", layout="centered")

# 2. إعداد Gemini (حطي مفتاحك هنا)
genai.configure(api_key="AIzaSyAg-7Wu_mCF-z9P-KEbkjpQEb7B3PB_hxo")
model_ai = genai.GenerativeModel('gemini-pro')

# 3. كود CSS عشان الشكل يبقى حلو (بلاش نغير الـ Layout، هنغير الألوان بس)
st.markdown("""
    <style>
    .stApp {background-color: #f0f8ff;} /* لون خلفية هادي */
    .stButton>button {
        background-color: #00ADB5; color: white; border-radius: 10px; width: 100%;
    }
    .result-box {
        padding: 20px; background-color: white; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #00ADB5;
    }
    </style>
""", unsafe_allow_html=True)
