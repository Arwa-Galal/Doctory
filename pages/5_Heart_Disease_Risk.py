import streamlit as st
import pandas as pd
from utils import MODELS

# --- Helper Functions for Feature Preparation ---
def calculate_bmi(height_cm, weight_kg):
    if height_cm == 0: return 0
    return weight_kg / ((height_cm / 100) ** 2)

def get_age_category(age):
    age = int(age)
    if 18 <= age <= 24: return 'Young'
    if 25 <= age <= 39: return 'Adult'
    if 40 <= age <= 54: return 'Mid-Aged'
    if 55 <= age <= 64: return 'Senior-Adult'
    if age >= 65: return 'Elderly'
    return 'Adult'

def prepare_heart_features(data):
    # Scaler
    scaler = MODELS['heart_scaler']
    
    # Inputs
    height = data.get('Height')
    weight = data.get('Weight')
    age = data.get('Age')
    bmi = calculate_bmi(height, weight)
    
    # Mappings
    general_health_map = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Poor': 3, 'Very Good': 4}
    checkup_map = {'More than 5 years': 0, 'Never': 1, 'Past 1 year': 2, 'Past 2 years': 3, 'Past 5 years': 4}
    binary_map = {'No': 0, 'Yes': 1} 
    diabetes_map = {'No': 0, 'No Pre Diabetes': 1, 'Only during pregnancy': 2, 'Yes': 3}
    age_category_map = {'Adult': 0, 'Elderly': 1, 'Mid-Aged': 2, 'Senior-Adult': 3, 'Young': 4}
    bmi_group_map = {'Normal weight': 0, 'Obese I': 1, 'Obese II': 2, 'Overweight': 3, 'Underweight': 4}

    # BMI Group Calculation
    bmi_bins = [12.02, 18.3, 26.85, 31.58, 37.8, 100]
    bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese I', 'Obese II']
    try:
        bmi_group_str = pd.cut(
