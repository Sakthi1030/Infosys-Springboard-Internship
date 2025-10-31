import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Load model
model = joblib.load('heart_disease_model.pkl')

# Page setup
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
    }
    .sub {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
    }
    .popup {
        position: fixed;
        top: 35%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: white;
        border: 3px solid #FF4B4B;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: fadeIn 0.5s ease-in-out;
        z-index: 1000;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translate(-50%, -60%);}
        to {opacity: 1; transform: translate(-50%, -50%);}
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)

# Input layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=56)
    sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type", [("Typical Angina", 0), ("Atypical Angina", 1),
                                          ("Non-anginal Pain", 2), ("Asymptomatic", 3)], format_func=lambda x: x[0])[1]
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 236)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [("True", 1), ("False", 0)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox("Resting ECG Results", [("Normal", 0), ("ST-T Abnormality", 1),
                                                   ("Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0])[1]
    thalach = st.number_input("Max Heart Rate", 70, 210, 178)
    exang = st.selectbox("Exercise-Induced Angina", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.5, 0.8)

with col3:
    slope = st.selectbox("Slope of ST Segment", [("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])[1]
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [("Normal", 0), ("Fixed Defect", 1), ("Reversible Defect", 2)], format_func=lambda x: x[0])[1]

# Prediction
if st.button("Predict My Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]], dtype=float)
    
    with st.spinner("Analyzing your data..."):
        time.sleep(1.5)
        prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        html_msg = """
        <div class='popup' style='background-color:#ffe6e6; color:#d40000;'>
        ⚠️ High Risk of Heart Disease detected! <br><br>
        Please consult a doctor immediately.
        </div>
        """
    else:
        html_msg = """
        <div class='popup' style='background-color:#e6ffed; color:#006600;'>
        🎉 Great News!<br><br>
        You are likely healthy. Keep up regular checkups and a balanced lifestyle.
        </div>
        """

    st.markdown(html_msg, unsafe_allow_html=True)

st.markdown("<br><hr><center>© 2025 Heart Disease Predictor | Built with Streamlit ❤️</center>", unsafe_allow_html=True)
