import streamlit as st
import requests

st.title("Diabetes Risk Demo")
Age = st.number_input("Age", 0, 120, 40)
BMI = st.number_input("BMI", 0.0, 100.0, 25.0)
BloodPressure = st.number_input("BloodPressure", 0.0, 200.0, 80.0)
Insulin = st.number_input("Insulin", 0.0, 1000.0, 80.0)
Glucose = st.number_input("Glucose", 0.0, 500.0, 100.0)
DPF = st.number_input("DiabetesPedigreeFunction", 0.0, 10.0, 0.5)
high_bp = st.selectbox("High BP (1=yes,0=no)", [0,1], index=0)
Gender_Male = st.selectbox("Gender Male (1=yes,0=no)", [1,0], index=0)

if st.button("Predict"):
    payload = {
        "Age": Age, "BMI": BMI, "BloodPressure": BloodPressure,
        "Insulin": Insulin, "Glucose": Glucose,
        "DiabetesPedigreeFunction": DPF, "high_bp": high_bp, "Gender_Male": Gender_Male
    }
    try:
        r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=5)
        st.json(r.json())
    except Exception as e:
        st.error(f"Request failed: {e}")
