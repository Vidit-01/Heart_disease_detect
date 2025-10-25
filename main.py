import streamlit as st
import numpy as np
import joblib

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details below:")
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/model.pkl')

# Inputs
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type", (0, 1, 2, 3), format_func=lambda x: f"Type {x}")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=700)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("True", "False"))
restecg = st.selectbox("Resting ECG Results", (0, 1, 2))
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250)
exang = st.selectbox("Exercise Induced Angina", ("Yes", "No"))
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", (0, 1, 2))
ca = st.selectbox("Number of Major Vessels (0–3)", (0, 1, 2, 3))
thal = st.selectbox("Thal", (0, 1, 2), format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

# Predict button
if st.button("Predict"):
    # Convert categorical to numeric
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "True" else 0
    exang = 1 if exang == "Yes" else 0

    # Combine all features into numpy array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])

    st.write("### Input Summary")
    st.write(features)

    user_input_scaled = scaler.transform(features)

    prediction = model.predict(user_input_scaled)[0]
    proba = model.predict_proba(user_input_scaled)[0][1]  # Probability of heart disease

    # Define confidence and risk level
    confidence = round(proba * 100, 2)

    if proba > 0.75:
        st.markdown(f"<h3 style='color:red;'>High Risk ({confidence}% confidence)</h3>", unsafe_allow_html=True)
    elif proba > 0.4:
        st.markdown(f"<h3 style='color:orange;'>Moderate Risk ({confidence}% confidence)</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:green;'>Low Risk ({confidence}% confidence)</h3>", unsafe_allow_html=True)
