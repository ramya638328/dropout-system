import streamlit as st
import numpy as np
import pickle
import os

st.title("🎓 AI Student Dropout Prediction System")

model_path = "dropout_system.pkl"

# Check model file
if not os.path.exists(model_path):
    st.error("❌ Model file not found. Upload dropout_system.pkl")
    st.stop()

# Load model safely
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("⚠️ Model loading failed. Check requirements.txt or model file.")
    st.stop()

st.subheader("Enter Student Details")

attendance = st.slider("Attendance",0,100,50)
marks = st.slider("Marks",0,100,50)
behavior = st.slider("Behavior (1-5)",1,5,3)
participation = st.slider("Participation (1-5)",1,5,3)

if st.button("Predict"):

    features = np.array([[attendance,marks,behavior,participation]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ Student at Risk – Counseling Recommended")
    else:
        st.success("✅ Student Safe")
