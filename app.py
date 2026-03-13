import streamlit as st
import numpy as np
import pickle
import os

# Page settings
st.set_page_config(page_title="Student Dropout Prediction", page_icon="🎓")

st.title("🎓 AI Based Student Dropout Prediction System")
st.write("This system predicts whether a student is at risk of dropping out.")

# Model file path
model_path = "dropout_system.pkl"

# Check if model exists
if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please upload dropout_system.pkl")
    st.stop()

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.subheader("📊 Enter Student Information")

# Input fields
attendance = st.slider("Attendance (%)", 0, 100, 50)
marks = st.slider("Marks (%)", 0, 100, 50)
behavior = st.slider("Behavior Score (1-5)", 1, 5, 3)
participation = st.slider("Participation Score (1-5)", 1, 5, 3)

# Predict button
if st.button("🔍 Predict Dropout Risk"):

    features = np.array([[attendance, marks, behavior, participation]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ Student is at risk of dropping out. Counseling recommended 👨‍🏫")
    else:
        st.success("✅ Student is performing well. No dropout risk 🎉")
