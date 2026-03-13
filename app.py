import streamlit as st
import numpy as np
import pickle
import os

# Page settings
st.set_page_config(page_title="AI Dropout Prediction", page_icon="🎓")

st.title("🎓 AI Based Student Dropout Prediction System")
st.write("Enter student details to predict dropout risk and provide counseling support.")

# Model file path
model_path = "AI_based_dropout_system.pkl"

# Load model safely
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error("❌ Error loading model file.")
        st.stop()
else:
    st.error("❌ Model file not found. Please upload AI_based_dropout_system.pkl")
    st.stop()

# Input section
st.subheader("📊 Enter Student Details")

attendance = st.slider("📊 Attendance (%)", 0, 100, 60)
marks = st.slider("📑 Marks", 0, 100, 60)
behavior = st.slider("🙂 Behavior (1-5)", 1, 5, 3)
participation = st.slider("🙋 Participation (1-5)", 1, 5, 3)

# Predict button
if st.button("🔍 Predict Dropout Risk"):

    try:
        features = np.array([[attendance, marks, behavior, participation]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            st.error("⚠️ High Dropout Risk Detected")
            st.warning("👨‍🏫 Counseling Support Recommended")
        else:
            st.success("✅ Student is Safe – No Dropout Risk")

    except Exception as e:
        st.error("❌ Prediction failed. Please check the model file.")

st.markdown("---")
st.markdown("🤖 Model: Logistic Regression")
st.markdown("📊 Features Used: Attendance, Marks, Behavior, Participation")
