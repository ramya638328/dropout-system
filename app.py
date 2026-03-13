import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="AI Dropout Prediction", page_icon="🎓")

st.title("🎓 AI Based Student Dropout Prediction System")
st.write("Predict student dropout risk and suggest counseling support.")

model_path = "student_dropout_system.pkl"

# Load model
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error("❌ Model file not found. Please upload student_dropout_system.pkl")
    st.stop()

st.subheader("📊 Enter Student Details")

attendance = st.slider("Attendance (%)", 0, 100, 50)
marks = st.slider("Marks", 0, 100, 50)
behavior = st.slider("Behavior (1-5)", 1, 5, 3)
participation = st.slider("Participation (1-5)", 1, 5, 3)

if st.button("🔍 Predict Dropout Risk"):

    features = np.array([[attendance, marks, behavior, participation]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ High Dropout Risk – Counseling Recommended 👨‍🏫")
    else:
        st.success("✅ Student is Safe – No Dropout Risk 🎓")
