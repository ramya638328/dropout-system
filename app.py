import streamlit as st
import numpy as np
import pickle
import os

# App title
st.set_page_config(page_title="AI Dropout Prediction", page_icon="🎓")

st.title("🎓 AI-Based Student Dropout Prediction System")
st.write("This system predicts whether a student is at risk of dropping out and suggests counseling support.")

# Load the trained model safely
model_path = "AI_based_dropout_system.pkl"

if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
else:
    st.error("❌ Model file not found. Please upload the .pkl file in the project folder.")
    st.stop()

st.subheader("📊 Enter Student Details")

# Input fields
attendance = st.slider("📊 Attendance (%)", 0, 100, 50)
marks = st.slider("📑 Marks", 0, 100, 50)
behavior = st.slider("🙂 Behavior (1-5)", 1, 5, 3)
participation = st.slider("🙋 Participation (1-5)", 1, 5, 3)

# Prediction button
if st.button("🔍 Predict Dropout Risk"):

    features = np.array([[attendance, marks, behavior, participation]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ High Dropout Risk Detected!")
        st.warning("👨‍🏫 Counseling Recommended to Support the Student.")
    else:
        st.success("✅ Student is Safe. No Dropout Risk.")

st.markdown("---")
st.markdown("🤖 **AI Model:** Logistic Regression")
st.markdown("📊 **Input Features:** Attendance, Marks, Behavior, Participation")
st.markdown("🎓 Developed for Student Support and Early Counseling")
