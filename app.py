import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("AI_based_dropout_system.pkl", "rb"))

st.title("🎓 AI Based Student Dropout Prediction System 🤖")

st.write("Enter student details to predict dropout risk.")

attendance = st.number_input("📊 Attendance (%)", 0, 100)
marks = st.number_input("📑 Marks", 0, 100)
behavior = st.number_input("🙂 Behavior (1-5)", 1, 5)
participation = st.number_input("🙋 Participation (1-5)", 1, 5)

if st.button("🔍 Predict Dropout Risk"):

    features = np.array([[attendance, marks, behavior, participation]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ High Dropout Risk – Counseling Recommended 👨‍🏫")
    else:
        st.success("✅ Student is Safe – No Dropout Risk 🎓")
