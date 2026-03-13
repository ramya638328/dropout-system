import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="AI Dropout Prediction", page_icon="🎓")

st.title("🎓 AI Based Student Dropout Prediction System")

st.write("This system predicts whether a student is at risk of dropping out and suggests counseling support.")

# Load Model
try:
    model = pickle.load(open("dropout.pkl","rb"))
except:
    st.error("❌ Model loading failed. Make sure dropout_system.pkl is uploaded.")
    st.stop()

st.subheader("📊 Enter Student Information")

attendance = st.slider("Attendance (%)",0,100,50)

marks = st.slider("Marks (%)",0,100,50)

behavior = st.slider("Behavior Score (1-5)",1,5,3)

participation = st.slider("Participation Score (1-5)",1,5,3)

if st.button("🔍 Predict Dropout Risk"):

    features = np.array([[attendance,marks,behavior,participation]])

    prediction = model.predict(features)

    if prediction[0] == 1:

        st.error("⚠️ High Dropout Risk Detected")

        st.write("👨‍🏫 Counseling Recommended")

    else:

        st.success("✅ Student is Safe")

        st.write("🎓 Continue Monitoring Performance")
