import streamlit as st
import numpy as np
import pickle

# Page settings
st.set_page_config(
    page_title="AI Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# Title
st.markdown(
"""
<h1 style='text-align: center; color:#4CAF50;'>
🎓 AI Based Student Dropout Prediction & Counseling System
</h1>
""",
unsafe_allow_html=True
)

st.write("This system analyzes student performance data and predicts the risk of dropout.")

# Load Model
try:
    model = pickle.load(open("dropout_system.pkl","rb"))
except:
    st.error("❌ Model loading failed. Please upload dropout_system.pkl")
    st.stop()

# Sidebar Inputs
st.sidebar.header("📊 Enter Student Details")

attendance = st.sidebar.number_input(
    "📅 Attendance (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

marks = st.sidebar.number_input(
    "📝 Marks (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

behavior = st.sidebar.number_input(
    "🙂 Behavior Score (1-5)",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)

participation = st.sidebar.number_input(
    "🙋 Participation Score (1-5)",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)

st.sidebar.write("Click the button below to predict dropout risk")

# Predict Button
predict = st.sidebar.button("🔍 Predict Dropout Risk")

# Main Area
st.subheader("📊 Student Data Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Attendance", attendance)
col2.metric("Marks", marks)
col3.metric("Behavior", behavior)
col4.metric("Participation", participation)

st.write("---")

# Prediction
if predict:

    features = np.array([[attendance, marks, behavior, participation]])

    prediction = model.predict(features)

    probability = model.predict_proba(features)[0][1]

    risk_percent = round(probability * 100, 2)

    st.subheader("🎯 Prediction Result")

    st.progress(int(risk_percent))

    if prediction[0] == 1:

        st.error(f"⚠️ High Dropout Risk Detected ({risk_percent}%)")

        st.markdown(
        """
        ### 👨‍🏫 Counseling Recommendation
        - Provide academic guidance  
        - Monitor attendance closely  
        - Encourage classroom participation  
        - Offer mentoring support  
        """
        )

    else:

        st.success(f"✅ Student is Safe ({risk_percent}% risk)")

        st.markdown(
        """
        ### 🎓 Student Performance Status
        - Student is performing well  
        - Continue regular monitoring  
        - Encourage consistent participation  
        """
        )

st.write("---")

st.caption("AI Dropout Prediction System | Machine Learning Project")
