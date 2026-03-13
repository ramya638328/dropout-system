import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="AI Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>

.stApp {
background: linear-gradient(to right,#e3f2fd,#ffffff);
}

h1 {
color:#1a237e;
text-align:center;
}

.sidebar .sidebar-content {
background-color:#f0f4ff;
}

.stButton>button {
background-color:#1976D2;
color:white;
font-size:18px;
border-radius:10px;
height:50px;
width:100%;
}

.stButton>button:hover {
background-color:#0d47a1;
color:white;
}

.result-safe {
background-color:#d4edda;
padding:20px;
border-radius:10px;
font-size:20px;
}

.result-risk {
background-color:#f8d7da;
padding:20px;
border-radius:10px;
font-size:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("# 🎓 AI Based Student Dropout Prediction System")

st.write(
"Analyze student performance data and predict dropout risk using Machine Learning."
)

# ---------- Load Model ----------
try:
    model = pickle.load(open("dropout.pkl","rb"))
except:
    st.error("❌ Model loading failed. Upload dropout.pkl")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.header("📊 Enter Student Details")

attendance = st.sidebar.number_input("📅 Attendance (%)",0,100,50)

marks = st.sidebar.number_input("📝 Marks (%)",0,100,50)

behavior = st.sidebar.number_input("🙂 Behavior Score (1-5)",1,5,3)

participation = st.sidebar.number_input("🙋 Participation Score (1-5)",1,5,3)

predict = st.sidebar.button("🔍 Predict Dropout Risk")

# ---------- Display Student Data ----------
st.subheader("📈 Student Data Overview")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Attendance",attendance)
col2.metric("Marks",marks)
col3.metric("Behavior",behavior)
col4.metric("Participation",participation)

st.write("---")

# ---------- Prediction ----------
if predict:

    features = np.array([[attendance,marks,behavior,participation]])

    prediction = model.predict(features)

    probability = model.predict_proba(features)[0][1]

    risk = round(probability*100,2)

    st.subheader("🎯 Prediction Result")

    st.progress(int(risk))

    if prediction[0] == 1:

        st.markdown(
        f"""
        <div class="result-risk">
        ⚠️ High Dropout Risk Detected <br>
        Risk Level : <b>{risk}%</b><br><br>
        👨‍🏫 Counseling Recommended
        </div>
        """,
        unsafe_allow_html=True
        )

        st.write("• Monitor attendance regularly")
        st.write("• Provide academic mentoring")
        st.write("• Encourage class participation")

    else:

        st.markdown(
        f"""
        <div class="result-safe">
        ✅ Student is Safe <br>
        Risk Level : <b>{risk}%</b>
        </div>
        """,
        unsafe_allow_html=True
        )

        st.write("• Student performance is stable")
        st.write("• Continue monitoring progress")

st.write("---")

st.caption("AI Dropout Prediction System | Machine Learning Project")
