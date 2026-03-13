from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("AI based dropout system.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    attendance = float(request.form["attendance"])
    marks = float(request.form["marks"])
    behavior = float(request.form["behavior"])
    participation = float(request.form["participation"])

    features = np.array([[attendance, marks, behavior, participation]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "⚠️ High Dropout Risk – Counseling Required 👨‍🏫"
    else:
        result = "✅ Student Safe – No Dropout Risk 🎓"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
