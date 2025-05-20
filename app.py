from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = 1 if request.form["Gender"] == "Male" else 0
        age = float(request.form["Age"])
        height = float(request.form["Height"])
        weight = float(request.form["Weight"])
        duration = float(request.form["Duration"])
        heart_rate = float(request.form["Heart_Rate"])
        body_temp = float(request.form["Body_Temp"])

        features = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction_text=f"Calories Burnt: {prediction:.2f} kcal")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
