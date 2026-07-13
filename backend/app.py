from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "diabetes_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_data = np.array([
        float(data["Pregnancies"]),
        float(data["Glucose"]),
        float(data["BloodPressure"]),
        float(data["SkinThickness"]),
        float(data["Insulin"]),
        float(data["BMI"]),
        float(data["DiabetesPedigreeFunction"]),
        float(data["Age"])
    ]).reshape(1, -1)

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
