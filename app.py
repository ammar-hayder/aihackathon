from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = "orders.csv"
model = None
encoders = {}
features = [
    "day_of_week", "customer_type", "meal_type", "quantity_ordered",
    "weather_condition", "traffic_level", "distance_km", "delivery_time_min",
    "is_promo_active", "hour", "day", "month"
]

# --------------------- MODEL TRAINING FUNCTION ---------------------
def train_model():
    global model, encoders

    path = os.path.join(UPLOAD_FOLDER, CSV_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError("orders.csv not found")

    data = pd.read_csv(path)
    data["order_datetime"] = pd.to_datetime(data["order_datetime"])
    data["hour"] = data["order_datetime"].dt.hour
    data["day"] = data["order_datetime"].dt.day
    data["month"] = data["order_datetime"].dt.month

    categorical_cols = ["day_of_week", "customer_type", "meal_type", "weather_condition", "traffic_level"]
    encoders = {col: LabelEncoder() for col in categorical_cols}
    for col in categorical_cols:
        data[col] = encoders[col].fit_transform(data[col])

    data["is_promo_active"] = data["is_promo_active"].astype(int)

    X = data[features]
    y = data["prepared_meals"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

# --------------------- ROUTE: Upload CSV ---------------------
@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file. Must be .csv"}), 400

    try:
        path = os.path.join(UPLOAD_FOLDER, CSV_FILENAME)
        file.save(path)

        train_model()
        return jsonify({"message": "CSV uploaded and model trained"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- ROUTE: Predict ---------------------
@app.route('/predict', methods=['GET'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not trained yet. Please upload CSV first."}), 400

        date_str = request.args.get("date")
        if not date_str:
            return jsonify({"error": "Missing 'date' parameter in format YYYY-MM-DD"}), 400

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_of_week = dt.strftime("%A")

        input_dict = {
            "day_of_week": encoders["day_of_week"].transform([day_of_week])[0],
            "customer_type": encoders["customer_type"].transform(["Returning"])[0],
            "meal_type": encoders["meal_type"].transform(["Lunch"])[0],
            "quantity_ordered": 200,
            "weather_condition": encoders["weather_condition"].transform(["Cloudy"])[0],
            "traffic_level": encoders["traffic_level"].transform(["Medium"])[0],
            "distance_km": 6.0,
            "delivery_time_min": 30,
            "is_promo_active": 1,
            "hour": 12,
            "day": dt.day,
            "month": dt.month,
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

        return jsonify({
            "date": date_str,
            "predicted_prepared_meals": round(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- START APP ---------------------
try:
    train_model()
except Exception:
    print("Model not trained. Please upload CSV using /upload.")
