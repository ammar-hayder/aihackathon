ffrom flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from datetime import datetime
import os
import io

app = Flask(__name__)
CORS(app)

# Max file size: 1 MB
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {'csv'}

model = None
encoders = {}
confidence_score = 0
required_columns = [
    "order_datetime", "day_of_week", "customer_type", "meal_type", "quantity_ordered",
    "weather_condition", "traffic_level", "distance_km", "delivery_time_min",
    "is_promo_active", "prepared_meals"
]
features = [
    "day_of_week", "customer_type", "meal_type", "quantity_ordered",
    "weather_condition", "traffic_level", "distance_km", "delivery_time_min",
    "is_promo_active", "hour", "day", "month"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------- MODEL TRAINING FUNCTION ---------------------
def train_model_from_df(data: pd.DataFrame):
    global model, encoders, confidence_score

    data["order_datetime"] = pd.to_datetime(data["order_datetime"], errors='coerce')
    data.dropna(subset=["order_datetime"], inplace=True)
    data["hour"] = data["order_datetime"].dt.hour
    data["day"] = data["order_datetime"].dt.day
    data["month"] = data["order_datetime"].dt.month

    # Fill numeric nulls with median, categorical with mode
    for col in data.columns:
        if data[col].dtype in [int, float]:
            data[col].fillna(data[col].median(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)

    categorical_cols = ["day_of_week", "customer_type", "meal_type", "weather_condition", "traffic_level"]
    encoders = {col: LabelEncoder() for col in categorical_cols}
    for col in categorical_cols:
        data[col] = encoders[col].fit_transform(data[col])

    data["is_promo_active"] = data["is_promo_active"].astype(int)

    X = data[features]
    y = data["prepared_meals"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    confidence_score = r2_score(y, model.predict(X))

# --------------------- ROUTE: Upload CSV ---------------------
@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Must be a .csv"}), 400

    try:
        # Read CSV directly from file stream
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        df = pd.read_csv(stream)

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing required columns: {missing}"}), 400

        train_model_from_df(df)
        return jsonify({"message": f"Model trained using {file.filename}"}), 200

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

        meal_type = "Lunch"  # You can make this dynamic if needed

        input_dict = {
            "day_of_week": encoders["day_of_week"].transform([day_of_week])[0],
            "customer_type": encoders["customer_type"].transform(["Returning"])[0],
            "meal_type": encoders["meal_type"].transform([meal_type])[0],
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
            "meal_type": meal_type,
            "predicted_prepared_meals": round(prediction),
            "confidence_score": round(confidence_score, 3),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- START APP ---------------------
if __name__ == '__main__':
    try:
        print("Waiting for CSV upload to train model...")
    except Exception:
        print("Startup failed.")
    app.run(debug=True)

# --------------------- START APP ---------------------
print("Waiting for CSV upload to train model...")
