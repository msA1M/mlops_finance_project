import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.alerts.alert_manager import check_aqi_alert, log_alert

app = FastAPI(title="AQI Forecasting API")


# -------------------------------------------------------
# 1️⃣ Load Model From registry.json
# -------------------------------------------------------
def load_latest_model():
    registry_path = "models/registry.json"

    if not os.path.exists(registry_path):
        raise FileNotFoundError("❌ Model registry.json not found in /models folder.")

    with open(registry_path, "r") as f:
        reg = json.load(f)

    model_path = reg["model_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    print(f"📌 Loading model: {os.path.abspath(model_path)}")
    return joblib.load(model_path)


model = load_latest_model()


# -------------------------------------------------------
# 2️⃣ Pydantic model for incoming prediction request
# MUST MATCH MODEL FEATURE NAMES EXACTLY
# -------------------------------------------------------
class AQIInput(BaseModel):
    aqi: float
    pm25: float
    pm10: float
    temperature: float
    humidity: float
    wind_speed: float
    hour: int
    day: int
    month: int
    aqi_lag_1: float
    aqi_lag_2: float
    aqi_lag_3: float
    aqi_lag_6: float
    aqi_lag_12: float
    aqi_rolling_mean_3: float
    aqi_rolling_std_3: float


# -------------------------------------------------------
# 3️⃣ Health Check Route
# -------------------------------------------------------
@app.get("/")
def home():
    return {"message": "AQI Forecasting API is running!"}


# -------------------------------------------------------
# 4️⃣ Prediction Route
# -------------------------------------------------------
@app.post("/predict")
def predict(input_data: AQIInput):
    try:
        # Convert incoming JSON → DataFrame
        df = pd.DataFrame([input_data.dict()])

        print("\n🔎 Incoming columns:", df.columns.tolist())
        print("🔎 Model expects:", model.feature_names_in_.tolist())

        # Ensure correct column order
        df = df[model.feature_names_in_]

        prediction = model.predict(df)[0]

        alert_level = "GOOD"

        if prediction >= 200:
            alert_level = "HAZARDOUS"
        elif prediction >= 150:
            alert_level = "UNHEALTHY"
        elif prediction >= 100:
            alert_level = "MODERATE"

        return {
            "predicted_aqi": float(prediction),
            "alert_level": alert_level
        }


    except Exception as e:
        print("🔥 Prediction Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
