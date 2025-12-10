import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/aqi_forecast.pkl")

def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None

def test_model_predicts():
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([{
        "aqi": 150,
        "pm25": 90,
        "pm10": 110,
        "temperature": 30,
        "humidity": 60,
        "wind_speed": 5,
        "hour": 12,
        "day": 1,
        "month": 12,
        "aqi_lag_1": 145,
        "aqi_lag_2": 140,
        "aqi_lag_3": 138,
        "aqi_lag_6": 130,
        "aqi_lag_12": 125,
        "aqi_rolling_mean_3": 142,
        "aqi_rolling_std_3": 6
    }])

    pred = model.predict(X)
    assert isinstance(pred[0], float)
