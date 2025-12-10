import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/aqi_forecast.pkl")

# Sample input (must match training features exactly)
sample = pd.DataFrame([{
    "aqi": 145,
    "pm25": 82,
    "pm10": 120,
    "temperature": 29,
    "humidity": 58,
    "wind_speed": 3.5,
    "hour": 14,
    "day": 2,
    "month": 12,
    "aqi_lag_1": 142,
    "aqi_lag_2": 138,
    "aqi_lag_3": 135,
    "aqi_lag_6": 130,
    "aqi_lag_12": 125,
    "aqi_rolling_mean_3": 138.3,
    "aqi_rolling_std_3": 3.5
}])

# SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Display feature impact
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.show()
