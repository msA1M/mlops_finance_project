import streamlit as st
import requests

st.set_page_config(page_title="AQI Forecasting Platform", layout="centered")

st.title("🌍 AQI Forecasting Dashboard")
st.write("Predict next-step Air Quality Index using Machine Learning")

API_URL = "http://api:8000/predict"


st.subheader("📥 Input AQI & Weather Data")

with st.form("aqi_form"):
    aqi = st.number_input("Current AQI", 0.0, 500.0, 145.0)
    pm25 = st.number_input("PM2.5", 0.0, 500.0, 80.0)
    pm10 = st.number_input("PM10", 0.0, 500.0, 120.0)
    temperature = st.number_input("Temperature (°C)", -10.0, 55.0, 29.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 3.5)

    hour = st.slider("Hour of Day", 0, 23, 14)
    day = st.slider("Day of Week (0=Mon)", 0, 6, 2)
    month = st.slider("Month", 1, 12, 12)

    aqi_lag_1 = st.number_input("AQI Lag 1", 0.0, 500.0, 142.0)
    aqi_lag_2 = st.number_input("AQI Lag 2", 0.0, 500.0, 138.0)
    aqi_lag_3 = st.number_input("AQI Lag 3", 0.0, 500.0, 135.0)
    aqi_lag_6 = st.number_input("AQI Lag 6", 0.0, 500.0, 130.0)
    aqi_lag_12 = st.number_input("AQI Lag 12", 0.0, 500.0, 125.0)

    aqi_rolling_mean_3 = st.number_input("AQI Rolling Mean (3)", 0.0, 500.0, 138.3)
    aqi_rolling_std_3 = st.number_input("AQI Rolling Std (3)", 0.0, 50.0, 3.5)

    submit = st.form_submit_button("🚀 Predict AQI")

if submit:
    payload = {
        "aqi": aqi,
        "pm25": pm25,
        "pm10": pm10,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "hour": hour,
        "day": day,
        "month": month,
        "aqi_lag_1": aqi_lag_1,
        "aqi_lag_2": aqi_lag_2,
        "aqi_lag_3": aqi_lag_3,
        "aqi_lag_6": aqi_lag_6,
        "aqi_lag_12": aqi_lag_12,
        "aqi_rolling_mean_3": aqi_rolling_mean_3,
        "aqi_rolling_std_3": aqi_rolling_std_3
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        prediction = result["predicted_aqi"]
        alert_level = result["alert_level"]


        st.success(f"✅ Predicted AQI: {prediction:.2f}")

        if prediction <= 50:
            st.success("🟢 Good Air Quality")
        elif prediction <= 100:
            st.info("🟡 Moderate Air Quality")
        elif prediction <= 150:
            st.warning("🟠 Unhealthy for Sensitive Groups")
        else:
            st.error("🔴 Unhealthy / Hazardous")

    except Exception as e:
        st.error(f"❌ API Error: {e}")
