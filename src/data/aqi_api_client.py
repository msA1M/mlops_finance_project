import requests
import pandas as pd
from datetime import datetime

API_KEY = "e568e66ccf9791b9fcc7ea8685420a57"
LAT = 31.5204
LON = 74.3587

def fetch_current_aqi(city: str = "Lahore") -> pd.DataFrame:
    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={LAT}&lon={LON}&appid={API_KEY}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    entry = data["list"][0]

    row = {
        "city": city,
        "timestamp": datetime.utcfromtimestamp(entry["dt"]),
        "aqi": entry["main"]["aqi"],        # scale 1–5
        "pm25": entry["components"]["pm2_5"],
        "pm10": entry["components"]["pm10"],
    }

    return pd.DataFrame([row])
