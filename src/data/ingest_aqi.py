import os
from src.data.aqi_api_client import fetch_current_aqi

RAW_DATA_PATH = "data/raw/aqi_raw.csv"
os.makedirs("data/raw", exist_ok=True)

def ingest(city: str = "Lahore"):
    df = fetch_current_aqi(city)

    if os.path.exists(RAW_DATA_PATH):
        df.to_csv(RAW_DATA_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(RAW_DATA_PATH, index=False)

    print("✅ AQI data ingested")
    print(df)

if __name__ == "__main__":
    ingest()
