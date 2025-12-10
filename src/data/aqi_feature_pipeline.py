import os
import pandas as pd

RAW_PATH = "data/raw/aqi_raw.csv"
FEATURE_STORE_PATH = "data/feature_store/aqi_features.csv"

os.makedirs("data/feature_store", exist_ok=True)


def build_feature_store():
    """
    Builds time-series features from raw AQI data
    and stores them in the feature store.
    """
    df = pd.read_csv(RAW_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df = df.sort_values("timestamp")

    # ---------- Basic cleaning ----------
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.dropna()

    # ---------- Time-based features ----------
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    # ---------- Lag features ----------
    for lag in [1, 2, 3, 6, 12]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    # ---------- Rolling statistics ----------
    df["aqi_rolling_mean_3"] = df["aqi"].rolling(3).mean()
    df["aqi_rolling_std_3"] = df["aqi"].rolling(3).std()

    # ---------- Forecast target ----------
    # Predict AQI 1 step ahead
    df["aqi_target"] = df["aqi"].shift(-1)

    # Drop rows created by shifting
    df = df.dropna()

    df.to_csv(FEATURE_STORE_PATH, index=False)
    print("✅ Feature store created:")
    print(df.head())

    return df


if __name__ == "__main__":
    build_feature_store()
