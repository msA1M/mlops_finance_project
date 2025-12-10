import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.utils.data_loader import load_stock_data
from src.utils.experiment_logger import log_experiment

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def create_lag_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["close"].shift(lag)
    df = df.dropna()
    return df


def train_timeseries():
    df = load_stock_data()
    df = create_lag_features(df, n_lags=5)

    feature_cols = [c for c in df.columns if c.startswith("lag_")]
    X = df[feature_cols]
    y = df["close"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5


    log_experiment(
        {
            "task": "timeseries",
            "model_name": "rf_lag",
            "rmse": rmse,
            "comment": "RandomForest with lag features",
        }
    )

    joblib.dump(
        {"model": model, "feature_cols": feature_cols},
        MODELS_DIR / "best_timeseries.pkl",
    )

    print(f"Time series RMSE: {rmse:.4f}")


if __name__ == "__main__":
    train_timeseries()
