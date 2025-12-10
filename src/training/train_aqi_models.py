import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.registry.model_registry import update_registry

FEATURE_PATH = "data/feature_store/aqi_features.csv"
MODEL_DIR = "models"
LOG_PATH = "experiments/experiment_log.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("experiments", exist_ok=True)


def load_data():
    df = pd.read_csv(FEATURE_PATH)

    target = "aqi_target"
    X = df.drop(columns=["aqi_target", "timestamp", "city"], errors="ignore")
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(y_true, y_pred):
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def log_experiment(model_name, metrics):
    row = {
        "task": "aqi_forecast",
        "model": model_name,
        **metrics
    }
    df_row = pd.DataFrame([row])

    if os.path.exists(LOG_PATH):
        df_old = pd.read_csv(LOG_PATH)
        df_all = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_all = df_row

    df_all.to_csv(LOG_PATH, index=False)


def train_models():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n🚀 Training {name}")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics

        print(f"📊 Metrics: {metrics}")

        log_experiment(name, metrics)

    # Select best model (lowest RMSE)
    best_model_name = min(results, key=lambda k: results[k]["rmse"])
    best_metrics = results[best_model_name]

    print(f"\n✅ Best Model: {best_model_name}")

    best_model = models[best_model_name]
    model_path = f"{MODEL_DIR}/aqi_forecast.pkl"
    joblib.dump(best_model, model_path)

    update_registry(
        model_name=best_model_name,
        path=model_path,
        metrics=best_metrics
    )

    print("✅ Best model saved & registered")


if __name__ == "__main__":
    train_models()
