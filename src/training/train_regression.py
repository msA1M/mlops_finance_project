import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


LOG_COLUMNS = [
    "task",
    "model_name",
    "accuracy",
    "f1",
    "rmse",
    "mae",
    "r2",
    "silhouette"
]


def log_results(rows):
    os.makedirs("experiments", exist_ok=True)
    log_path = "experiments/experiment_log.csv"

    df_new = pd.DataFrame(rows, columns=LOG_COLUMNS)

    if os.path.exists(log_path):
        df_old = pd.read_csv(log_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(log_path, index=False)


def train_and_evaluate():
    # Load data
    df = pd.read_csv("data/credit_default.csv")
    df = df.rename(columns={
        "default.payment.next.month": "default_payment_next_month"
    })

    # Regression target
    y = df["LIMIT_BAL"]
    X = df.drop(columns=["LIMIT_BAL", "default_payment_next_month"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    results = []

    # =========================
    # Baseline: Linear Regression
    # =========================
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)

    lr_rmse = mean_squared_error(y_test, lr_pred) ** 0.5

    results.append({
        "task": "regression",
        "model_name": "linear_regression",
        "accuracy": None,
        "f1": None,
        "rmse": lr_rmse,
        "mae": mean_absolute_error(y_test, lr_pred),
        "r2": r2_score(y_test, lr_pred),
        "silhouette": None
    })

    # =========================
    # Improved: Random Forest
    # =========================
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5

    results.append({
        "task": "regression",
        "model_name": "rf_improved",
        "accuracy": None,
        "f1": None,
        "rmse": rf_rmse,
        "mae": mean_absolute_error(y_test, rf_pred),
        "r2": r2_score(y_test, rf_pred),
        "silhouette": None
    })

    # =========================
    # Save logs
    # =========================
    log_results(results)

    print("✅ Regression results logged successfully.")


if __name__ == "__main__":
    train_and_evaluate()
