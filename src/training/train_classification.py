import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.utils.data_loader import load_credit_data
from src.utils.experiment_logger import log_experiment

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

TARGET_COL = "default_payment_next_month"


def get_features_target(df: pd.DataFrame):
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def train_and_evaluate():
    df = load_credit_data()
    df = df.dropna()  # simple data quality handling

    X, y = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logreg_baseline": LogisticRegression(max_iter=1000),
        "rf_improved": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),
        "gb_improved": GradientBoostingClassifier(random_state=42)
    }

    best_model_name = None
    best_f1 = -1.0

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        log_experiment(
            {
                "task": "classification",
                "model_name": name,
                "accuracy": acc,
                "f1": f1,
                "roc_auc": roc,
                "rmse": "",
                "comment": "baseline" if "baseline" in name else "improved",
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            joblib.dump(pipe, MODELS_DIR / "best_classification.pkl")

    print(f"Best classification model: {best_model_name} with F1={best_f1:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
