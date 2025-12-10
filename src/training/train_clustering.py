import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


import joblib
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils.data_loader import load_credit_data
from src.utils.experiment_logger import log_experiment

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def train_clustering():
    df = load_credit_data()
    df = df.dropna()

    # Use numeric features only, drop target
    if "default_payment_next_month" in df.columns:
        df = df.drop(columns=["default_payment_next_month"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    sil = silhouette_score(X_pca, clusters)

    log_experiment(
        {
            "task": "clustering",
            "model_name": "kmeans_pca",
            "silhouette": sil,
            "comment": "PCA + KMeans 4 clusters",
        }
    )

    joblib.dump(
        {"scaler": scaler, "pca": pca, "kmeans": kmeans},
        MODELS_DIR / "best_cluster.pkl",
    )

    print(f"Clustering silhouette score: {sil:.4f}")


if __name__ == "__main__":
    train_clustering()
