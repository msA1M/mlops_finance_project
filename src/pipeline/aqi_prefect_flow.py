from prefect import flow, task
import subprocess
import json
import datetime
import os

# -------------------------------
# TASK 1 — Ingest AQI data
# -------------------------------
@task(retries=2, retry_delay_seconds=5)
def ingest_aqi():
    print("🌍 Ingesting AQI data...")
    subprocess.run(
        ["python", "src/data/ingest_aqi.py"],
        check=True,
        env={**dict(os.environ), "PYTHONPATH": "."}
    )

# -------------------------------
# TASK 2 — Feature engineering
# -------------------------------
@task
def build_features():
    print("🛠 Building feature store...")
    subprocess.run(
        ["python", "src/data/aqi_feature_pipeline.py"],
        check=True,
        env={**dict(os.environ), "PYTHONPATH": "."}
    )

# -------------------------------
# TASK 3 — Train model
# -------------------------------
@task
def train_model():
    print("🤖 Training AQI model...")
    subprocess.run(
        ["python", "src/training/train_aqi_models.py"],
        check=True,
        env={**dict(os.environ), "PYTHONPATH": "."}
    )

# -------------------------------
# TASK 4 — Update registry
# -------------------------------
@task
def update_registry():
    print("📝 Updating model registry...")

    registry = {
        "model_name": "random_forest",
        "model_path": "models/aqi_forecast.pkl",
        "registered_at": datetime.datetime.now().isoformat()
    }

    with open("models/registry.json", "w") as f:
        json.dump(registry, f, indent=2)

# -------------------------------
# MAIN FLOW
# -------------------------------
@flow(name="AQI Full Training Pipeline")
def aqi_pipeline():
    ingest_aqi()
    build_features()
    train_model()
    update_registry()
    print("✅ AQI pipeline completed successfully!")

if __name__ == "__main__":
    aqi_pipeline()
