import json
import os
from datetime import datetime

REGISTRY_PATH = "models/registry.json"

def update_registry(model_name: str, path: str, metrics: dict):
    """
    Updates the model registry with the latest best model.
    """
    os.makedirs("models", exist_ok=True)

    registry_entry = {
        "model_name": model_name,
        "model_path": path,
        "metrics": metrics,
        "registered_at": datetime.utcnow().isoformat()
    }

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry_entry, f, indent=4)

    print("✅ Model registry updated")
