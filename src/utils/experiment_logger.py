import csv
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("experiments/experiment_log.csv")

def log_experiment(model_name, metrics: dict):
    LOG_PATH.parent.mkdir(exist_ok=True)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        **metrics
    }

    file_exists = LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
