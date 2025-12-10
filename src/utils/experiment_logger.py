import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

EXPERIMENT_LOG = Path("experiments/experiment_log.csv")
EXPERIMENT_LOG.parent.mkdir(exist_ok=True, parents=True)


def log_experiment(record: Dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **record
    }
    write_header = not EXPERIMENT_LOG.exists()

    with EXPERIMENT_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)
