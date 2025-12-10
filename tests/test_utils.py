from src.utils.experiment_logger import log_experiment
from pathlib import Path

def test_experiment_logging(tmp_path):
    log_experiment(
        "test_model",
        {"rmse": 10.2, "mae": 8.5}
    )
    assert Path("experiments/experiment_log.csv").exists()
