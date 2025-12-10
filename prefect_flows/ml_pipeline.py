import sys
from pathlib import Path

print("✅ ml_pipeline.py started")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from prefect import flow, task

from src.training.train_classification import train_and_evaluate as train_cls
from src.training.train_regression import train_and_evaluate as train_reg
from src.training.train_clustering import train_clustering
from src.training.train_timeseries import train_timeseries


@task(retries=2, retry_delay_seconds=30)
def classification_task():
    print("▶ Running classification task")
    train_cls()


@task(retries=2, retry_delay_seconds=30)
def regression_task():
    print("▶ Running regression task")
    train_reg()


@task(retries=2, retry_delay_seconds=30)
def clustering_task():
    print("▶ Running clustering task")
    train_clustering()


@task(retries=2, retry_delay_seconds=30)
def timeseries_task():
    print("▶ Running timeseries task")
    train_timeseries()


@flow(name="finance-ml-pipeline")
def main_pipeline():
    print("🚀 Prefect flow started")
    classification_task()
    regression_task()
    clustering_task()
    timeseries_task()
    print("✅ Prefect flow finished")


if __name__ == "__main__":
    print("✅ __main__ triggered")
    main_pipeline()
