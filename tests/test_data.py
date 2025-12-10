import pandas as pd
import os

DATA_PATH = "data/credit_default.csv"


def test_data_file_exists():
    """Check if data file exists"""
    assert os.path.exists(DATA_PATH), "Dataset file not found"


def test_data_loads():
    """Check if dataset loads correctly"""
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset is empty"


def test_required_columns_exist():
    """Check required columns are present"""
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "ID",
        "LIMIT_BAL",
        "AGE",
        "default.payment.next.month"
    ]

    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
