import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/feature_store/aqI_features.csv")

def load_features():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Feature store not found.")
    return pd.read_csv(DATA_PATH)
