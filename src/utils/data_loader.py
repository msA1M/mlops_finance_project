import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def load_credit_data(filename: str = "credit_default.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    df = pd.read_csv(path)

    # ✅ STANDARDIZE TARGET COLUMN NAME
    if "default.payment.next.month" in df.columns:
        df = df.rename(
            columns={"default.payment.next.month": "default_payment_next_month"}
        )

    return df

def load_stock_data(filename: str = "stock_prices.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    df = pd.read_csv(path)

    # ✅ STANDARDIZE DATE COLUMN
    if "date" not in df.columns:
        for col in df.columns:
            if col.lower() in ["date", "datetime", "timestamp", "time"]:
                df = df.rename(columns={col: "date"})
                break

    # ✅ STANDARDIZE CLOSE PRICE COLUMN
    if "close" not in df.columns:
        for col in df.columns:
            if col.lower() == "close":
                df = df.rename(columns={col: "close"})
                break

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(
            "stock_prices.csv must contain a date/time column and a close price column"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    return df

