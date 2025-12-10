from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# ------------------------------------------------------
# Initialize FastAPI
# ------------------------------------------------------
app = FastAPI(
    title="Credit Default Prediction API",
    description="ML model API for predicting credit default using trained Gradient Boosting / Random Forest model",
    version="1.0.0"
)

# ------------------------------------------------------
# Load Model on Startup
# ------------------------------------------------------
MODEL_PATH = "models/best_classifier.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = model.feature_names_in_.tolist()

print("✅ Loaded model:", type(model))


# ------------------------------------------------------
# Input Schema for Predictions
# These fields must match your training data features.
# ------------------------------------------------------
class InputData(BaseModel):
    ID: int
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


# ------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------
@app.get("/")
def home():
    return {"status": "ok", "message": "Credit Default Prediction API is running"}


# ------------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])

        # DEBUG: print columns
        print("Incoming columns:", df.columns.tolist())

        prediction = model.predict(df)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        print("🔥 ERROR INSIDE /predict:", str(e))
        raise e

