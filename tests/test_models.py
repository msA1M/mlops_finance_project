import joblib
import pandas as pd
import os

MODEL_PATH = "models/best_classifier.pkl"


def test_model_file_exists():
    """Check if trained model file exists"""
    assert os.path.exists(MODEL_PATH), "Model file not found"


def test_model_loads():
    """Check if model loads correctly"""
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_prediction():
    """Check model can make a prediction"""
    model = joblib.load(MODEL_PATH)

    sample_input = pd.DataFrame([{
        "ID": 1,
        "LIMIT_BAL": 20000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 5000,
        "BILL_AMT2": 2000,
        "BILL_AMT3": 1000,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 1000,
        "PAY_AMT2": 500,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0
    }])

    prediction = model.predict(sample_input)[0]

    assert prediction in [0, 1], "Prediction is not binary"
