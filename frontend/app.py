import streamlit as st
import requests

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Default Prediction",
    layout="centered"
)

st.title("💳 Credit Default Prediction App")
st.write("Enter customer details to predict default risk.")

# --------------------------------------------------
# Input Form
# --------------------------------------------------
with st.form("prediction_form"):
    ID = st.number_input("Customer ID", value=1)

    LIMIT_BAL = st.number_input("Credit Limit", value=20000)
    AGE = st.number_input("Age", value=30)
    SEX = st.selectbox("Sex", [1, 2])  # 1=Male, 2=Female
    EDUCATION = st.selectbox("Education Level", [1, 2, 3, 4])
    MARRIAGE = st.selectbox("Marriage Status", [1, 2, 3])

    PAY_0 = st.number_input("PAY_0", value=0)
    PAY_2 = st.number_input("PAY_2", value=0)
    PAY_3 = st.number_input("PAY_3", value=0)
    PAY_4 = st.number_input("PAY_4", value=0)
    PAY_5 = st.number_input("PAY_5", value=0)
    PAY_6 = st.number_input("PAY_6", value=0)

    BILL_AMT1 = st.number_input("BILL_AMT1", value=0)
    BILL_AMT2 = st.number_input("BILL_AMT2", value=0)
    BILL_AMT3 = st.number_input("BILL_AMT3", value=0)
    BILL_AMT4 = st.number_input("BILL_AMT4", value=0)
    BILL_AMT5 = st.number_input("BILL_AMT5", value=0)
    BILL_AMT6 = st.number_input("BILL_AMT6", value=0)

    PAY_AMT1 = st.number_input("PAY_AMT1", value=0)
    PAY_AMT2 = st.number_input("PAY_AMT2", value=0)
    PAY_AMT3 = st.number_input("PAY_AMT3", value=0)
    PAY_AMT4 = st.number_input("PAY_AMT4", value=0)
    PAY_AMT5 = st.number_input("PAY_AMT5", value=0)
    PAY_AMT6 = st.number_input("PAY_AMT6", value=0)

    submit = st.form_submit_button("🔍 Predict")

# --------------------------------------------------
# API Call
# --------------------------------------------------
if submit:
    payload = {
        "ID": ID,
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "PAY_3": PAY_3,
        "PAY_4": PAY_4,
        "PAY_5": PAY_5,
        "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1,
        "BILL_AMT2": BILL_AMT2,
        "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4,
        "BILL_AMT5": BILL_AMT5,
        "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1,
        "PAY_AMT2": PAY_AMT2,
        "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4,
        "PAY_AMT5": PAY_AMT5,
        "PAY_AMT6": PAY_AMT6
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()

        if result["prediction"] == 1:
            st.error("⚠️ High Risk: Customer is likely to default.")
        else:
            st.success("✅ Low Risk: Customer is unlikely to default.")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")
