import streamlit as st
import joblib
import pandas as pd

# Load model
data = joblib.load("best_model.pkl")
model = data["model"]
threshold = data["threshold"]
features = data["features"]

st.title("💳 Fraud Detection System")

# --- INPUTS ---
step = st.number_input("Step", value=1)
type_ = st.number_input("Type (encoded)", value=1)
amount = st.number_input("Amount", value=0.0)
oldbalanceOrg = st.number_input("Old Balance Origin", value=0.0)
newbalanceOrig = st.number_input("New Balance Origin", value=0.0)
oldbalanceDest = st.number_input("Old Balance Destination", value=0.0)
newbalanceDest = st.number_input("New Balance Destination", value=0.0)

if st.button("Predict"):

    # Feature Engineering (same as training)
    balance_diff_orig = oldbalanceOrg - newbalanceOrig
    balance_diff_dest = newbalanceDest - oldbalanceDest
    hour = step % 24

    # Create dataframe in SAME ORDER
    input_data = pd.DataFrame([{
        'step': step,
        'type': type_,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'balance_diff_orig': balance_diff_orig,
        'balance_diff_dest': balance_diff_dest,
        'hour': hour
    }])

    # Ensure correct column order
    input_data = input_data[features]

    # Prediction
    prob = model.predict_proba(input_data)[:, 1][0]

    if prob >= threshold:
        st.error(f"⚠️ Fraud Detected (Prob: {prob:.2f})")
    else:
        st.success(f"✅ Safe Transaction (Prob: {prob:.2f})")
        