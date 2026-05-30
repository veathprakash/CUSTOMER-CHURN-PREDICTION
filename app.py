import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction App")

# Load model + columns
try:
    model = joblib.load("customer_churn_model.pkl")
    columns = joblib.load("columns.pkl")
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -------------------------
# USER INPUT SECTION
# -------------------------

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Create input dataframe
input_dict = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service
}

input_df = pd.DataFrame([input_dict])

# -------------------------
# MATCH TRAINING FORMAT
# -------------------------

# Encode same way as training (IMPORTANT)
for col in input_df.columns:
    if input_df[col].dtype == "object":
        input_df[col] = input_df[col].astype("category").cat.codes

# Reorder columns same as training
input_df = input_df.reindex(columns=columns, fill_value=0)

# -------------------------
# PREDICTION
# -------------------------

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠ Customer will CHURN")
    else:
        st.success(" Customer will NOT churn")