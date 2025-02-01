import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("churn_model.pkl", "rb"))

# Streamlit UI
st.title("Customer Churn Prediction App")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100)
subscription_length = st.number_input("Subscription Length (Months)", min_value=1, max_value=60)
monthly_bill = st.number_input("Monthly Bill ($)", min_value=0.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0)
payment_method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Debit Card"])
contract_type = st.selectbox("Contract Type", ["Monthly", "Yearly"])

# Convert inputs
gender = 1 if gender == "Male" else 0
payment_method = {"Credit Card": 0, "PayPal": 1, "Debit Card": 2}[payment_method]
contract_type = 1 if contract_type == "Yearly" else 0

# Make prediction
if st.button("Predict Churn"):
    input_data = np.array([[gender, age, subscription_length, monthly_bill, total_charges, payment_method, contract_type]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("This customer is likely to churn!")
    else:
        st.success("This customer is likely to stay!")
