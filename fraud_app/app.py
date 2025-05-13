import streamlit as st
import joblib # type: ignore
import numpy as np

# Load the saved model and scaler
model = joblib.load('fraud_model.pkl') #this is the logistic regression model
scaler = joblib.load('scaler.pkl') #this is the standardscaler that was applied

st.title("Credit Card Fraud Detection App")

st.markdown("This tool helps detect whether a transaction is potentially fraudulent based on key details.")


st.write("Enter transaction details below:")

# Input form for numerical answers
home_distance = st.number_input("How far is the transaction location from the cardholder’s home? (e.g., km)", min_value=0.0, step=0.1)
last_transaction_distance = st.number_input("How far is this transaction from the location of the last one? ", min_value=0.0, step=0.1)
price_ratio = st.number_input("How does the price of this transaction compare to average spending? (Enter a ratio)", min_value=0.0, step=0.01)

#adding a slider for ratio
#price_ratio = st.slider("How does the price of this transaction compare to average spending? (Ratio)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)

#is_repeat = st.selectbox("Is this a repeat transaction?", [0, 1])
#card_used = st.selectbox("Card Used?", [0, 1])
#pin_used = st.selectbox("PIN Used?", [0, 1])
#online_order = st.selectbox("Online Order?", [0, 1])

#dropdown menu for categorical features
is_repeat = st.selectbox("Has the customer purchased from this retailer before?", ["Yes", "No"])
card_used = st.selectbox("Was the physical card used for this transaction?", ["Yes", "No"])
pin_used = st.selectbox("Was a PIN number used during the transaction?", ["Yes", "No"])
online_order = st.selectbox("Was this transaction made online?", ["Yes", "No"])

#converting the no/yes to 0 or 1 as the model uses
is_repeat = 1 if is_repeat == "Yes" else 0
card_used = 1 if card_used == "Yes" else 0
pin_used = 1 if pin_used == "Yes" else 0
online_order = 1 if online_order == "Yes" else 0

# Predict button
if st.button("Predict Fraud"):
    input_data = np.array([[home_distance, last_transaction_distance, price_ratio,
                            is_repeat, card_used, pin_used, online_order]])
    
    # Scale the input just like during training, using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Making the prediction based on the input 
    prediction = model.predict(input_scaled) 

    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction")