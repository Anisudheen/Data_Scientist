import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Social Network Predictor")

st.title("🔹 Social Network Predicting App")

# Load saved model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.subheader("Enter Your Details:")

# INPUTS (change names if needed)
feature1 = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=18,
    step=1
)

feature2 = st.number_input(
    "Salary",
    min_value=10000,
    max_value=200000,
    value=20000,
    step=500
)

if st.button("Predict"):
    # Convert input to array
    input_data = np.array([[feature1, feature2]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    st.success(f"🎯 Predicted Value: {prediction[0]:.4f}")
