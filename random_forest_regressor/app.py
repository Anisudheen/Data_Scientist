import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Boston House Price Prediction", layout="centered")

st.title("Boston House Price Prediction (Random Forest)")

# ---- Load model and metadata ----
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Boston House Price Prediction", layout="centered")

# YOUR FILE ACTUALLY CONTAINS ONLY THE MODEL
model = joblib.load("random_forest_model.pkl")   # <-- NO saved[...] here
feature_names = ["CRIM","ZN","INDUS","CHAS","NOX",
                 "RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

# Optional: use data just to get reasonable default values
data = pd.read_csv("boston.csv")

# Short -> full display names
DISPLAY_NAMES = {
    "CRIM":   "Per capita crime rate by town",
    "ZN":     "Residential land zoned > 25,000 sq.ft. (%)",
    "INDUS":  "Non-retail business acres per town",
    "CHAS":   "Charles River dummy (1 = tract bounds river)",
    "NOX":    "Nitric oxides concentration (parts per 10 million)",
    "RM":     "Average number of rooms per dwelling",
    "AGE":    "Owner-occupied units built prior to 1940 (%)",
    "DIS":    "Weighted distance to Boston employment centers",
    "RAD":    "Accessibility to radial highways (index)",
    "TAX":    "Property-tax rate per $10,000",
    "PTRATIO":"Pupil–teacher ratio by town",
    "B":      "1000(Bk - 0.63)² (Bk = proportion of Black residents)",
    "LSTAT":  "Lower status population (%)"
}

st.sidebar.header("Input house features")

def user_inputs():
    values = {}
    inputs = []

    for col in feature_names:
        label = DISPLAY_NAMES.get(col, col)  # full name in UI
        if col in data.columns:
            col_min = float(data[col].min())
            col_max = float(data[col].max())
            col_mean = float(data[col].mean())
        else:
            col_min, col_max, col_mean = 0.0, 100.0, 0.0

        val = st.sidebar.number_input(
            label,
            value=col_mean,
            min_value=col_min,
            max_value=col_max
        )
        values[col] = val
        inputs.append(val)

    X_user = np.array(inputs).reshape(1, -1)
    return X_user, values

X_user, values = user_inputs()

if st.button("Predict price"):
    pred = model.predict(X_user)[0]
    st.subheader("Predicted Medium Value (house price)")
    st.write(f"**{pred:.2f}** (in $1000s)")

    st.subheader("Features used")
    st.json(values)
