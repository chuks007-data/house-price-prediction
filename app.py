import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data/house_sales.csv")

# ----------------------------
# Prepare Data
# ----------------------------
df = pd.get_dummies(df, drop_first=True)

X = df.drop("sale_price", axis=1)
y = df["sale_price"]

# ----------------------------
# Train Model
# ----------------------------
model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Streamlit App
# ----------------------------
st.title("🏠 House Price Prediction App")

st.write("Predict house prices using machine learning.")

# ----------------------------
# User Inputs
# ----------------------------
area = st.number_input("Area (Square Feet)", min_value=500, max_value=10000, value=2000)

bedrooms = st.slider("Bedrooms", 1, 10, 3)

bathrooms = st.slider("Bathrooms", 1, 10, 2)

garage = st.slider("Garage Spaces", 0, 5, 1)

months_listed = st.slider("Months Listed", 1, 24, 6)

# ----------------------------
# Create Input Data
# ----------------------------
input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'garage': [garage],
    'months_listed': [months_listed]
})

# ----------------------------
# Match Training Columns
# ----------------------------
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]

# ----------------------------
# Prediction
# ----------------------------
prediction = model.predict(input_data)

# ----------------------------
# Display Result
# ----------------------------
st.subheader("Predicted House Price")

st.success(f"${prediction[0]:,.2f}")

# ----------------------------
# Business Recommendation
# ----------------------------
st.subheader("Business Recommendation")

if prediction[0] > 300000:
    st.write("This property is classified as a high-value investment property.")

elif prediction[0] > 150000:
    st.write("This property is classified as a medium-value residential property.")

else:
    st.write("This property is classified as a budget-friendly property.")