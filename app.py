import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# PAGE TITLE

st.title("🏠 House Price Prediction App")

st.write(
    "This app predicts house prices based on "
    "months listed, number of bedrooms, and area."
)


# LOAD DATA

df = pd.read_csv("data/house_sales.csv")


# SELECT FEATURES

X = df[['months_listed', 'bedrooms', 'area']]
y = df['sale_price']


# REMOVE MISSING VALUES

X = X.dropna()
y = y.loc[X.index]


# TRAIN MODEL

model = LinearRegression()
model.fit(X, y)


# USER INPUTS

st.sidebar.header("Enter House Details")

months_listed = st.sidebar.number_input(
    "Months Listed",
    min_value=0.0,
    value=3.0
)

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    value=3
)

area = st.sidebar.number_input(
    "Area (sq ft)",
    min_value=500,
    value=1500
)


# PREDICTION

input_data = np.array([[months_listed, bedrooms, area]])

prediction = model.predict(input_data)


# DISPLAY RESULT
st.subheader("Predicted House Price")

st.success(f"${prediction[0]:,.2f}")


# SHOW DATASET
st.subheader("Dataset Preview")

st.dataframe(df.head())