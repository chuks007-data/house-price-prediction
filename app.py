import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# PAGE TITLE

st.title("🏠 House Price Prediction App")

st.write(
    "This app predicts house prices based on the number of bedrooms."
)


# LOAD DATA

df = pd.read_csv("data/house_sales.csv")


# SELECT FEATURES

X = df[['bedrooms']]
y = df['sale_price']


# REMOVE MISSING VALUES

X = X.dropna()
y = y.loc[X.index]


# TRAIN MODEL

model = LinearRegression()
model.fit(X, y)


# USER INPUT

st.sidebar.header("Enter House Details")

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)


# PREDICTION

input_data = np.array([[bedrooms]])

prediction = model.predict(input_data)


# DISPLAY RESULT

st.subheader("Predicted House Price")

st.success(f"${prediction[0]:,.2f}")


# SHOW DATASET

st.subheader("Dataset Preview")

st.dataframe(df.head())