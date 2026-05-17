import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# PAGE TITLE
st.title("🏠 Advanced House Price Prediction App")

st.write(
    "Predict house prices using bedrooms, area, months listed, city, and house type."
)

# LOAD DATA
df = pd.read_csv("data/house_sales.csv")

# CLEAN AREA COLUMN
df["area"] = (
    df["area"]
    .astype(str)
    .str.replace("sq.m.", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["area"] = pd.to_numeric(df["area"], errors="coerce")

# CONVERT MONTHS LISTED
df["months_listed"] = pd.to_numeric(df["months_listed"], errors="coerce")

# DROP MISSING VALUES
df = df.dropna(subset=["months_listed", "bedrooms", "area", "sale_price"])

# ENCODE CATEGORICAL VARIABLES
city_encoder = LabelEncoder()
house_encoder = LabelEncoder()

df["city_encoded"] = city_encoder.fit_transform(df["city"])
df["house_encoded"] = house_encoder.fit_transform(df["house_type"])

# FEATURES
X = df[
    [
        "months_listed",
        "bedrooms",
        "area",
        "city_encoded",
        "house_encoded"
    ]
]

# TARGET
y = df["sale_price"]

# TRAIN MODEL
model = LinearRegression()
model.fit(X, y)

# SIDEBAR
st.sidebar.header("Enter House Details")

months_listed = st.sidebar.slider(
    "Months Listed",
    1.0,
    12.0,
    6.0
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

area = st.sidebar.number_input(
    "Area (sq.m.)",
    min_value=50.0,
    max_value=1000.0,
    value=250.0
)

city = st.sidebar.selectbox(
    "City",
    df["city"].unique()
)

house_type = st.sidebar.selectbox(
    "House Type",
    df["house_type"].unique()
)

# ENCODE USER INPUT
city_encoded = city_encoder.transform([city])[0]
house_encoded = house_encoder.transform([house_type])[0]

# PREDICTION
prediction = model.predict([[
    months_listed,
    bedrooms,
    area,
    city_encoded,
    house_encoded
]])

# DISPLAY RESULT
st.subheader("Predicted House Price")

st.success(f"${prediction[0]:,.2f}")

# SHOW DATA
st.subheader("Dataset Preview")
st.dataframe(df.head())