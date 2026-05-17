import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv('data/house_sales.csv')

# -----------------------------
# Data Cleaning
# -----------------------------

clean_data = df.copy()

# City cleaning
clean_data['city'] = clean_data['city'].replace('--', 'Unknown')

# House type cleaning
clean_data['house_type'] = clean_data['house_type'].replace({
    'Det.': 'Detached',
    'Semi': 'Semi-detached',
    'Terr.': 'Terraced'
})

# Area cleaning
clean_data['area'] = (
    clean_data['area']
    .str.replace(' sq.m.', '', regex=False)
)

clean_data['area'] = pd.to_numeric(
    clean_data['area'],
    errors='coerce'
)

clean_data['area'] = clean_data['area'].fillna(
    clean_data['area'].mean()
)

# Bedrooms
clean_data['bedrooms'] = pd.to_numeric(
    clean_data['bedrooms'],
    errors='coerce'
)

clean_data['bedrooms'] = clean_data['bedrooms'].fillna(
    clean_data['bedrooms'].mean()
)

# Months listed
clean_data['months_listed'] = pd.to_numeric(
    clean_data['months_listed'],
    errors='coerce'
)

clean_data['months_listed'] = clean_data['months_listed'].fillna(
    clean_data['months_listed'].mean()
)

# Sale price
clean_data['sale_price'] = pd.to_numeric(
    clean_data['sale_price'],
    errors='coerce'
)

clean_data = clean_data.dropna(subset=['sale_price'])

# -----------------------------
# Feature Engineering
# -----------------------------

model_data = pd.get_dummies(
    clean_data,
    columns=['city', 'house_type'],
    drop_first=True
)

X = model_data.drop(
    columns=['sale_price', 'sale_date']
)

y = model_data['sale_price']

# -----------------------------
# Train model
# -----------------------------

model = LinearRegression()

model.fit(X, y)

# -----------------------------
# Streamlit App
# -----------------------------

st.title("House Price Prediction App")

st.write(
    "Predict house sale prices using machine learning."
)

# User inputs

area = st.number_input(
    "Area (sq.m.)",
    min_value=10.0,
    value=100.0
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    value=3
)

months_listed = st.number_input(
    "Months Listed",
    min_value=1.0,
    value=5.0
)

city = st.selectbox(
    "City",
    ['Riverford', 'Silvertown', 'Teasdale', 'Poppleton']
)

house_type = st.selectbox(
    "House Type",
    ['Detached', 'Semi-detached', 'Terraced']
)

# -----------------------------
# Prepare input data
# -----------------------------

input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'months_listed': [months_listed],
    'city_Riverford': [1 if city == 'Riverford' else 0],
    'city_Silvertown': [1 if city == 'Silvertown' else 0],
    'city_Teasdale': [1 if city == 'Teasdale' else 0],
    'house_type_Semi-detached': [
        1 if house_type == 'Semi-detached' else 0
    ],
    'house_type_Terraced': [
        1 if house_type == 'Terraced' else 0
    ]
})

# Ensure correct column order
input_data = input_data.reindex(
    columns=X.columns,
    fill_value=0
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict House Price"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}"
    )