import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# PAGE CONFIG


st.set_page_config(
    page_title="Smart House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# TITLE


st.title("🏠 Smart House Price Prediction System")

st.write(
    """
    Predict house prices using:
    - Bedrooms
    - Area
    - Months Listed
    - City
    - House Type
    """
)


# LOAD DATA


df = pd.read_csv("data/house_sales.csv")


# CLEAN DATA


# Clean area column
df["area"] = (
    df["area"]
    .astype(str)
    .str.replace("sq.m.", "", regex=False)
    .str.replace(",", "", regex=False)
)

# Convert columns to numeric
df["area"] = pd.to_numeric(df["area"], errors="coerce")
df["months_listed"] = pd.to_numeric(df["months_listed"], errors="coerce")
df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")

# Remove missing values
df = df.dropna()


# ENCODE CATEGORICAL VARIABLES


city_encoder = LabelEncoder()
house_encoder = LabelEncoder()

df["city_encoded"] = city_encoder.fit_transform(df["city"])
df["house_encoded"] = house_encoder.fit_transform(df["house_type"])


# FEATURES AND TARGET


X = df[
    [
        "months_listed",
        "bedrooms",
        "area",
        "city_encoded",
        "house_encoded"
    ]
]

y = df["sale_price"]


# SPLIT DATA


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# TRAIN MODEL


model = LinearRegression()

model.fit(X_train, y_train)


# MODEL EVALUATION


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 
# SIDEBAR INPUTS

st.sidebar.header("Enter House Details")

months_listed = st.sidebar.slider(
    "Months Listed",
    min_value=1.0,
    max_value=12.0,
    value=6.0
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

area = st.sidebar.number_input(
    "Area (sq.m.)",
    min_value=50.0,
    max_value=1000.0,
    value=250.0
)

city = st.sidebar.selectbox(
    "City",
    sorted(df["city"].unique())
)

house_type = st.sidebar.selectbox(
    "House Type",
    sorted(df["house_type"].unique())
)


# ENCODE USER INPUT


city_encoded = city_encoder.transform([city])[0]
house_encoded = house_encoder.transform([house_type])[0]


# MAKE PREDICTION


prediction = model.predict([[
    months_listed,
    bedrooms,
    area,
    city_encoded,
    house_encoded
]])


# DISPLAY PREDICTION


st.subheader("Predicted House Price")

st.success(f"${prediction[0]:,.2f}")


# MODEL PERFORMANCE


st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Mean Absolute Error",
        value=f"${mae:,.2f}"
    )

with col2:
    st.metric(
        label="R² Score",
        value=f"{r2:.2f}"
    )


# CHARTS

st.subheader("House Price Distribution")

st.bar_chart(df["sale_price"])

st.subheader("Bedrooms vs Sale Price")

chart_data = df[["bedrooms", "sale_price"]]

st.scatter_chart(chart_data)


# DATASET PREVIEW


st.subheader("Dataset Preview")

st.dataframe(df.head(10))


# FOOTER


st.markdown("---")

st.write("Developed by Chukwuka Odey")