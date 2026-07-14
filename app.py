import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Saved Model and Files
# -----------------------------

model = joblib.load("car_price_model.pkl")

brand_encoder = joblib.load("brand_encoder.pkl")
fuel_encoder = joblib.load("fuel_encoder.pkl")
seller_encoder = joblib.load("seller_encoder.pkl")
owner_encoder = joblib.load("owner_encoder.pkl")
transmission_encoder = joblib.load("transmission_encoder.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗"
)

st.title("🚗 Car Price Prediction System")

st.write(
    "Enter car details to predict the estimated selling price."
)


# -----------------------------
# User Input
# -----------------------------

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2015
)

km_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=50000
)

fuel = st.selectbox(
    "Fuel Type",
    fuel_encoder.classes_
)

seller_type = st.selectbox(
    "Seller Type",
    seller_encoder.classes_
)

transmission = st.selectbox(
    "Transmission",
    transmission_encoder.classes_
)

owner = st.selectbox(
    "Owner Type",
    owner_encoder.classes_
)

mileage = st.number_input(
    "Mileage",
    value=20.0
)

engine = st.number_input(
    "Engine",
    value=1200.0
)

max_power = st.number_input(
    "Max Power",
    value=80.0
)

brand = st.selectbox(
    "Car Brand",
    brand_encoder.classes_
)


# Calculate car age
car_age = 2026 - year


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Selling Price"):

    try:

        # Encoding categorical values
        fuel = fuel_encoder.transform([fuel])[0]
        seller_type = seller_encoder.transform([seller_type])[0]
        transmission = transmission_encoder.transform([transmission])[0]
        owner = owner_encoder.transform([owner])[0]
        brand = brand_encoder.transform([brand])[0]


        input_data = [[
            year,
            km_driven,
            fuel,
            seller_type,
            transmission,
            owner,
            mileage,
            engine,
            max_power,
            brand,
            car_age
        ]]


        # Scaling input
        input_data = scaler.transform(input_data)


        # Random Forest Prediction
        prediction = model.predict(input_data)


        st.success(
            f"Estimated Selling Price: ₹ {prediction[0]:,.0f}"
        )

    except Exception as e:
        st.error(e)