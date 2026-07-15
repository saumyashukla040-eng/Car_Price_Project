import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

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
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


# -----------------------------
# Dark Theme
# -----------------------------

st.markdown("""
<style>

.stApp{
    background-color:#0d0d0d;
    color:white;
}

section[data-testid="stSidebar"]{
    background:#111111;
}

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

.stButton>button{
    width:100%;
    background:#8b5cf6;
    color:white;
    border-radius:12px;
    height:55px;
    font-size:20px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:#6d28d9;
}

div[data-baseweb="select"]>div{
    background:#1b1b1b;
    color:white;
    border:1px solid #8b5cf6;
}

.stNumberInput input{
    background:#1b1b1b;
    color:white;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("🚗 Car Price")
    st.subheader("Prediction")

    st.markdown("---")

    st.subheader("ℹ About This App")
    st.write(
        "Predict the estimated selling price of a used car using Machine Learning."
    )

    st.markdown("---")

    st.subheader("⭐ Features")
    st.write("✔ Accurate Prediction")
    st.write("✔ Easy to Use")
    st.write("✔ Machine Learning Model")
    st.write("✔ Streamlit Web App")

    st.markdown("---")

    st.subheader("👨‍💻 Developer")
    st.write("Saumya Shukla")


# -----------------------------
# Main Title
# -----------------------------

st.markdown(
    "<h1 style='text-align:center;'>🚗 Car Price Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align:center;color:#b3b3b3;'>Using Machine Learning</h3>",
    unsafe_allow_html=True
)

st.info("Enter car details to predict the estimated selling price.")


# -----------------------------
# User Inputs
# -----------------------------

col1, col2 = st.columns(2)


with col1:

    year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=2026,
        value=2015
    )

    fuel = st.selectbox(
        "Fuel Type",
        fuel_encoder.classes_
    )

    transmission = st.selectbox(
        "Transmission",
        transmission_encoder.classes_
    )

    mileage = st.number_input(
        "Mileage",
        value=20.0
    )

    max_power = st.number_input(
        "Max Power",
        value=80.0
    )


with col2:

    brand = st.selectbox(
        "Car Brand",
        brand_encoder.classes_
    )

    seller_type = st.selectbox(
        "Seller Type",
        seller_encoder.classes_
    )

    owner = st.selectbox(
        "Owner Type",
        owner_encoder.classes_
    )

    engine = st.number_input(
        "Engine",
        value=1200.0
    )

    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=50000
    )


car_age = datetime.now().year - year
# -----------------------------
# Prediction
# -----------------------------

if st.button("🚗 Predict Selling Price"):

    try:

        # Encoding categorical values
        fuel = fuel_encoder.transform([fuel])[0]
        seller_type = seller_encoder.transform([seller_type])[0]
        transmission = transmission_encoder.transform([transmission])[0]
        owner = owner_encoder.transform([owner])[0]
        brand = brand_encoder.transform([brand])[0]


        # Create DataFrame
        input_data = pd.DataFrame([{
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage(km/ltr/kg)': mileage,
            'engine': engine,
            'max_power': max_power,
            'brand': brand,
            'car_age': car_age
        }])


        # Scale numerical columns
        num_cols = [
            'mileage(km/ltr/kg)',
            'engine',
            'max_power',
            'car_age',
            'km_driven'
        ]

        input_data[num_cols] = scaler.transform(input_data[num_cols])


        # Prediction
        prediction = model.predict(input_data)


        # Show Result
        st.markdown("---")

        st.success("✅ Prediction Completed Successfully!")

        st.write(f"₹ {prediction[0]:,.0f}")


    except Exception as e:
        st.error(f"Error: {e}")