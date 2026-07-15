🚗 Car Price Prediction Using Machine Learning

A Machine Learning project that predicts the estimated selling price of used cars based on various vehicle features.

---

📌 Project Overview

This project uses Machine Learning techniques to predict the selling price of cars. It includes data preprocessing, exploratory data analysis (EDA), model training, model evaluation, and deployment using Streamlit.

The application allows users to enter car details and receive an estimated selling price instantly.

---

🎯 Objectives

- Analyze used car data
- Perform data preprocessing
- Build and compare Machine Learning models
- Predict car selling prices
- Deploy the model using Streamlit

---

🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

🔄 Project Workflow

Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Data Encoding
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Streamlit Deployment

---

📊 Exploratory Data Analysis

- Selling Price Distribution
- Scatter Plot Analysis
- Correlation Analysis

---

🤖 Machine Learning Models

- Linear Regression
- Random Forest Regressor

Input Features

- Car Brand
- Manufacturing Year
- Fuel Type
- Transmission
- Mileage
- Engine
- Max Power
- Seller Type
- Owner Type

Target Variable

- Selling Price

---

⚙️ Data Preprocessing

- Removed unnecessary columns
- Handled missing values
- Encoded categorical features
- Split data into training and testing sets
- Saved model and encoders using Joblib

---

🌐 Streamlit Web Application

Features:

- Enter car details
- Select fuel type and transmission
- Predict the estimated selling price

---

📂 Project Structure

Car-Price-Prediction/

├── app.py
├── car_price_model.pkl
├── brand_encoder.pkl
├── fuel_encoder.pkl
├── seller_encoder.pkl
├── owner_encoder.pkl
├── transmission_encoder.pkl
├── requirements.txt
├── README.md
└── dataset/
    └── cardekho.xlsx

---

🚀 How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

---

👨‍💻 Author

Saumya Shukla

BCA Student | Machine Learning Enthusiast
