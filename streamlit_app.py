import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# Define custom model class (needed for loading AveragingModel)
class AveragingModel(BaseEstimator, RegressorMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def predict(self, X):
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        return (pred1 + pred2) / 2

# Load model, scaler, and feature names
model = joblib.load("clv_predictor_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("model_features.pkl")  # <-- ensure column alignment

# Streamlit UI
st.title("Customer Lifetime Value Predictor")

vehicle_class = st.selectbox("Vehicle Class", ['Two-Door Car', 'Four-Door Car', 'SUV', 'Luxury Car', 'Sports Car'])
coverage = st.selectbox("Coverage", ['Basic', 'Extended', 'Premium'])
renew_offer = st.selectbox("Renew Offer Type", ['Offer1', 'Offer2', 'Offer3', 'Offer4'])
employment_status = st.selectbox("Employment Status", ['Employed', 'Unemployed', 'Medical Leave', 'Retired', 'Disabled'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
education = st.selectbox("Education", ['High School', 'College', 'Bachelor', 'Master', 'Doctor'])

num_policies = st.slider("Number of Policies", 1, 10, 2)
monthly_premium = st.slider("Monthly Premium Auto", 10, 200, 85)
claim_amount = st.slider("Total Claim Amount", 0, 2000, 500)
income = st.slider("Income", 0, 200000, 60000)

# Prepare input
input_dict = {
    'Vehicle Class': [vehicle_class],
    'Coverage': [coverage],
    'Renew Offer Type': [renew_offer],
    'Employment Status': [employment_status],
    'Marital Status': [marital_status],
    'Education': [education],
    'Number of Policies': [num_policies],
    'Monthly Premium Auto': [monthly_premium],
    'Total Claim Amount': [claim_amount],
    'Income': [income]
}
input_df = pd.DataFrame(input_dict)

# Preprocessing
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=feature_names, fill_value=0)

numeric_cols = ['Income', 'Monthly Premium Auto', 'Total Claim Amount', 'Number of Policies']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Predict
if st.button("Predict CLV"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Customer Lifetime Value: ${prediction:,.2f}")
