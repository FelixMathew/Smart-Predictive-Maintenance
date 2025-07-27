import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('model/failure_predictor.pkl')
scaler = joblib.load('model/scaler.pkl')

# Streamlit app title
st.title("🔧 Smart Predictive Maintenance Dashboard")
st.markdown("Predict **machine failure** using Tool Wear, Torque, and Rotational Speed.")

# Sidebar for input
st.sidebar.header("🔍 Input Sensor Data")

tool_wear = st.sidebar.slider("Tool Wear [min]", 0.0, 250.0, 50.0)
torque = st.sidebar.slider("Torque [Nm]", 0.0, 100.0, 40.0)
rot_speed = st.sidebar.slider("Rotational Speed [rpm]", 1000.0, 3000.0, 1500.0)

# Prepare input for model
input_data = np.array([[tool_wear, torque, rot_speed]])
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Output
st.subheader("🧠 Prediction Result")
if prediction == 1:
    st.error(f"⚠️ Machine Failure Likely (Risk Score: {probability:.2f})")
else:
    st.success(f"✅ No Failure Predicted (Confidence: {1 - probability:.2f})")

# Extra info
st.markdown("---")
st.markdown("Model trained on **AI4I 2020 Predictive Maintenance Dataset** using Random Forest Classifier.")
