import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

# --- Load the lightweight scikit-learn model and scaler ---
# This uses very little memory and is perfect for hosting.
try:
    model = joblib.load('model/failure_predictor.pkl')
    scaler = joblib.load('model/scaler.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Please ensure 'failure_predictor.pkl' and 'scaler.pkl' are in the 'model' directory in your GitHub repository.")
    st.stop()


# --- App Title and Description ---
st.title("🔧 Smart Predictive Maintenance Dashboard")
st.markdown("Predict **machine failure** by providing real-time sensor data.")


# --- Sidebar for User Input ---
st.sidebar.header("Input Sensor Data")

# NOTE: This model was trained on fewer features. We match the inputs to the model.
air_temp = st.sidebar.slider('Air temperature [K]', 295.0, 305.0, 300.1, 0.1)
rpm = st.sidebar.slider("Rotational Speed [rpm]", 1000.0, 3000.0, 1500.0)
torque = st.sidebar.slider("Torque [Nm]", 0.0, 100.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0.0, 250.0, 50.0)

# --- Prediction Logic ---
if st.sidebar.button("Predict Failure"):
    # Prepare input data in the same way the model was trained
    input_data = np.array([[air_temp, rpm, torque, tool_wear]])
    input_scaled = scaler.transform(input_data)

    # Make prediction and get probability
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display the result
    st.subheader("🧠 Prediction Result")
    if prediction == 1:
        st.error(f"🚨 High Risk: Machine Failure Predicted! (Risk Score: {probability:.2%})", icon="🚨")
    else:
        st.success(f"✅ Low Risk: No Failure Predicted. (Confidence: {(1 - probability):.2%})", icon="✅")


# --- Footer ---
st.markdown("---")
st.markdown("Model trained on the **AI4I 2020 Dataset** using a Random Forest Classifier.")

