# 🛠️ Smart Predictive Maintenance for Industrial Equipment

By **Felix Mathew** (RA2311028020016)  
Member of **SRM Institute of Science and Technology**

---

## 📌 Project Overview

A machine learning-driven predictive maintenance system designed to detect potential machine failures in industrial environments using key parameters like tool wear, torque, and rotational speed. This project uses real-time simulation, data preprocessing, and model training to build a failure prediction pipeline.

---

## 🎯 Objective

To reduce unplanned downtime and maintenance costs by predicting failures **before they occur** using IoT-style sensor data and machine learning classification models.

---

## 📊 Dataset

- **Name**: AI4I 2020 Predictive Maintenance Dataset  
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/505/ai4i+2020+predictive+maintenance+dataset)  
- **Features Used**:
  - Tool wear [min]
  - Torque [Nm]
  - Rotational speed [rpm]
  - Machine failure (Target)

---

## ⚙️ Tech Stack

| Layer                | Tools / Libraries                             |
|---------------------|------------------------------------------------|
| Language            | Python                                         |
| Data Handling       | Pandas, NumPy                                  |
| Preprocessing       | Scikit-learn (StandardScaler)                  |
| ML Model            | RandomForestClassifier                         |
| Visualization / UI  | Streamlit                                      |
| Model Persistence   | Joblib                                         |
| IDE                 | Visual Studio Code                             |
| Environment         | venv (Python Virtual Environment)              |
| OS                  | Windows 10                                     |

---

## 🧪 Project Components

### 🔹 Data Preprocessing

- Feature selection: Tool wear, Torque, Rotational speed
- Scaling using `StandardScaler`
- Splitting: 80% training / 20% testing

### 🔹 Machine Learning Model

- Model: `RandomForestClassifier` with 100 estimators
- Accuracy optimized through train-test splitting
- Saved using `joblib` into `/model/` directory

### 🔹 Streamlit Web App

- User inputs 3 key parameters
- Predicts whether failure is likely (`Yes`/`No`)
- Simple UI for demo and interaction

---

## 🧠 How It Works

1. User enters tool wear, torque, and RPM.
2. The app scales the input using the same `StandardScaler` used in training.
3. The trained `RandomForestClassifier` model predicts failure risk.
4. Displays prediction result instantly on the web interface.

---
