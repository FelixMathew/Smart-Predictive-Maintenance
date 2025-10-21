# 🛠️ Smart Predictive Maintenance for Industrial Equipment
🔗 **Live App**: [Advanced IoT Big Data Project: Smart Predictive Maintenance for Industrial Equipment (Streamlit)](https://smart-predictive-maintenance-sdxrkuxnhzxywrx3bmgncb.streamlit.app/)

----

By **Felix Mathew** (RA2311028020016)  
Member of **SRM Institute of Science and Technology**

---

#📌 Project Overview

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

| **Layer**             | **Big Data Training (Local)**       | **Web App Deployment (Cloud)**            |
| --------------------- | ----------------------------------- | ----------------------------------------- |
| **Language**          | Python                              | Python                                    |
| **Core Framework**    | Apache Spark (PySpark)              | Streamlit                                 |
| **Data Handling**     | Spark DataFrame API, Pandas, NumPy  | Pandas, NumPy                             |
| **ML Model**          | `pyspark.ml.RandomForestClassifier` | `sklearn.ensemble.RandomForestClassifier` |
| **Model Persistence** | Spark ML Format                     | Joblib (.pkl)                             |
| **Environment**       | Java 11/17, Hadoop WinUtils, venv   | Streamlit Community Cloud                 |
| **IDE & OS**          | Visual Studio Code (Windows)        | N/A                                       |


---

 🧪 Project Components

🔹 1. PySpark Training & Prediction Pipeline

Directory: /scripts
Files:
pyspark_train_model.py
Initializes a SparkSession
Loads and preprocesses the dataset
Trains a RandomForestClassifier using Spark MLlib
Achieves ~97% accuracy on the test set
Saves model to /model/pyspark_rf_model/
predict.py
Loads the trained PySpark model
Predicts failure probabilities for new batch data
---

## 🧠 How It Works

1. User enters tool wear, torque, and RPM.
2. The app scales the input using the same `StandardScaler` used in training.
3. The trained `RandomForestClassifier` model predicts failure risk.
4. Displays prediction result instantly on the web interface.

---
 # 🚀 How to Run Locally
🔸 Step 1: Run the PySpark Training Pipeline
Prerequisites
Java JDK 11 or 17
Hadoop WinUtils configured (HADOOP_HOME set)
Commands:
Activate virtual environment
.\venv\Scripts\Activate.ps1
Run training script
python scripts/pyspark_train_model.py

🔸 Step 2: Run the Streamlit Web App

Commands:
Activate virtual environment
.\venv\Scripts\Activate.ps1
Launch Streamlit app
streamlit run app.py
---
