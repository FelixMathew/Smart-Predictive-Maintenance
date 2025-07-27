import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load dataset
df = pd.read_csv('../data/ai4i2020.csv')

# Set target
df['Target'] = df['Machine failure']

# Select features
features = ['Tool wear [min]', 'Torque [Nm]', 'Rotational speed [rpm]']
X = df[features]
y = df['Target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split & train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & scaler
os.makedirs('../model', exist_ok=True)
joblib.dump(model, '../model/failure_predictor.pkl')
joblib.dump(scaler, '../model/scaler.pkl')

print("✅ Model training complete and saved!")
