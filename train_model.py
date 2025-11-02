import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# Load dataset
df = pd.read_csv("data/UCI_Credit_Card.csv")

# Features and target
X = df.drop("default.payment.next.month", axis=1)
y = df["default.payment.next.month"]

# Split and scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {acc*100:.2f}%")

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/trained_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model and scaler saved in /model folder.")
