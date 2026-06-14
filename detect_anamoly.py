# detect_anomaly.py
# Unsupervised Anomaly Detection + Save Model

import os
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("\n🚀 Starting Anomaly Detection Module...\n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "analysis", "selected_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------
# Load Data
# -----------------------------------------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_FILE)
print("Dataset Loaded:", df.shape)

# Remove labels
X = df.drop(columns=["label_binary", "label_multi"], errors="ignore")

# -----------------------------------------------------------
# Scale features
# -----------------------------------------------------------
print("\n🔄 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------
# Train Isolation Forest
# -----------------------------------------------------------
print("\n🌲 Training Isolation Forest (Unsupervised)...")

iso = IsolationForest(
    n_estimators=150,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

iso.fit(X_scaled)

print("🎯 Training complete!")

# -----------------------------------------------------------
# Save the model (IMPORTANT)
# -----------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "anomaly_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "anomaly_scaler.pkl")

joblib.dump(iso, model_path)
joblib.dump(scaler, scaler_path)

print(f"💾 Anomaly Model saved at: {model_path}")
print(f"💾 Scaler saved at: {scaler_path}")

# -----------------------------------------------------------
# Predict anomalies
# -----------------------------------------------------------
print("\n🔍 Detecting anomalies...")
preds = iso.predict(X_scaled)

df["anomaly_flag"] = preds
df["anomaly_flag"] = df["anomaly_flag"].map({1: "normal", -1: "anomaly"})

# Save output
OUT_FILE = os.path.join(BASE_DIR, "analysis", "anomaly_detection_output.csv")
df.to_csv(OUT_FILE, index=False)

print(f"\n💾 Results saved to: {OUT_FILE}")

total = len(df)
anomalies = sum(df["anomaly_flag"] == "anomaly")

print("\n📌 SUMMARY")
print("Total records:", total)
print("Anomalies found:", anomalies)
print("Normal records:", total - anomalies)

print("\n🎉 Anomaly Detection Completed Successfully!")
