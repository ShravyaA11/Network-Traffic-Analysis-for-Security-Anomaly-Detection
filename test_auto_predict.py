import os
import glob
import pandas as pd
import joblib
import numpy as np

# ================================
# LOAD MODELS + SCALER + FEATURES
# ================================
print("🔄 Loading Models...")

binary_model = joblib.load("models/rf_binary.pkl")
multi_model = joblib.load("models/rf_multiclass.pkl")
scaler = joblib.load("models/supervised_scaler.pkl")
model_features = joblib.load("models/model_features.pkl")

print("✅ Models Loaded Successfully!")


# ================================
# AUTO-DETECT LAST UPLOADED FILE
# ================================
def get_latest_csv(folder="uploads"):
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise FileNotFoundError("❌ No CSV file found in uploads/ folder.")
    latest = max(files, key=os.path.getctime)
    print("📌 Auto-selected CSV:", latest)
    return latest


# ================================
# PREPROCESS DATA LIKE TRAINING
# ================================
def preprocess_new_file(df):
    df = df.copy()

    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Convert numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Ensure flow_duration exists
    if "flow_duration" not in df.columns:
        df["flow_duration"] = 0.001

    df["flow_duration"] = df["flow_duration"].replace(0, 0.001)

    # Create engineered features (same as training)
    if "flow_pkts_per_sec" not in df.columns:
        df["flow_pkts_per_sec"] = (
            df.get("fwd_pkts_tot", 0) + df.get("bwd_pkts_tot", 0)
        ) / df["flow_duration"]

    if "fwd_pkts_per_sec" not in df.columns:
        df["fwd_pkts_per_sec"] = df.get("fwd_pkts_tot", 0) / df["flow_duration"]

    if "bwd_pkts_per_sec" not in df.columns:
        df["bwd_pkts_per_sec"] = df.get("bwd_pkts_tot", 0) / df["flow_duration"]

    if "payload_bytes_per_second" not in df.columns:
        df["payload_bytes_per_second"] = (
            df.get("payload_bytes", 0) / df["flow_duration"]
        )

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Add model missing features
    for f in model_features:
        if f not in df.columns:
            df[f] = 0

    # Keep only model features
    df_final = df[model_features]

    # Ensure numeric
    df_final = df_final.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df_final


# ================================
# MAIN PREDICTION FUNCTION
# ================================
def predict_any_csv(path):
    print("\n📥 Loading file:", path)
    df = pd.read_csv(path)

    print("⚙ Preprocessing file...")
    X = preprocess_new_file(df)

    print("📏 Scaling...")
    X_scaled = scaler.transform(X)

    print("🤖 Predicting...")
    df["binary_pred"] = binary_model.predict(X_scaled)
    df["multi_pred"] = multi_model.predict(X_scaled)
    df["predicted_label"] = df["binary_pred"].apply(
        lambda x: "Attack" if x == 1 else "Normal"
    )

    print("\n🎯 SUMMARY:")
    print(df["predicted_label"].value_counts())
    print("\nDetailed:")
    print(df[["binary_pred", "multi_pred"]].value_counts())

    return df


# ================================
# AUTO-RUN PREDICTION PIPELINE
# ================================
if __name__ == "__main__":
    latest_file = get_latest_csv("uploads")
    result = predict_any_csv(latest_file)

    output = "real_test_predictions.csv"
    result.to_csv(output, index=False)

    print("\n💾 Saved:", output)
    print("🎉 DONE!")
