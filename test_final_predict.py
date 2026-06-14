import pandas as pd
import joblib
import numpy as np

# ================================
# LOAD MODELS + SCALER + FEATURES
# ================================
binary_model = joblib.load("models/rf_binary.pkl")
multi_model = joblib.load("models/rf_multiclass.pkl")
scaler = joblib.load("models/supervised_scaler.pkl")
model_features = joblib.load("models/model_features.pkl")

# ================================
# PREPROCESS LIKE TRAINING PIPELINE
# ================================
def preprocess_new_file(df):
    df = df.copy()

    # 1️⃣ Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 2️⃣ Generate missing numeric features (same as training)
    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Some datasets have flow_duration missing → create fallback
    if "flow_duration" not in df.columns:
        df["flow_duration"] = 0.001

    # Make these safe (avoid division by zero)
    df["flow_duration"] = df["flow_duration"].replace(0, 0.001)

    # ⏳ FWD/BWD pkts per second
    if "flow_pkts_per_sec" not in df.columns:
        df["flow_pkts_per_sec"] = (
            df.get("fwd_pkts_tot", 0) + df.get("bwd_pkts_tot", 0)
        ) / df["flow_duration"]

    if "fwd_pkts_per_sec" not in df.columns:
        df["fwd_pkts_per_sec"] = df.get("fwd_pkts_tot", 0) / df["flow_duration"]

    if "bwd_pkts_per_sec" not in df.columns:
        df["bwd_pkts_per_sec"] = df.get("bwd_pkts_tot", 0) / df["flow_duration"]

    # Payload per second
    if "payload_bytes_per_second" not in df.columns:
        df["payload_bytes_per_second"] = (
            df.get("payload_bytes", 0) / df["flow_duration"]
        )

    # 3️⃣ Fill missing numeric values
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # 4️⃣ Add missing model features (model_features = 30 features)
    for f in model_features:
        if f not in df.columns:
            df[f] = 0

    # 5️⃣ Extract only model features
    df_final = df[model_features]

    # 6️⃣ Ensure numeric
    df_final = df_final.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df_final

# ================================
# MAIN FUNCTION FOR ANY CSV
# ================================
def predict_any_csv(path):
    print("\n📥 Loading file:", path)
    df = pd.read_csv(path)

    print("⚙ Preprocessing file (same as training)...")
    X = preprocess_new_file(df)

    print("📏 Scaling...")
    X_scaled = scaler.transform(X)

    print("🤖 Predicting...")
    df["binary_pred"] = binary_model.predict(X_scaled)
    df["multi_pred"] = multi_model.predict(X_scaled)

    df["predicted_label"] = df["binary_pred"].apply(
        lambda x: "Attack" if x == 1 else "Normal"
    )

    print("\n🎯 RESULT SUMMARY:")
    print(df["predicted_label"].value_counts())
    print("\nDetailed:")
    print(df[["binary_pred", "multi_pred"]].value_counts())

    return df

# ================================
# RUN TEST
# ================================
if __name__ == "__main__":
    result = predict_any_csv("network_detection_test_realistic.csv")
    result.to_csv("real_test_predictions.csv", index=False)
    print("\n💾 Saved: real_test_predictions.csv")
