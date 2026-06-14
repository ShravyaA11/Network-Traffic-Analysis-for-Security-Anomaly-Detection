import pandas as pd
import joblib

# ================================
# LOAD MODELS
# ================================
binary_model = joblib.load("models/rf_binary.pkl")
multi_model = joblib.load("models/rf_multiclass.pkl")
scaler = joblib.load("models/supervised_scaler.pkl")
model_features = joblib.load("models/model_features.pkl")  # List of 30 features

# ================================
# LOAD TEST CSV
# ================================
df = pd.read_csv("real_test.csv")   # <-- Change filename if needed

# Clean column names
df.columns = [c.lower().strip() for c in df.columns]

# ================================
# MATCH FEATURES
# ================================
# Add missing features
for f in model_features:
    if f not in df.columns:
        df[f] = 0  # FILL MISSING FEATURE

# Ensure numeric
X = df[model_features].apply(pd.to_numeric, errors="coerce").fillna(0)

# ================================
# SCALE INPUT
# ================================
X_scaled = scaler.transform(X)

# ================================
# PREDICT
# ================================
df["binary_pred"] = binary_model.predict(X_scaled)
df["multi_pred"] = multi_model.predict(X_scaled)

# ================================
# PRINT SUMMARY
# ================================
print(df[["binary_pred", "multi_pred"]].value_counts())
print("\nSample predictions:")
print(df[["binary_pred", "multi_pred"]].head(20))
