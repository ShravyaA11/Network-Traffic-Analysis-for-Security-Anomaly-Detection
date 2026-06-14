# # File: ml_model.py
# # FINAL FIXED VERSION — WITH SUPERVISED SCALER + MODEL FEATURES SAVED

# import os
# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# print("\n🚀 Starting ML Training (BINARY + MULTI-CLASS WITH SCALING)...\n")

# # -----------------------------------------------------------
# # Paths
# # -----------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# INPUT_FILE = os.path.join(BASE_DIR, "analysis", "selected_features.csv")
# OUTPUT_DIR = os.path.join(BASE_DIR, "models")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -----------------------------------------------------------
# # Load dataset
# # -----------------------------------------------------------
# print("📥 Loading dataset...")
# df = pd.read_csv(INPUT_FILE)
# print("✅ Loaded:", df.shape)

# # -----------------------------------------------------------
# # Input Features + Labels
# # -----------------------------------------------------------
# X_full = df.drop(columns=["label_binary", "label_multi"], errors="ignore")
# y_binary = df["label_binary"]
# y_multi = df["label_multi"]

# # Save feature list for Flask usage
# model_features = list(X_full.columns)
# joblib.dump(model_features, os.path.join(OUTPUT_DIR, "model_features.pkl"))
# print(f"📌 Saved {len(model_features)} model features.")

# # -----------------------------------------------------------
# # STRATIFIED SAMPLING (~700k rows)
# # -----------------------------------------------------------
# print("\n🔄 Performing stratified sampling (~700,000 rows)...")

# SAMPLE_SIZE = 700000
# df_sampled_list = []

# for label_value in y_binary.unique():
#     df_class = df[df["label_binary"] == label_value]

#     sample_count = int((len(df_class) / len(df)) * SAMPLE_SIZE)
#     sample_count = max(sample_count, 1)

#     df_sampled_list.append(df_class.sample(n=sample_count, random_state=42))

# df_sample = pd.concat(df_sampled_list).sample(frac=1, random_state=42)
# print("📊 Sampled dataset:", df_sample.shape)

# # final sampled X and y
# X = df_sample.drop(columns=["label_binary", "label_multi"], errors="ignore")
# y_b = df_sample["label_binary"]
# y_m = df_sample["label_multi"]

# # -----------------------------------------------------------
# # SCALE FEATURES (CRITICAL FIX!)
# # -----------------------------------------------------------
# print("\n📏 Scaling features...")
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# joblib.dump(scaler, os.path.join(OUTPUT_DIR, "supervised_scaler.pkl"))
# print("📌 Saved supervised scaler.")

# # -----------------------------------------------------------
# # Train/Test splits
# # -----------------------------------------------------------
# print("\n✂ Splitting BINARY dataset...")
# X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
#     X_scaled, y_b, test_size=0.25, random_state=42, stratify=y_b
# )

# print("\n✂ Splitting MULTI-CLASS dataset...")
# X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
#     X_scaled, y_m, test_size=0.25, random_state=42, stratify=y_m
# )

# # -----------------------------------------------------------
# # Train Binary RandomForest
# # -----------------------------------------------------------
# print("\n🌲 Training BINARY RandomForest...")
# rf_binary = RandomForestClassifier(
#     n_estimators=150,
#     max_depth=20,
#     random_state=42,
#     class_weight="balanced",
#     n_jobs=-1
# )
# rf_binary.fit(X_train_b, y_train_b)
# pred_b = rf_binary.predict(X_test_b)

# acc_b = accuracy_score(y_test_b, pred_b)
# print(f"🎯 Binary Accuracy: {acc_b:.4f}")

# joblib.dump(rf_binary, os.path.join(OUTPUT_DIR, "rf_binary.pkl"))

# # -----------------------------------------------------------
# # Train Multi-Class RandomForest
# # -----------------------------------------------------------
# print("\n🌲 Training MULTI-CLASS RandomForest...")
# rf_multi = RandomForestClassifier(
#     n_estimators=180,
#     max_depth=25,
#     random_state=42,
#     class_weight="balanced",
#     n_jobs=-1
# )
# rf_multi.fit(X_train_m, y_train_m)
# pred_m = rf_multi.predict(X_test_m)

# acc_m = accuracy_score(y_test_m, pred_m)
# print(f"🎯 Multi-Class Accuracy: {acc_m:.4f}")

# joblib.dump(rf_multi, os.path.join(OUTPUT_DIR, "rf_multiclass.pkl"))

# # -----------------------------------------------------------
# # Save reports
# # -----------------------------------------------------------
# with open(os.path.join(OUTPUT_DIR, "binary_report.txt"), "w") as f:
#     f.write(classification_report(y_test_b, pred_b))

# with open(os.path.join(OUTPUT_DIR, "multi_report.txt"), "w") as f:
#     f.write(classification_report(y_test_m, pred_m))

# # -----------------------------------------------------------
# # Confusion Matrices
# # -----------------------------------------------------------
# plt.figure(figsize=(8, 5))
# sns.heatmap(confusion_matrix(y_test_b, pred_b), annot=True, fmt="d", cmap="Blues")
# plt.title("Binary Confusion Matrix")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "binary_cm.png"))
# plt.close()

# plt.figure(figsize=(12, 8))
# sns.heatmap(confusion_matrix(y_test_m, pred_m), annot=True, fmt="d", cmap="Reds")
# plt.title("Multi-Class Confusion Matrix")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "multi_cm.png"))
# plt.close()

# print("\n🎉 TRAINING COMPLETE!")
# print("📌 Models saved at:", OUTPUT_DIR)




# File: ml_model.py
# FINAL FIXED VERSION — WITH SUPERVISED SCALER + MODEL FEATURES SAVED
# Minimal safe change: use replace=True only for classes that need oversampling.
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("\n🚀 Starting ML Training (BINARY + MULTI-CLASS WITH SCALING)...\n")

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "analysis", "selected_features.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
print("📥 Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print("✅ Loaded:", df.shape)

# -----------------------------------------------------------
# Input Features + Labels
# -----------------------------------------------------------
X_full = df.drop(columns=["label_binary", "label_multi"], errors="ignore")
y_binary = df["label_binary"]
y_multi = df["label_multi"]

# Save feature list for Flask usage
model_features = list(X_full.columns)
joblib.dump(model_features, os.path.join(OUTPUT_DIR, "model_features.pkl"))
print(f"📌 Saved {len(model_features)} model features.")

# -----------------------------------------------------------
# STRATIFIED SAMPLING (~700k rows) - minimal safe fix
# -----------------------------------------------------------
print("\n🔄 Performing stratified sampling (~700,000 rows)...")

SAMPLE_SIZE = 700000
df_sampled_list = []

for label_value in y_binary.unique():
    df_class = df[df["label_binary"] == label_value]

    sample_count = int((len(df_class) / len(df)) * SAMPLE_SIZE)
    sample_count = max(sample_count, 1)

    # MINIMAL CHANGE: allow replacement only when requested sample > available rows
    replace_flag = sample_count > len(df_class)
    if replace_flag:
        print(f"ℹ Oversampling class {label_value}: requested {sample_count}, available {len(df_class)} -> using replace=True")

    df_sampled_list.append(df_class.sample(n=sample_count, replace=replace_flag, random_state=42))

df_sample = pd.concat(df_sampled_list).sample(frac=1, random_state=42).reset_index(drop=True)
print("📊 Sampled dataset:", df_sample.shape)

# final sampled X and y
X = df_sample.drop(columns=["label_binary", "label_multi"], errors="ignore")
y_b = df_sample["label_binary"]
y_m = df_sample["label_multi"]

# -----------------------------------------------------------
# SCALE FEATURES (CRITICAL FIX!)
# -----------------------------------------------------------
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "supervised_scaler.pkl"))
print("📌 Saved supervised scaler.")

# -----------------------------------------------------------
# Train/Test splits
# -----------------------------------------------------------
print("\n✂ Splitting BINARY dataset...")
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_scaled, y_b, test_size=0.25, random_state=42, stratify=y_b
)

print("\n✂ Splitting MULTI-CLASS dataset...")
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_scaled, y_m, test_size=0.25, random_state=42, stratify=y_m
)

# -----------------------------------------------------------
# Train Binary RandomForest
# -----------------------------------------------------------
print("\n🌲 Training BINARY RandomForest...")
rf_binary = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf_binary.fit(X_train_b, y_train_b)
pred_b = rf_binary.predict(X_test_b)

acc_b = accuracy_score(y_test_b, pred_b)
print(f"🎯 Binary Accuracy: {acc_b:.4f}")

joblib.dump(rf_binary, os.path.join(OUTPUT_DIR, "rf_binary.pkl"))

# -----------------------------------------------------------
# Train Multi-Class RandomForest
# -----------------------------------------------------------
print("\n🌲 Training MULTI-CLASS RandomForest...")
rf_multi = RandomForestClassifier(
    n_estimators=180,
    max_depth=25,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf_multi.fit(X_train_m, y_train_m)
pred_m = rf_multi.predict(X_test_m)

acc_m = accuracy_score(y_test_m, pred_m)
print(f"🎯 Multi-Class Accuracy: {acc_m:.4f}")

joblib.dump(rf_multi, os.path.join(OUTPUT_DIR, "rf_multiclass.pkl"))

# -----------------------------------------------------------
# Save reports
# -----------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "binary_report.txt"), "w") as f:
    f.write(classification_report(y_test_b, pred_b))

with open(os.path.join(OUTPUT_DIR, "multi_report.txt"), "w") as f:
    f.write(classification_report(y_test_m, pred_m))

# -----------------------------------------------------------
# Confusion Matrices
# -----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test_b, pred_b), annot=True, fmt="d", cmap="Blues")
plt.title("Binary Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "binary_cm.png"))
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test_m, pred_m), annot=True, fmt="d", cmap="Reds")
plt.title("Multi-Class Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "multi_cm.png"))
plt.close()

print("\n🎉 TRAINING COMPLETE!")
print("📌 Models saved at:", OUTPUT_DIR)
