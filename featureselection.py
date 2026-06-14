import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

print("\n🚀 Starting MEMORY-SAFE Feature Selection Pipeline...\n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "preprocess_output.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "analysis")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("❌ preprocess_output.csv not found! Run preprocess.py first.")

# Smaller chunk size to avoid 137MB RAM crash
CHUNK_SIZE = 200_000

# ----------------------------------------------
# 1️⃣ FIRST PASS — Detect numeric columns
# ----------------------------------------------
print("📥 First pass: scanning dataset safely...")

first_chunk = next(pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False))
numeric_candidates = first_chunk.select_dtypes(include=["int64","float64"]).columns.tolist()

print("\n🔢 NUMERIC COLUMNS:")
print(numeric_candidates)

# Label extraction container
attack_types = []

# Read all attack labels (chunked)
for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    attack_types.extend(chunk["attack_type"].astype(str).tolist())

# Label encoding
label_encoder = LabelEncoder()
label_encoder.fit(sorted(set(attack_types)))

label_map = {cls: int(idx) for idx, cls in enumerate(label_encoder.classes_)}
print("\n🎯 Multi-class label mapping ready.")

# Save label map
with open(os.path.join(OUTPUT_FOLDER, "label_mapping_multiclass.json"), "w") as f:
    json.dump(label_map, f, indent=4)

# ----------------------------------------------
# 2️⃣ SCALING in 2 passes (very RAM-safe)
# ----------------------------------------------
print("\n⚙ Second pass: fitting scaler incrementally...")

scaler = StandardScaler()
first_fit = True

for chunk in tqdm(pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
    # Convert numeric safely
    for col in numeric_candidates:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

    if first_fit:
        scaler.partial_fit(chunk[numeric_candidates])
        first_fit = False
    else:
        scaler.partial_fit(chunk[numeric_candidates])

print("📏 Scaler trained safely.")

# ----------------------------------------------
# 3️⃣ THIRD PASS — SCALE + SAMPLE only 600k rows
# ----------------------------------------------
print("\n⚙ Third pass: scaling and sampling...")

sampled_rows = []

for chunk in tqdm(pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
    # Prepare numeric data
    for col in numeric_candidates:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

    # Encode label
    chunk["label_multi"] = label_encoder.transform(chunk["attack_type"])

    # Scale numeric features
    chunk[numeric_candidates] = scaler.transform(chunk[numeric_candidates])

    # Randomly sample 8k rows per chunk
    sampled = chunk.sample(n=min(8000, len(chunk)), random_state=42)
    sampled_rows.append(sampled)

# Merge sampled rows
df_sample = pd.concat(sampled_rows, ignore_index=True)
print(f"\n📌 Sample size for RF: {df_sample.shape}")

# Add binary label (needed for saving selected_features.csv)
df_sample["label_binary"] = df_sample["attack_type"].apply(lambda x: 0 if x == "benign" else 1)


# ----------------------------------------------
# 4️⃣ RandomForest FEATURE IMPORTANCE
# ----------------------------------------------
print("\n🌲 Training RandomForest on sample...")

X = df_sample[numeric_candidates]
y = df_sample["label_multi"]

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X, y)

importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

top_feats = importances.head(30)
print("\n🔥 Top 30 features selected.")

# ----------------------------------------------
# 5️⃣ SAVE EVERYTHING
# ----------------------------------------------
# Save full processed dataset (scaled)
print("\n💾 Saving processed_features.csv in chunks...")

processed_path = os.path.join(OUTPUT_FOLDER, "processed_features.csv")
if os.path.exists(processed_path):
    os.remove(processed_path)

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    for col in numeric_candidates:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

    chunk[numeric_candidates] = scaler.transform(chunk[numeric_candidates])
    chunk["label_multi"] = label_encoder.transform(chunk["attack_type"])
    chunk["label_binary"] = chunk["attack_type"].apply(lambda x: 0 if x == "benign" else 1)

    chunk.to_csv(
        processed_path,
        mode="a",
        index=False,
        header=not os.path.exists(processed_path)
    )

selected_path = os.path.join(OUTPUT_FOLDER, "selected_features.csv")
df_sample[top_feats["Feature"].tolist() + ["label_multi", "label_binary"]] \
    .to_csv(selected_path, index=False)

importances.to_csv(os.path.join(OUTPUT_FOLDER, "feature_importance_mc.csv"), index=False)

print("\n🎉 FEATURE SELECTION COMPLETE!")
print("📂 Processed dataset saved at:", processed_path)
print("📂 Selected features saved at:", selected_path)
print("📄 Feature importance saved.")
print("🚀 Ready for ML training!")
