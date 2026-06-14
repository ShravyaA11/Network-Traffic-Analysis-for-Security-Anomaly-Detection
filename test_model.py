import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load binary & multi-class models
model_binary = joblib.load(os.path.join(BASE_DIR, "models", "rf_binary.pkl"))
model_multi = joblib.load(os.path.join(BASE_DIR, "models", "rf_multiclass.pkl"))

# Load sample dataset
df = pd.read_csv(os.path.join(BASE_DIR, "analysis", "selected_features.csv"))

# Features only
X = df.drop(columns=["label_binary", "label_multi"], errors="ignore")
y_binary = df["label_binary"]
y_multi = df["label_multi"]

print("\n==============================")
print("TEST 1: RANDOM SAMPLE TEST")
print("==============================\n")

sample = df.sample(5, random_state=42)
X_sample = sample.drop(columns=["label_binary", "label_multi"], errors="ignore")

print("Actual Binary:", sample["label_binary"].values)
print("Predicted Binary:", model_binary.predict(X_sample))

print("\nActual Multi:", sample["label_multi"].values)
print("Predicted Multi:", model_multi.predict(X_sample))


print("\n==============================")
print("TEST 2: BENIGN SAMPLE TEST")
print("==============================\n")

ben = df[df["label_binary"] == 0].sample(3, random_state=42)
print("Actual:", ben["label_binary"].values)
print("Predicted:", model_binary.predict(ben.drop(columns=["label_binary", "label_multi"], errors="ignore")))


print("\n==============================")
print("TEST 3: ATTACK SAMPLE TEST")
print("==============================\n")

att = df[df["label_binary"] == 1].sample(3, random_state=42)
print("Actual:", att["label_binary"].values)
print("Predicted:", model_binary.predict(att.drop(columns=["label_binary", "label_multi"], errors="ignore")))


print("\n==============================")
print("TEST 4: MULTI-CLASS LABEL TEST")
print("==============================\n")

multi_sample = df.sample(5, random_state=10)
print("Actual:", multi_sample["label_multi"].values)
print("Predicted:", model_multi.predict(multi_sample.drop(columns=["label_binary", "label_multi"], errors="ignore")))


print("\n==============================")
print("TEST 5: SMALL ACCURACY CHECK")
print("==============================\n")

small_test = df.sample(20000, random_state=55)
X_s = small_test.drop(columns=["label_binary", "label_multi"], errors="ignore")

from sklearn.metrics import accuracy_score

print("Binary accuracy:",
      accuracy_score(small_test["label_binary"], model_binary.predict(X_s)))

print("Multi-class accuracy:",
      accuracy_score(small_test["label_multi"], model_multi.predict(X_s)))

print("\n🎯 ALL TESTS COMPLETED.")
