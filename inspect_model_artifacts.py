# inspect_model_artifacts.py
import os, joblib, json
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
analysis = os.path.join(BASE_DIR, "analysis")

MODEL_PATH = os.path.join(analysis, "network_intrusion_model.pkl")
SCALER_PATH = os.path.join(analysis, "scaler.pkl")
FEATURES_PATH = os.path.join(analysis, "model_features.pkl")

def try_load(path):
    if not os.path.exists(path):
        print("MISSING:", path)
        return None
    try:
        obj = joblib.load(path)
        print("LOADED:", path, "type:", type(obj))
        return obj
    except Exception as e:
        print("FAILED to load", path, ":", e)
        return None

model = try_load(MODEL_PATH)
scaler = try_load(SCALER_PATH)
features = try_load(FEATURES_PATH)

if features is not None:
    print("\nFirst 40 features (or fewer):")
    try:
        if isinstance(features, (list, tuple)):
            print(features[:40])
        else:
            # sometimes it's numpy array or pandas Index
            print(list(features)[:40])
    except Exception as e:
        print("Could not print features:", e)