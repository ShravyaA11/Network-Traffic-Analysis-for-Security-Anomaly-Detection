# utils/ml_utils.py
import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "analysis", "network_intrusion_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "analysis", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "analysis", "model_features.pkl")

def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

def predict_from_df(df):
    model, scaler, features = load_models()
    df = df.fillna(0)
    for c in df.select_dtypes(include='object').columns:
        df[c] = pd.factorize(df[c])[0]
    for f in features:
        if f not in df.columns:
            df[f] = 0
    df = df[features]
    scaled = scaler.transform(df)
    preds = model.predict(scaled)
    df['Prediction'] = ['Attack' if p == 1 else 'Normal' for p in preds]
    return df