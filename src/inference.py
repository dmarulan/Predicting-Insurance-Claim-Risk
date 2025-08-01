import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """Load a trained model from disk."""
    return joblib.load(model_path)

def preprocess_input(data, feature_columns, scaler_path=None):
    """Preprocess input data before inference."""
    X = data[feature_columns].copy()

    if scaler_path:
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)

    return X

def predict(model, processed_data):
    """Return predictions and prediction probabilities."""
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]
    return predictions, probabilities

def predict_from_csv(model_path, csv_path, feature_columns, scaler_path=None):
    """Run full prediction pipeline on data in CSV file."""
    data = pd.read_csv(csv_path)
    model = load_model(model_path)
    processed = preprocess_input(data, feature_columns, scaler_path)
    preds, probs = predict(model, processed)
    data['prediction'] = preds
    data['risk_score'] = probs
    return data
