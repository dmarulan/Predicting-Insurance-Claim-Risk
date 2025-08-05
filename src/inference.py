import pandas as pd
import joblib
import os

def load_model(model_path='models/xgb_model_tuned.pkl'):
    """Load a trained XGBoost model from disk"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    print(f"[INFO] Loaded model from: {model_path}")
    return model

def load_test_data(test_data_path='data/test_data.csv'):
    """Load test data for inference"""
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")
    data = pd.read_csv(test_data_path)
    print(f"[INFO] Loaded test data with shape: {data.shape}")
    return data

def preprocess_test_data(data):
    """Preprocess test data before prediction"""
    if "id" in data.columns:
        ids = data["id"]
        features = data.drop(columns=["id"])
    else:
        ids = pd.Series(range(len(data)), name="id")
        features = data
    return ids, features

def predict(model, features):
    """Generate predicted probabilities and binary labels"""
    y_pred_proba = model.predict_proba(features)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    return y_pred_proba, y_pred_binary

def save_predictions(ids, y_pred_proba, y_pred_binary, output_path='results/predictions.csv'):
    """Save predictions to a CSV file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = pd.DataFrame({
        "id": ids,
        "predicted_probability": y_pred_proba,
        "predicted_label": y_pred_binary
    })
    results.to_csv(output_path, index=False)
    print(f"[INFO] Saved predictions to: {output_path}")
    return results

def run_inference(model_path='models/xgb_model_tuned.pkl', test_data_path='data/test_data.csv', output_path='results/predictions.csv'):
    """Main inference function to be imported and run from main.py"""
    model = load_model(model_path)
    test_data = load_test_data(test_data_path)
    ids, features = preprocess_test_data(test_data)
    y_pred_proba, y_pred_binary = predict(model, features)
    results = save_predictions(ids, y_pred_proba, y_pred_binary, output_path)
    print(results.head())
    return results

def main():
    run_inference()

if __name__ == "__main__":
    main()
