import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import os

def load_data(data_path):
    """Load preprocessed data"""
    return pd.read_csv(data_path)

def train_model(X_train, y_train, X_valid, y_valid):
    """Train an XGBoost model and return the trained model"""
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='auc'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=50,
        verbose=True
    )
    return model

def evaluate_model(model, X_valid, y_valid):
    """Evaluate the model on the validation set"""
    preds = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, preds)
    print(f"Validation AUC: {score:.4f}")
    return score

def save_model(model, output_path='models/xgb_model.pkl'):
    """Save the trained model to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)

def main():
    data = load_data("data/processed_data.csv")  # Adjust path as needed
    X = data.drop(columns=["target", "id"])
    y = data["target"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train, X_valid, y_valid)
    evaluate_model(model, X_valid, y_valid)
    save_model(model)

if __name__ == "__main__":
    main()
