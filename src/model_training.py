import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import os

def load_data(data_path):
    """Load preprocessed data"""
    return pd.read_csv(data_path)

def train_model(X_train, y_train, X_valid, y_valid):
    """Train an XGBoost model and return the trained model"""
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42
    )

    try:
        model.fit(
            X_train,
            y_train,
            # early_stopping_rounds=10,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
    except TypeError:
        # Handle newer XGBoost versions with callback interface
        from xgboost.callback import EarlyStopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            # callbacks=[EarlyStopping(rounds=10)],
            verbose=False
        )

    # Compute and print AUC on validation set
    y_valid_pred_proba = model.predict_proba(X_valid)[:, 1]
    val_auc = roc_auc_score(y_valid, y_valid_pred_proba)
    print(f"[INFO] Validation AUC: {val_auc:.4f}")
    
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
    print(f"[INFO] Model saved to: {output_path}")

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
