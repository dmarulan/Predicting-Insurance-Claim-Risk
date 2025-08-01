# src/utils.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

def save_model(model, path):
    """Save a model to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

def load_model(path):
    """Load a model from disk"""
    return joblib.load(path)

def evaluate_classification_model(model, X, y, return_report=False):
    """Evaluate a classification model and print performance metrics"""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    metrics = {
        "AUC": roc_auc_score(y, proba),
        "Accuracy": accuracy_score(y, preds),
        "Confusion Matrix": confusion_matrix(y, preds)
    }

    print("Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}:")
        print(v if k == "Confusion Matrix" else f"{v:.4f}")
        print()

    if return_report:
        print("Classification Report:")
        print(classification_report(y, preds))
    
    return metrics
