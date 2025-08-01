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

def reduce_memory_usage(df):
    """Downcast numeric columns to reduce memory footprint"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Initial memory usage of dataframe: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and str(col_type)[:3] != 'dat':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if np.iinfo(np.int8).min < c_min < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif np.iinfo(np.int16).min < c_min < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif np.iinfo(np.int32).min < c_min < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Final memory usage of dataframe: {end_mem:.2f} MB")
    print(f"Reduced by: {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df
