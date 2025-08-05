import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_path):
    """Load preprocessed data"""
    print(f"[INFO] Loading data from: {data_path}")
    return pd.read_csv(data_path)

def train_model(X_train, y_train, X_valid, y_valid):
    """Train an XGBoost model using sample_weight for class imbalance"""

    # Calculate class weights manually
    counter = Counter(y_train)
    total = sum(counter.values())
    class_weights = {cls: total / count for cls, count in counter.items()}
    print(f"[INFO] Class distribution: {dict(counter)}")
    print(f"[INFO] Class weights: {class_weights}")

    # Assign a weight to each training sample
    sample_weight = y_train.map(class_weights)

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

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    return model

def evaluate_model(model, X_valid, y_valid):
    """Evaluate the model on the validation set with multiple metrics"""
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    y_pred_binary = model.predict(X_valid)

    auc = roc_auc_score(y_valid, y_pred_proba)
    recall = recall_score(y_valid, y_pred_binary)
    f1 = f1_score(y_valid, y_pred_binary)
    f2 = fbeta_score(y_valid, y_pred_binary, beta=2)

    print(f"[RESULT] Validation AUC: {auc:.4f}")
    print(f"[RESULT] Recall: {recall:.4f}")
    print(f"[RESULT] F1 Score: {f1:.4f}")
    print(f"[RESULT] F2 Score: {f2:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_valid, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return auc

def save_model(model, output_path='models/xgb_model.pkl'):
    """Save the trained model to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"[INFO] Model saved to: {output_path}")

def main():
    # Load cleaned dataset
    data = load_data("data/processed_data.csv")  # Update path as needed
    X = data.drop(columns=["target", "id"])
    y = data["target"]

    # Train-validation split with stratification
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # Evaluate the model
    evaluate_model(model, X_valid, y_valid)

    # Save the model
    save_model(model)

if __name__ == "__main__":
    main()
