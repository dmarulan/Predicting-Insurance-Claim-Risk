import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)
from xgboost import XGBClassifier


def load_data(data_path):
    """Load preprocessed dataset"""
    return pd.read_csv(data_path)


def compute_sample_weights(y_train):
    """Compute sample weights for class imbalance"""
    counter = Counter(y_train)
    total = sum(counter.values())
    class_weights = {cls: total / count for cls, count in counter.items()}
    sample_weights = y_train.map(class_weights)
    print(f"[INFO] Sample weights applied. Class distribution: {dict(counter)}")
    return sample_weights


def tune_hyperparameters(X_train, y_train, sample_weights):
    """Perform RandomizedSearchCV to tune hyperparameters with sample_weight"""
    param_grid = {
        'n_estimators': [100, 300, 500, 700],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 1, 5],
        'reg_lambda': [1, 5, 10],
    }

    base_model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=25,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    print("[INFO] Starting hyperparameter tuning...")
    random_search.fit(X_train, y_train, sample_weight=sample_weights)
    print("[INFO] Best parameters found:")
    print(random_search.best_params_)

    return random_search.best_estimator_


def evaluate_model(model, X_valid, y_valid):
    """Evaluate model performance on validation set"""
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

    # Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return auc


def save_model(model, output_path='models/xgb_model_tuned.pkl'):
    """Save trained model to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"[INFO] Model saved to: {output_path}")


def train_model(data_path="data/processed_data.csv"):
    """Main training pipeline"""
    data = load_data(data_path)
    X = data.drop(columns=["target", "id"])
    y = data["target"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sample_weights = compute_sample_weights(y_train)
    model = tune_hyperparameters(X_train, y_train, sample_weights)
    auc = evaluate_model(model, X_valid, y_valid)
    save_model(model)

    return model, auc


if __name__ == "__main__":
    train_model()
