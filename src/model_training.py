def train_model(X_train, y_train, X_valid, y_valid):
    """Train an XGBoost model using sample_weight for class imbalance"""

    # Calculate class weights manually
    counter = Counter(y_train)
    total = sum(counter.values())
    class_weights = {cls: total / count for cls, count in counter.items()}
    print(f"[INFO] Class weights: {class_weights}")

    # Map each sample to its class weight
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
