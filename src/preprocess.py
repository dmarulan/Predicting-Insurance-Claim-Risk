import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    """Load the training and test datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def get_feature_types(df):
    """Identify numerical and categorical columns (based on naming convention)."""
    cat_features = [col for col in df.columns if '_cat' in col]
    bin_features = [col for col in df.columns if '_bin' in col]
    num_features = [col for col in df.columns if col not in cat_features + bin_features + ['id', 'target']]
    return num_features, cat_features, bin_features

def build_preprocessing_pipeline(num_features, cat_features, bin_features):
    """Create a preprocessing pipeline for numerical and categorical features."""

    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline (one-hot encoding after filling -1 with most frequent)
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Binary features: just impute with most frequent (no scaling needed)
    bin_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
        ('bin', bin_pipeline, bin_features)
    ])

    return preprocessor

def preprocess_data(train, test):
    """Preprocess the data using a pipeline."""
    num_features, cat_features, bin_features = get_feature_types(train)

    # Drop ID columns and store them separately
    train_ids = train['id']
    test_ids = test['id']
    y = train['target']

    train = train.drop(['id', 'target'], axis=1)
    test = test.drop(['id'], axis=1)

    preprocessor = build_preprocessing_pipeline(num_features, cat_features, bin_features)
    X_train_processed = preprocessor.fit_transform(train)
    X_test_processed = preprocessor.transform(test)

    return X_train_processed, X_test_processed, y, train_ids, test_ids, preprocessor

def save_processed_data(X_train, X_test, y, out_dir='data/processed'):
    """Save the processed data as NumPy arrays."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(out_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(out_dir, 'y_train.npy'), y)

if __name__ == '__main__':
    train, test = load_data()
    X_train, X_test, y, train_ids, test_ids, preprocessor = preprocess_data(train, test)
    save_processed_data(X_train, X_test, y)
