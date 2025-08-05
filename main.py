import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess.data_loader import load_data
from src.preprocess.cleaning import clean_data
from src.model.train import train_model
from src.model.inference import run_inference

def main(train_path, test_path, model_path):
    # Load and clean training data
    print("[INFO] Loading and cleaning training dataset...")
    df_train = load_data(train_path)
    df_train = clean_data(df_train)
    X = df_train.drop("target", axis=1)
    y = df_train["target"]

    # Split into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    print("[INFO] Split data into training and validation sets.")

    # Train model
    print("[INFO] Training model...")
    model = train_model(X_train, y_train, X_valid, y_valid)

    # Save model
    print(f"[INFO] Saving model to: {model_path}")
    pd.to_pickle(model, model_path)

    # Run inference on test data
    print("[INFO] Running inference on test dataset...")
    predictions_df = run_inference(model_path, test_path)

    # Show preview of predictions
    print("[INFO] Inference complete. Here are the first few predictions:")
    print(predictions_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insurance Claim Prediction Pipeline")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save trained model (.pkl)")
    args = parser.parse_args()

    main(args.train_path, args.test_path, args.model_path)
