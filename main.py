import argparse
import pandas as pd
from src.data_preprocessing import clean_data
from src.model_training import train_model, evaluate_model, save_model
from src.inference import run_inference

def load_and_prepare_data(train_path):
    print("[INFO] Loading and cleaning training dataset...")
    df = pd.read_csv(train_path)
    print(f"[INFO] Successfully loaded data from: {train_path}")
    df = clean_data(df)
    return df

def main(args):
    # Load and preprocess training data
    df = load_and_prepare_data(args.train_path)

    # Split features and target
    X = df.drop(columns=["target", "id"])
    y = df["target"]

    # Split into train and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("[INFO] Split data into training and validation sets.")

    # Train the model
    print("[INFO] Training model...")
    model = train_model(X_train, y_train, X_valid, y_valid)

    # Evaluate the model
    print("[INFO] Evaluating model...")
    evaluate_model(model, X_valid, y_valid)

    # Save the trained model
    print(f"[INFO] Saving model to: {args.model_path}")
    save_model(model, args.model_path)

    # Run inference if test path is provided
    if args.test_path:
        run_inference(args.test_path, args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test data CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save trained model")
    args = parser.parse_args()
    main(args)
