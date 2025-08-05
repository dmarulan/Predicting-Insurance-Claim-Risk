# import argparse
# import pandas as pd
# from sklearn.model_selection import train_test_split

# from src.preprocess.data_loader import load_data
# from src.preprocess.cleaning import clean_data
# from src.model.train import train_model
# from src.model.inference import run_inference

# def main(train_path, test_path, model_path):
#     # Load and clean training data
#     print("[INFO] Loading and cleaning training dataset...")
#     df_train = load_data(train_path)
#     df_train = clean_data(df_train)
#     X = df_train.drop("target", axis=1)
#     y = df_train["target"]

#     # Split into training and validation sets
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#     print("[INFO] Split data into training and validation sets.")

#     # Train model
#     print("[INFO] Training model...")
#     model = train_model(X_train, y_train, X_valid, y_valid)

#     # Save model
#     print(f"[INFO] Saving model to: {model_path}")
#     pd.to_pickle(model, model_path)

#     # Run inference on test data
#     print("[INFO] Running inference on test dataset...")
#     predictions_df = run_inference(model_path, test_path)

#     # Show preview of predictions
#     print("[INFO] Inference complete. Here are the first few predictions:")
#     print(predictions_df.head())

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Insurance Claim Prediction Pipeline")
#     parser.add_argument("--train_path", type=str, required=True, help="Path to training CSV file")
#     parser.add_argument("--test_path", type=str, required=True, help="Path to test CSV file")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to save trained model (.pkl)")
#     args = parser.parse_args()

#     main(args.train_path, args.test_path, args.model_path)

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess.data_loader import load_data
from src.preprocess.cleaning import clean_data
from src.model_training import train_model
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
    X = df.drop(columns=["target"])
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

