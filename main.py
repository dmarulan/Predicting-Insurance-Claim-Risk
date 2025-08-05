import argparse
from src.preprocess.data_loader import load_data
from src.preprocess.cleaning import clean_data
from src.model_training import train_model
from src.inference import run_inference

def main(train_path, test_path, model_path, predictions_path):
    print("[INFO] Loading and cleaning training dataset...")
    df_train = load_data(train_path)
    X_train, y_train = clean_data(df_train, is_train=True)

    print("[INFO] Training model...")
    model = train_model(X_train, y_train, model_path)

    print("[INFO] Running inference on test dataset...")
    predictions_df = run_inference(model_path, test_path, predictions_path)

    print(f"[INFO] Predictions saved to: {predictions_path}")
    print(predictions_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate insurance claim model.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to testing CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--predictions_path", type=str, default="predictions.csv", help="Path to save predictions")

    args = parser.parse_args()

    main(
        train_path=args.train_path,
        test_path=args.test_path,
        model_path=args.model_path,
        predictions_path=args.predictions_path
    )
