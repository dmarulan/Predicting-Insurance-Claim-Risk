import pandas as pd
import joblib
from src.preprocess.cleaning import clean_data


def run_inference(model_path: str, test_path: str, output_path: str = "predictions.csv"):
    """
    Loads the trained model and performs inference on test data.

    Args:
        model_path (str): Path to the trained model .pkl file.
        test_path (str): Path to the test CSV file.
        output_path (str): File path where predictions will be saved.
    """
    print("[INFO] Loading test dataset...")
    test_df = pd.read_csv(test_path, encoding='ISO-8859-1')

    # Creating a dummy train DataFrame for cleaning structure
    dummy_train_df = pd.DataFrame(columns=test_df.columns.tolist() + ["target"])

    print("[INFO] Cleaning test dataset...")
    _, cleaned_test = clean_data(dummy_train_df, test_df)

    print("[INFO] Loading trained model...")
    model = joblib.load(model_path)

    print("[INFO] Running predictions...")
    predictions = model.predict(cleaned_test)

    results = pd.DataFrame({
        "id": test_df.index,
        "prediction": predictions
    })

    results.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to {output_path}")

    return results
