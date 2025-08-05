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
    try:
        test_df = pd.read_csv(test_path, encoding='latin1', engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"[ERROR] Failed to read test file: {e}")
        return

    # Ensure 'id' column exists
    if "id" not in test_df.columns:
        print("[ERROR] 'id' column not found in test data.")
        return

    # Create a dummy train DataFrame with matching columns for cleaning
    dummy_train_df = pd.DataFrame(columns=test_df.columns.tolist() + ["target"])

    print("[INFO] Cleaning test dataset...")
    _, cleaned_test = clean_data(dummy_train_df, test_df)

    print("[INFO] Loading trained model...")
    model = joblib.load(model_path)

    print("[INFO] Running predictions...")
    try:
        predictions = model.predict(cleaned_test)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return

    results = pd.DataFrame({
        "id": test_df.index,  # Use the actual ID column
        "prediction": predictions
    })

    results.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to {output_path}")

    return results
