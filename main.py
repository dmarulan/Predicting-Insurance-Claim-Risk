import os

# Detect if running on Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Mount Google Drive and set data path
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = '/content/drive/MyDrive/Insurance_Claims_Porto_Seguro'
else:
    BASE_PATH = '.'

# Set paths for data and model output
DATA_PATH = os.path.join(BASE_PATH, 'data')
MODEL_OUTPUT_PATH = os.path.join(BASE_PATH, 'models')
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
TEST_PATH = os.path.join(DATA_PATH, 'test.csv')

# Import custom modules
from src.preprocess.data_loader import load_data
from src.preprocess.cleaning import clean_data
from src.utils import evaluate_model
from src.model_training import train_model
from src.inference import make_predictions

def main():
    print("🚀 Starting Machine Learning pipeline...")

    # Load data
    print("📥 Loading data...")
    df_train, df_test = load_data(TRAIN_PATH, TEST_PATH)

    # Clean data
    print("🧹 Cleaning data...")
    df_train_cleaned, df_test_cleaned = clean_data(df_train, df_test)

    # Train model
    print("🧠 Training model...")
    model, X_valid, y_valid = train_model(df_train_cleaned)

    # Evaluate model
    print("📊 Evaluating model...")
    evaluate_model(model, X_valid, y_valid)

    # Inference on test data
    print("🔍 Running inference...")
    predictions = make_predictions(model, df_test_cleaned)

    # Save predictions
    output_path = os.path.join(BASE_PATH, 'submission.csv')
    predictions.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
