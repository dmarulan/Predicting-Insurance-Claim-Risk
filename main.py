import argparse
from src import preprocess, model_training, inference

def run_preprocessing(input_path, output_path):
    print("Running data preprocessing...")
    df = preprocess.load_and_clean_data(input_path)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def run_training(train_path, model_path, scaler_path):
    print("Running model training...")
    df = preprocess.load_and_clean_data(train_path)
    model, scaler, features = model_training.train_model(df)
    model_training.save_model_and_scaler(model, scaler, features, model_path, scaler_path)
    print(f"Model and scaler saved to {model_path}, {scaler_path}")

def run_inference(csv_path, model_path, scaler_path, feature_columns):
    print("Running inference...")
    df = inference.predict_from_csv(model_path, csv_path, feature_columns, scaler_path)
    print(df[['prediction', 'risk_score']].head())

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline for Risk Prediction")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['preprocess', 'train', 'predict'],
                        help="Mode to run: preprocess / train / predict")

    parser.add_argument('--input', type=str, help="Input CSV path")
    parser.add_argument('--output', type=str, help="Output CSV path")
    parser.add_argument('--model', type=str, default='models/risk_model.pkl', help="Model path")
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl', help="Scaler path")
    parser.add_argument('--features', type=str, nargs='+', help="List of feature column names")

    args = parser.parse_args()

    if args.mode == 'preprocess':
        run_preprocessing(args.input, args.output)
    elif args.mode == 'train':
        run_training(args.input, args.model, args.scaler)
    elif args.mode == 'predict':
        if not args.features:
            raise ValueError("Feature columns are required for prediction.")
        run_inference(args.input, args.model, args.scaler, args.features)

if __name__ == "__main__":
    main()
