import pandas as pd

def load_data(train_path: str, test_path: str):
    """
    Loads training and testing datasets from the given paths.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.

    Returns:
        Tuple of pd.DataFrame: (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df
