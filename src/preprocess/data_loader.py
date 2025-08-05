import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file from the given path.

    Parameters:
    - path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Successfully loaded data from: {path}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load data from {path}: {e}")
        raise

