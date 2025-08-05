import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning operations on the input DataFrame.

    Cleaning steps:
    - Drop constant columns (columns with only one unique value).
    - Fill missing values with column medians.

    Parameters:
    - df (pd.DataFrame): Raw input DataFrame.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    print("[INFO] Cleaning dataset...")

    # Drop columns with only one unique value (constants)
    nunique = df.nunique()
    constant_cols = nunique[nunique == 1].index.tolist()
    if constant_cols:
        print(f"[INFO] Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # Fill missing values with median of the column
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"[INFO] Filling missing values in columns: {missing_cols}")
        df = df.fillna(df.median(numeric_only=True))

    return df
