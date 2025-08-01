import pandas as pd

def clean_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Cleans the training and test dataframes. This may include:
      - Removing duplicates
      - Filling missing values
      - Dropping unnecessary columns

    Args:
        train_df (pd.DataFrame): Raw training dataframe.
        test_df (pd.DataFrame): Raw test dataframe.

    Returns:
        Tuple of cleaned pd.DataFrame: (cleaned_train_df, cleaned_test_df)
    """
    # Remove duplicate rows
    train_df = train_df.drop_duplicates()

    # Fill missing values with median
    train_df = train_df.fillna(train_df.median(numeric_only=True))
    test_df = test_df.fillna(test_df.median(numeric_only=True))

    # Drop ID columns if present
    id_cols = [col for col in train_df.columns if 'id' in col.lower()]
    train_df = train_df.drop(columns=id_cols, errors='ignore')
    test_df = test_df.drop(columns=id_cols, errors='ignore')

    return train_df, test_df
