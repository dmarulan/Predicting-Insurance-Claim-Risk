import pandas as pd

def clean_data(train_df, test_df=None):
    """
    Cleans the dataset(s): handles missing values, encodes categoricals, drops unneeded features.
    If test_df is provided, applies same transformations using train_df logic.
    Returns: cleaned_train_df [, cleaned_test_df if test_df is provided]
    """
    def preprocess(df):
        df = df.copy()

        # Drop constant columns
        nunique = df.nunique()
        const_cols = nunique[nunique == 1].index.tolist()
        df.drop(columns=const_cols, inplace=True)

        # Drop ID if exists
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)

        # Handle missing values: fill with median
        df.fillna(df.median(numeric_only=True), inplace=True)

        return df

    train_cleaned = preprocess(train_df)

    if test_df is not None:
        test_cleaned = preprocess(test_df)

        # Align test to train (e.g. drop columns that are only in train or test)
        common_cols = train_cleaned.columns.intersection(test_cleaned.columns)
        return train_cleaned[common_cols], test_cleaned[common_cols]

    return train_cleaned
