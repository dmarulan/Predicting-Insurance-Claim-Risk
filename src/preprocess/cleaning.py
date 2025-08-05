# import pandas as pd

# def clean_data(train_df, test_df=None):
#     """
#     Cleans the dataset(s): handles missing values, encodes categoricals, drops unneeded features.
#     If test_df is provided, applies same transformations using train_df logic.
#     Returns: cleaned_train_df [, cleaned_test_df if test_df is provided]
#     """
#     def preprocess(df):
#         df = df.copy()

#         # Drop constant columns
#         nunique = df.nunique()
#         const_cols = nunique[nunique == 1].index.tolist()
#         df.drop(columns=const_cols, inplace=True)

#         # # Drop ID if exists
#         # if 'id' in df.columns:
#         #     df.drop(columns=['id'], inplace=True)

#         # Handle missing values: fill with median
#         df.fillna(df.median(numeric_only=True), inplace=True)

#         return df

#     train_cleaned = preprocess(train_df)

#     if test_df is not None:
#         test_cleaned = preprocess(test_df)

#         # Align test to train (e.g. drop columns that are only in train or test)
#         common_cols = train_cleaned.columns.intersection(test_cleaned.columns)
#         return train_cleaned[common_cols], test_cleaned[common_cols]

#     return train_cleaned

import pandas as pd

def clean_data(train_df, test_df=None):
    """
    Cleans the dataset(s): handles missing values, encodes categoricals, drops unneeded features.
    If test_df is provided, applies same transformations using train_df logic.
    Returns: cleaned_train_df [, cleaned_test_df if test_df is provided]
    """

    def preprocess(df):
        df = df.copy()

        # Save ID column if it exists (especially for test set)
        id_col = df['id'] if 'id' in df.columns else None

        # Drop constant columns
        nunique = df.nunique()
        const_cols = nunique[nunique == 1].index.tolist()
        df.drop(columns=const_cols, inplace=True)

        # Drop ID column temporarily to avoid affecting processing
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)

        # Fill missing values with median (numerical only)
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Reattach ID column if it was present
        if id_col is not None:
            df['id'] = id_col

        return df

    # Clean train data
    train_cleaned = preprocess(train_df)

    # Clean test data if provided
    if test_df is not None:
        test_cleaned = preprocess(test_df)

        # Align columns: match only on features, not ID
        features = train_cleaned.drop(columns=["id"]).columns.intersection(
            test_cleaned.drop(columns=["id"]).columns
        )

        # Return both cleaned sets with 'id' preserved
        return (
            train_cleaned[features.tolist() + ["id"]],
            test_cleaned[features.tolist() + ["id"]],
        )

    # Return only cleaned train set
    return train_cleaned
