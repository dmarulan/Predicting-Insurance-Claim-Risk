# Porto Seguro Safe Driver Prediction - Data Exploration & Cleaning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# Load the data
train_df = pd.read_csv('/content/data/train.csv')
test_df = pd.read_csv('/content/data/test.csv')

# Basic info
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain data types:")
print(train_df.dtypes.value_counts())

# Preview data
train_df.head()

# Check for missing values
def missing_values(df):
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    return pd.DataFrame({'Total Missing': total, 'Percent': percent}).sort_values(by='Percent', ascending=False)

print("\nMissing values in training data:")
print(missing_values(train_df))

# In this dataset, missing values are represented as -1
missing_summary = (train_df == -1).sum()
missing_percent = 100 * missing_summary / len(train_df)
missing_df = pd.DataFrame({'Missing Count': missing_summary, 'Percent': missing_percent})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percent', ascending=False)

print("\nFeatures with missing values (-1):")
print(missing_df)

# Plot missing data
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_df.index, y='Percent', data=missing_df)
plt.xticks(rotation=90)
plt.title('Missing Value Percentage by Feature')
plt.ylabel('% Missing')
plt.tight_layout()
plt.show()

# Target distribution
sns.countplot(x='target', data=train_df)
plt.title('Target Class Distribution')
plt.show()

print("\nTarget value counts:")
print(train_df['target'].value_counts(normalize=True))

# Drop features with >40% missing values (if any)
drop_cols = missing_df[missing_df['Percent'] > 40].index.tolist()
print("\nDropping columns with >40% missing:", drop_cols)
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# Replace -1 with NaN for imputation
train_df.replace(-1, np.nan, inplace=True)
test_df.replace(-1, np.nan, inplace=True)

# Separate features by type
bin_cols = [col for col in train_df.columns if '_bin' in col]
cat_cols = [col for col in train_df.columns if '_cat' in col]
num_cols = [col for col in train_df.columns if col not in bin_cols + cat_cols + ['id', 'target']]

print("\nBinary columns:", bin_cols)
print("\nCategorical columns:", cat_cols)
print("\nNumerical columns:", num_cols)

# Optional: Impute missing values with median (can be replaced with advanced imputation later)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
test_df[num_cols] = imputer.transform(test_df[num_cols])

# Save cleaned versions for modeling
train_df.to_csv('/content/data/train_clean.csv', index=False)
test_df.to_csv('/content/data/test_clean.csv', index=False)

print("\nSaved cleaned training and testing data.")
