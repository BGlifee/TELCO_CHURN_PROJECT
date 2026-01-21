"""
Utilities for preprocessing the Telco Churn dataset.
Provides a `preprocess` function that returns X_train, X_test, y_train, y_test, meta_test.
Key changes:
- Ensure TotalCharges is numeric and impute median for missing values
- Create simple feature engineering (service_count, tenure groups)
- Avoid chained assignments
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Preprocess input dataframe and return train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, meta_test
    """
    df = df.copy()

    # Ensure TotalCharges is numeric (some rows can be empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Impute with median when missing
    total_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(total_median)

    # Simple feature engineering
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    # Create a numeric count of subscribed services (treat 'Yes' as 1)
    df['service_count'] = (df[service_cols] == 'Yes').sum(axis=1)

    # Tenure groups (categorical)
    df['tenure_group'] = pd.cut(
        df['tenure'], bins=[-1, 6, 12, 24, 48, 72], labels=['0-6', '7-12', '13-24', '25-48', '49-72']
    )

    # Target
    y = (df['Churn'] == 'Yes').astype(int)

    # Metadata to keep for scored outputs (keep index alignment)
    meta = df[['customerID', 'MonthlyCharges', 'tenure', 'TotalCharges']].copy()

    # Features: drop identifiers and target
    drop_cols = ['customerID', 'Churn']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Train-test split with stratification
    X_train, X_test, y_train, y_test, meta_train_idx, meta_test_idx = train_test_split(
        X, y, df.index, test_size=test_size, stratify=y, random_state=random_state
    )

    # Keep meta_test aligned with X_test's original indices
    meta_test = meta.loc[meta_test_idx].reset_index(drop=True)

    # Reset index of X/y to keep things clean downstream
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test, meta_test


# When run as a script, do a quick smoke test (not executed during import)
if __name__ == '__main__':
    sample = pd.DataFrame({
        'customerID': ['A', 'B'],
        'tenure': [1, 24],
        'MonthlyCharges': [30.0, 70.0],
        'TotalCharges': ['30.0', '1680.0'],
        'PhoneService': ['Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['No', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['No', 'Yes'],
        'StreamingMovies': ['No', 'No'],
        'Churn': ['No', 'Yes']
    })
    print(preprocess(sample))
