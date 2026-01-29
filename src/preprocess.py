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
    df = df.copy()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    total_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(total_median)

    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['service_count'] = (df[service_cols] == 'Yes').sum(axis=1)

    df['tenure_group'] = pd.cut(
        df['tenure'], bins=[-1, 6, 12, 24, 48, 72],
        labels=['0-6', '7-12', '13-24', '25-48', '49-72']
    )

    y = (df['Churn'] == 'Yes').astype(int)

    drop_cols = ['customerID', 'Churn']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index, test_size=test_size, stratify=y, random_state=random_state
    )

    # ⭐ 원본 전체를 test 순서대로 가져오기
    original_test = df.loc[test_idx].reset_index(drop=True)

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test, float(total_median)

