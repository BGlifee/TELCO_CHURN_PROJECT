import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def preprocess(df):

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


    # KEEP customerID for dashboard
    meta_cols = ["customerID", "MonthlyCharges", "tenure", "Contract"]
    df_meta = df[meta_cols].copy()

    df_model = df.drop(["customerID"], axis=1)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = le.fit_transform(df_model[col])

    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df_meta,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, meta_test
