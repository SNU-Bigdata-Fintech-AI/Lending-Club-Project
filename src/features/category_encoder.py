import pandas as pd

def one_hot_encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dummy_cols = ['addr_state', 'home_ownership', 'purpose', 'verification_status']
    for col in dummy_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(" ", "_")

    return pd.get_dummies(df, columns=[col for col in dummy_cols if col in df.columns], dtype='uint8', dummy_na=True)
 