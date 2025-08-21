import pandas as pd
import os
from src.data.load_data import load_data

RAW_PATH = './data/raw/lending_club_2020_raw.csv'
PROCESSED_PATH = './data/processed/lending_club_2020_processed.csv'

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by removing unnecessary columns and handling missing values.

    Args:
        df (pd.DataFrame): Raw DataFrame

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    drop_cols = [col for col in df.columns if 'unnamed' in col.lower() ]
    df = df.drop(columns=drop_cols, errors='ignore')

    df.columns = df.columns.str.strip()
    return df

def make_dataset():
    """
    Load, preprocess, and save the dataset.
    """
    print("Loading raw data...")
    raw_data = load_data(RAW_PATH)

    print("Preprocessing data...")
    df_processed = preprocess_data(raw_data)

    print("Saving processed data...")
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df_processed.to_csv(PROCESSED_PATH, index=False)

if __name__ == "__main__":
    make_dataset()
    print("Dataset creation complete.")

