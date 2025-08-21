from pathlib import Path
import pandas as pd

from data.load_data import load_data
from features.build_features import build_features

# ---- Relative paths (based on this file) ----
BASE_DIR: Path = Path(__file__).resolve().parent          # .../src
RAW_PATH: Path = BASE_DIR / "raw" / "lending_club_2020_train.csv"
PROCESSED_PATH: Path = BASE_DIR / "processed" / "lending_club_2020_train_processed.csv"

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by removing unnecessary columns and handling whitespace in headers.
    """
    drop_cols = [col for col in df.columns if 'unnamed' in col.lower()]
    df = df.drop(columns=drop_cols, errors='ignore')
    df.columns = df.columns.str.strip()
    return df

def make_dataset() -> None:
    """
    Load, preprocess, build features, and save the dataset.
    """
    print(f"Loading raw data from: {RAW_PATH}")
    raw_data = load_data(str(RAW_PATH))

    print("Preprocessing data...")
    df_processed = preprocess_data(raw_data)

    print("Building features...")
    df_build = build_features(df_processed)

    print(f"Saving processed data to: {PROCESSED_PATH}")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_build.to_csv(PROCESSED_PATH, index=False)

if __name__ == "__main__":
    make_dataset()
    print("Dataset creation complete.")