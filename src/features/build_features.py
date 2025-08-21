from data.load_data import load_data
from features.feature_engineering import feature_engineering
from features.feature_selector import remove_unnecessary_columns
from features.category_encoder import encode_categoricals
import pandas as pd

PROCESSED_PATH = '../data/processed/lending_club_2020_processed.csv'

def build_features(filepath: str) -> pd.DataFrame:
    df = load_data(filepath)
    df = feature_engineering(df)
    df = remove_unnecessary_columns(df)
    df = encode_categoricals(df)
    return df

if __name__ == "__main__":
    df_features = build_features(PROCESSED_PATH)
    print(df_features.head())