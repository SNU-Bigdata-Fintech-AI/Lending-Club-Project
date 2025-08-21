import pandas as pd

from data.load_data import load_data
from features.feature_engineering import feature_engineering
from features.feature_selector import select_columns
from features.category_encoder import one_hot_encode_columns

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = feature_engineering(df)
    df = one_hot_encode_columns(df)
    df = select_columns(df)

    return df
