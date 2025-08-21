import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a CSV or Parquet file.

    Args:
        filepath (str): Path to the data file

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, low_memory=False)
    else:
        raise ValueError("Data file must be a CSV file.")