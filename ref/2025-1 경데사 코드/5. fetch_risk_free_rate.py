from fredapi import Fred
from utils.fred_secrets import load_fred_api_key
import pandas as pd
import numpy as np

def connect_fred_api():
    api_key = load_fred_api_key()
    fred = Fred(api_key=api_key)
    return fred

def load_risk_free_series():
    fred = connect_fred_api()
    rate_series_3y = fred.get_series('GS3')
    rate_series_5y = fred.get_series('GS5')
    return rate_series_3y, rate_series_5y

def fetch_risk_free_rate(issue_d, term, rate_series_3y, rate_series_5y):
    if pd.isnull(issue_d) or pd.isnull(term):
        return np.nan
    
    issue_d = pd.to_datetime(issue_d)

    if term == 36:
        target_series = rate_series_3y
    elif term == 60:
        target_series = rate_series_5y
    else:
        return np.nan
    
    try:
        idx = target_series.index.get_indexer([issue_d], method='nearest')[0]
        rate = target_series.iloc[idx] / 100
        return rate
    except Exception as e:
        print(f"Error fetching rate for {issue_d}: {e}")
        return np.nan

def apply_risk_free_rate(df, rate_series_3y, rate_series_5y):
    df['risk_free_rate'] = df.apply(lambda x: fetch_risk_free_rate(x['issue_d'], x['term'], rate_series_3y, rate_series_5y), axis=1)
    return df