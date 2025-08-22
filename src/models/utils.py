import numpy as np
import pandas as pd
import numpy_financial as npf

from matplotlib.dates import relativedelta
from fredapi import Fred

# Sharpe 계산 함수
def calculate_sharpe(returns, risk_free_rates):
    excess = returns - risk_free_rates
    if excess.std(ddof=1) == 0:
        return -np.inf
    return excess.mean() / excess.std(ddof=1)

def get_irr(cash_flow):
    if not isinstance(cash_flow, list) or len(cash_flow) == 0:
        return np.nan
    irr_monthly = npf.irr(cash_flow)
    if irr_monthly is None or np.isnan(irr_monthly):
        return np.nan
    return (1 + irr_monthly) ** 12 - 1  # 연환산

def create_cash_flow_from_dates(row):
    try:
        term = int(row['term'])
        default = int(row['default'])
        issue_d = row['issue_d']  # 이미 datetime 형식
        last_pymnt_d = pd.to_datetime(row['last_pymnt_d'])  # 이건 문자열일 수도 있으니 변환
        installment = float(row['installment'])
        funded_amnt = float(row['funded_amnt'])
        recoveries = float(row['recoveries'])
        collection_fee = float(row['collection_recovery_fee'])

        # 첫 현금흐름: 대출 실행
        cash_flow = [-funded_amnt]

        if pd.isnull(issue_d) or pd.isnull(last_pymnt_d):
            return np.nan

        # 몇 회차까지 납입했는지 계산
        delta = relativedelta(last_pymnt_d, issue_d)
        last_pymnt_num = delta.years * 12 + delta.months

        for month in range(1, term + 1):
            if default == 1:
                if month <= last_pymnt_num:
                    cash_flow.append(installment)
                elif month == last_pymnt_num + 1:
                    cash_flow.append(recoveries - collection_fee)
                else:
                    cash_flow.append(0)
            else:
                cash_flow.append(installment)

        return cash_flow

    except Exception as e:
        print(f"[오류] index={row.name}, error={e}")
        return np.nan

def get_nearest_rate(issue_date, rate_series):
    if pd.isnull(issue_date):
        return np.nan
    try:
        idx = rate_series.index.get_indexer([issue_date], method='nearest')[0]
        return rate_series.iloc[idx] / 100  # % 단위 → 소수로 변환
    except Exception as e:
        print(f"Error: {issue_date} ▶ {e}")
        return np.nan

def _term_to_int(term):
    """Extract 36/60 from '36', '36 months', '60개월' 등 문자열에서 숫자만 뽑기"""
    if pd.isnull(term):
        return np.nan
    s = str(term)
    nums = ''.join(ch for ch in s if ch.isdigit())
    return int(nums) if nums else np.nan

def _nearest_pct(series: pd.Series, date):
    """DatetimeIndex 가진 금리(%) 시리즈에서 issue_date와 가장 가까운 값을 반환"""
    if series is None or date is None or pd.isnull(date):
        return np.nan
    idx = series.index.get_indexer([pd.to_datetime(date)], method='nearest')[0]
    return float(series.iloc[idx])

def get_risk_free_rate(issue_date, term, gs3_series, gs5_series):
    """
    36개월이면 GS3, 60개월이면 GS5를 선택해서
    연 단위 소수 (예: 0.0325)로 반환.
    """
    if pd.isnull(issue_date) or pd.isnull(term):
        return np.nan
    t = _term_to_int(term)
    if t == 36:
        val_pct = _nearest_pct(gs3_series, issue_date)
        return val_pct/100.0 if pd.notnull(val_pct) else np.nan
    elif t == 60:
        val_pct = _nearest_pct(gs5_series, issue_date)
        return val_pct/100.0 if pd.notnull(val_pct) else np.nan
    else:
        return np.nan

def get_fred(df: pd.DataFrame) -> pd.DataFrame:
    # FRED API 연결
    df = df.copy()
    fred = Fred(api_key="05ceea53cbc890aa3e4416729c89001b")  # 실제 API 키로 대체해야 함

    # 미국 3년 만기 국채 수익률(GS3) 시계열 불러오기
    gs3_series = fred.get_series('GS3')  # 3-year Treasury
    gs5_series = fred.get_series('GS5')  # 5-year Treasury
    gs3_series.index = pd.to_datetime(gs3_series.index)
    gs5_series.index = pd.to_datetime(gs5_series.index)

    df['risk_free_rate'] = df.apply(lambda r: get_risk_free_rate(r['issue_d'], r['term'], gs3_series, gs5_series), axis=1)

    return df

def get_cash_flow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'])
    df['issue_d'] = pd.to_datetime(df['issue_d'])  # 혹시 모르니 이것도
    df['cash_flow'] = df.apply(create_cash_flow_from_dates, axis=1)

    df['irr'] = df['cash_flow'].apply(get_irr)
    df['irr'] = df['irr'].fillna(df['risk_free_rate'])

    return df