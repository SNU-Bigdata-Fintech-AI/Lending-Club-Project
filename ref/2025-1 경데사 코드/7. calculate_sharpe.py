import numpy as np
import numpy_financial as npf
import pandas as pd
import fredapi as Fred
from utils.make_cashflow import create_cash_flow

# IRR 계산 함수
def calculate_irr(cash_flow):
    irr_monthly = npf.irr(cash_flow)
    if irr_monthly is not None and not np.isnan(irr_monthly):
        irr_annual = (1 + irr_monthly) ** 12 - 1
    else:
        irr_annual = np.nan
    return irr_annual

# Sharpe Ratio 계산 함수
def calculate_sharpe(returns, risk_free):
    returns     = np.asarray(returns, dtype=float) # 벡터
    risk_free   = np.asarray(risk_free, dtype=float) # 벡터
    mask = ~np.isnan(returns) & ~np.isnan(risk_free)
    excess = returns[mask] - risk_free[mask] # nan이 없을 때
    if excess.size == 0 or np.nanstd(returns, ddof = 1) == 0: # ddof=1 : 표본 표준편차
        return np.nan
    return np.nanmean(excess) / np.nanstd(excess, ddof = 1) 

def calculate_sharpe_from_df(df):
    df['cash_flow'] = df.apply(create_cash_flow, axis=1)
    df['irr'] = df['cash_flow'].apply(calculate_irr)
    sharpe_ratio = calculate_sharpe(df['irr'], df['risk_free_rate'])
    return sharpe_ratio

irr_cache = {}

def get_irr(cash_flow):
    if not isinstance(cash_flow, list) or len(cash_flow) == 0:
        return np.nan
    key = tuple(round(v, 6) for v in cash_flow)
    if key not in irr_cache:
        irr_cache[key] = calculate_irr(cash_flow)
    return irr_cache[key]

def precompute_cashflow_and_irr(df):
    # cash_flow와 irr을 전체 데이터셋에 대해 미리 계산
    df = df.copy()
    df['cash_flow'] = df.apply(create_cash_flow, axis=1)
    df['irr'] = df['cash_flow'].apply(lambda cf: calculate_irr(cf) if isinstance(cf, list) else np.nan)
    df['irr'] = df['irr'].fillna(df['risk_free_rate'])

def compute_sharpe_for_threshold(df, threshold):
    # pred_prob 기준으로 필터만 수행하여 Sharpe 계산
    mask = df['pred_prob'] <= threshold
    selected = df.loc[mask]
    return calculate_sharpe(selected['irr'].values, selected['risk_free_rate'].values)
