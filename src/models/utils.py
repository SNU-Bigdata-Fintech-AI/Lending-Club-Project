# util.py

import re
import os
import numpy as np
import pandas as pd
import numpy_financial as npf

from pathlib import Path
from dateutil.relativedelta import relativedelta
from fredapi import Fred
from typing import Tuple, Optional, List, Union


# -----------------------------
# Metrics / Financial utilities
# -----------------------------
def calculate_sharpe(returns: pd.Series, risk_free_rates: pd.Series) -> float:
    excess = returns - risk_free_rates
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return -np.inf
    return float(excess.mean() / std)


def get_irr(cash_flow):
    if not isinstance(cash_flow, list) or len(cash_flow) == 0:
        return np.nan
    irr_m = npf.irr(cash_flow)
    if irr_m is None or np.isnan(irr_m):
        return np.nan
    return (1 + irr_m) ** 12 - 1  # 연환산


# -----------------------------
# Cash-flow construction
# -----------------------------
def create_cash_flow_from_dates(row: pd.Series):
    """
    row에 포함되어야 하는 컬럼:
    ['term','default','issue_d','last_pymnt_d','installment','funded_amnt',
     'recoveries','collection_recovery_fee']
    - issue_d / last_pymnt_d 는 datetime 가정(문자열이면 외부에서 to_datetime 처리 권장)
    """
    try:
        # term: '36', '36 months', '60개월' 등에서 숫자만 추출
        term_raw = str(row['term'])
        digits = ''.join(ch for ch in term_raw if ch.isdigit())
        term = int(digits) if digits else int(term_raw)

        default = int(row['default'])
        issue_d = row['issue_d']
        last_pymnt_d = row['last_pymnt_d']
        installment = float(row['installment'])
        funded_amnt = float(row['funded_amnt'])
        recoveries = float(row['recoveries'])
        collection_fee = float(row['collection_recovery_fee'])

        if pd.isnull(issue_d) or pd.isnull(last_pymnt_d):
            return np.nan

        cash_flow = [-funded_amnt]

        delta = relativedelta(last_pymnt_d, issue_d)
        last_pymnt_num = max(0, delta.years * 12 + delta.months)

        for m in range(1, term + 1):
            if default == 1:
                if m <= last_pymnt_num:
                    cash_flow.append(installment)
                elif m == last_pymnt_num + 1:
                    cash_flow.append(recoveries - collection_fee)
                else:
                    cash_flow.append(0.0)
            else:
                cash_flow.append(installment)

        return cash_flow
    except Exception:
        return np.nan


# -----------------------------
# Term parsing helpers
# -----------------------------
def _term_to_months(x):
    """남아있는 외부 참조를 위해 유지 (숫자만 추출)."""
    if pd.isnull(x):
        return np.nan
    m = re.search(r'\d+', str(x))
    return int(m.group()) if m else np.nan


def _term_to_int(term):
    """'36', '36 months', '60개월' 등에서 숫자만 추출하여 int 반환"""
    if pd.isnull(term):
        return np.nan
    s = str(term)
    digits = ''.join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else np.nan


# -----------------------------
# Risk-free (FRED) utilities
# -----------------------------
def _nearest_pct(series: pd.Series, date):
    """DatetimeIndex(일/월 단위) % 시리즈에서 date와 가장 가까운 값을 % 그대로 반환."""
    if series is None or date is None or pd.isnull(date):
        return np.nan
    idx = series.index.get_indexer([pd.to_datetime(date)], method='nearest')[0]
    return float(series.iloc[idx])  # % 값


def get_risk_free_rate(issue_date, term, gs3_series: pd.Series, gs5_series: pd.Series):
    """36개월→GS3, 60개월→GS5 선택해 '연 단위 소수'(예: 0.0325)로 반환"""
    if pd.isnull(issue_date) or pd.isnull(term):
        return np.nan
    t = _term_to_int(term)
    if t == 36:
        v = _nearest_pct(gs3_series, issue_date)
        return v / 100.0 if pd.notnull(v) else np.nan
    elif t == 60:
        v = _nearest_pct(gs5_series, issue_date)
        return v / 100.0 if pd.notnull(v) else np.nan
    else:
        return np.nan


def get_nearest_rate(issue_date, rate_series: pd.Series):
    """
    과거 코드 호환용: 특정 시리즈에서 가장 가까운 날짜 값을 '연 소수'로 반환.
    (rate_series 값이 %라고 가정)
    """
    if pd.isnull(issue_date):
        return np.nan
    try:
        issue_date = pd.to_datetime(issue_date)
        rate_series = rate_series.sort_index()
        idx = rate_series.index.get_indexer([issue_date], method='nearest')[0]
        return float(rate_series.iloc[idx]) / 100.0
    except Exception:
        return np.nan


def get_fred() -> pd.DataFrame:
    """
    GS3, GS5를 %값으로 반환.
    환경변수 FRED_API_KEY가 있으면 사용, 없으면 하드코드 키가 필요.
    """
    api_key = os.getenv("FRED_API_KEY", "05ceea53cbc890aa3e4416729c89001b")
    fred = Fred(api_key=api_key)
    gs3 = fred.get_series('GS3')  # %
    gs5 = fred.get_series('GS5')  # %
    data = pd.DataFrame({"GS3": gs3, "GS5": gs5})
    data.index = pd.to_datetime(data.index)
    return data.sort_index()


# -----------------------------
# Threshold search (Sharpe-max)
# -----------------------------
def pick_best_threshold_by_sharpe(
    y_proba: np.ndarray,
    df_subset: pd.DataFrame,
    risk_col: str = "risk_free_rate",
    irr_col: str = "irr",
    step: float = 0.05,
) -> tuple[float, float]:
    """
    승인 기준: y_proba <= threshold
    승인=IRR, 거절=risk_free로 수익률을 구성 후 Sharpe 계산 → 최댓값 임계값 선택
    """
    thresholds = np.arange(0.0, 1.0, step)
    best_sharpe = -np.inf
    best_thr = None

    for thr in thresholds:
        approved_mask = y_proba <= thr
        denied_mask = ~approved_mask

        tmp = df_subset.copy()
        tmp.loc[approved_mask, 'irr_adj'] = tmp.loc[approved_mask, irr_col]
        tmp.loc[denied_mask,  'irr_adj'] = tmp.loc[denied_mask,  risk_col]

        valid = tmp['irr_adj'].notnull() & tmp[risk_col].notnull()
        if valid.sum() < 2:
            continue

        sharpe = calculate_sharpe(tmp.loc[valid, 'irr_adj'], tmp.loc[valid, risk_col])
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_thr = float(thr)

    return best_thr, best_sharpe


# -----------------------------
# Training data preparation
# -----------------------------
def prepare_training_data(
    data_path: Union[str, Path],
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    1) 전처리된 CSV 로드
    2) FRED GS3/GS5로 risk_free_rate 계산(36/60개월 기준, 연 소수)
    3) cash_flow / irr 생성 (irr 결측은 risk_free_rate로 대체)
    4) X, y 분리
    """
    if drop_cols is None:
        drop_cols = [
            'term', 'last_pymnt_d', 'installment', 'funded_amnt',
            'recoveries', 'collection_recovery_fee', 'default', 'issue_d'
        ]

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # 날짜형 보정(이미 datetime이면 그대로, 문자열일 경우 포맷 가정 없이 파싱)
    if 'issue_d' in df.columns and not np.issubdtype(df['issue_d'].dtype, np.datetime64):
        df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    if 'last_pymnt_d' in df.columns and not np.issubdtype(df['last_pymnt_d'].dtype, np.datetime64):
        # 학습 파이프라인에서 처리되어 있을 가능성이 높지만, 안전하게 파싱 시도
        # (월-연 포맷/연-월-일 포맷 모두 수용하도록 errors='coerce')
        parsed = pd.to_datetime(df['last_pymnt_d'], errors='coerce')
        mask_na = parsed.isna()
        if mask_na.any():
            # 다른 흔한 포맷 시도 (예: %b-%Y)
            parsed2 = pd.to_datetime(df.loc[mask_na, 'last_pymnt_d'], format='%b-%Y', errors='coerce')
            parsed.loc[mask_na] = parsed2
        df['last_pymnt_d'] = parsed

    # FRED 금리 불러와 리스크프리 구성 (연 소수)
    fred_data = get_fred()
    gs3_series = fred_data['GS3']  # %
    gs5_series = fred_data['GS5']  # %

    df.loc[:, 'risk_free_rate'] = df.apply(
        lambda r: get_risk_free_rate(r.get('issue_d'), r.get('term'), gs3_series, gs5_series),
        axis=1
    )

    # 현금흐름/IRR 계산
    need_cols = ['term','default','issue_d','last_pymnt_d','installment',
                 'funded_amnt','recoveries','collection_recovery_fee']
    if all(c in df.columns for c in need_cols):
        df.loc[:, 'cash_flow'] = df.apply(create_cash_flow_from_dates, axis=1)
        df.loc[:, 'irr'] = df['cash_flow'].apply(get_irr)
        df.loc[:, 'irr'] = df['irr'].fillna(df['risk_free_rate'])
    else:
        # 필요한 원천 컬럼이 없으면 IRR 계산 불가 → risk_free 사용
        df.loc[:, 'irr'] = df['risk_free_rate']

    # X, y 분리
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['default'].astype(int)

    return df, X, y