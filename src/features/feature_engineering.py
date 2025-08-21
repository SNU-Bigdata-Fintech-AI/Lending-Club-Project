import numpy as np
import pandas as pd
import string
import re

# Add default features based on loan status
def add_default_features(df: pd.DataFrame) -> pd.DataFrame:
    # 슬라이스 방지: 명시적 copy
    mask = df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])
    df = df.loc[mask].copy()

    df.loc[:, 'default'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)
    return df

# Clean issue date format to datetime
def clean_issue_date_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'issue_d' in df.columns:
        df.loc[:, 'issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    return df

# Clean term format to extract numeric values
def clean_term_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'term' in df.columns:
        df.loc[:, 'term'] = pd.to_numeric(df['term'].astype(str).str.extract(r'(\d+)')[0],errors='coerce').fillna(0).astype(int)
    return df

# Process employment length
def process_emp_length(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if '< 1' in s:
        return 0.5
    if '10+' in s:
        return 10.0
    
    # 정규식으로 숫자만 추출
    match = re.search(r'(\d+)', s)
    if match:
        return float(match.group(1))
    return np.nan

def clean_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'emp_length' in df.columns:
        df.loc[:, 'emp_length'] = df['emp_length'].apply(process_emp_length)
        df.loc[:, 'emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce').fillna(0).astype(float)
    return df

# Process revol_util
def preprocess_revol_util(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace('%', '')
    try:
        return float(x)
    except Exception:
        return np.nan

def clean_revol_util(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'revol_util' in df.columns:
        df.loc[:, 'revol_util'] = df['revol_util'].apply(preprocess_revol_util)
    return df

# Encode sub_grade as numeric values
def encode_sub_grade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sub_grades = [f"{l}{n}" for l in string.ascii_uppercase[:7] for n in range(1, 6)]
    sub_grade_map = {grade: idx for idx, grade in enumerate(sub_grades, start=1)}
    if 'sub_grade' in df.columns:
        df.loc[:, 'sub_grade'] = df['sub_grade'].map(sub_grade_map)
    return df

# -----------------------------
# Missing value handling (< 10%)
# -----------------------------
def handle_low_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 결측률 계산
    missing_ratio = df.isnull().mean()

    # 결측률이 0보다 크고 10% 미만인 변수만 선택
    low_missing_cols = missing_ratio[(missing_ratio > 0) & (missing_ratio < 0.1)].index.tolist()

    # emp_length: 라벨 + 0
    if 'emp_length' in df.columns:
        df.loc[:, 'emp_length_missing'] = df['emp_length'].isnull().astype(int)
        df.loc[:, 'emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce').fillna(0.0).astype(float)

    # percent_bc_gt_75: 라벨 + 0
    if 'percent_bc_gt_75' in df.columns:
        df.loc[:, 'percent_bc_gt_75_missing'] = df['percent_bc_gt_75'].isnull().astype(int)
        df.loc[:, 'percent_bc_gt_75'] = df['percent_bc_gt_75'].fillna(0)

    # 결측치 = 0으로 채운 변수들
    fillna0_cols = [
        'collections_12_mths_ex_med', 'tot_coll_amt', 'chargeoff_within_12_mths',
        'mo_sin_old_il_acct', 'inq_last_6mths'
    ]
    present_fill0 = [c for c in fillna0_cols if c in df.columns]
    if present_fill0:
        df.loc[:, present_fill0] = df[present_fill0].fillna(0)

    # 중앙값으로 채운 변수들
    median_replacement_cols = [
        'tot_cur_bal', 'total_rev_hi_lim', 'acc_open_past_24mths', 'avg_cur_bal',
        'bc_open_to_buy', 'bc_util', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc',
        'num_accts_ever_120_pd', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
        'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
        'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
        'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 'tax_liens',
        'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
        'dti', 'revol_util'
    ]
    present_median = [c for c in median_replacement_cols if c in df.columns]
    if present_median:
        med = df[present_median].median(numeric_only=True)
        df.loc[:, present_median] = df[present_median].fillna(med)

    # 이미 처리한 변수들
    already_processed_cols = ['emp_length', 'percent_bc_gt_75'] + present_fill0 + present_median

    # low_missing_cols 중 아직 안 처리한 것 → 중앙값 대체
    remaining_cols = [col for col in low_missing_cols if col not in already_processed_cols]
    if remaining_cols:
        # 수치형만 중앙값
        numeric_remaining = df[remaining_cols].select_dtypes(include='number').columns.tolist()
        if numeric_remaining:
            med2 = df[numeric_remaining].median(numeric_only=True)
            df.loc[:, numeric_remaining] = df[numeric_remaining].fillna(med2)

    return df

# -----------------------------
# Missing value handling (>= 10%)
# -----------------------------
def handle_high_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include='number').columns
    missing_ratio = df.isnull().mean()

    high_missing_numeric_cols = [
        col for col in numeric_cols
        if (missing_ratio.get(col, 0) >= 0.1) and not set(df[col].dropna().unique()).issubset({0, 1})
    ]

    if high_missing_numeric_cols:
        # 결측 라벨을 한 번에 생성하여 concat (fragmentation 방지)
        flag_df = pd.DataFrame(
            {f'{col}_missing': df[col].isnull().astype(int) for col in high_missing_numeric_cols},
            index=df.index
        )
        df = pd.concat([df, flag_df], axis=1)

        # 원 컬럼은 0으로 채움 (loc로 명시)
        df.loc[:, high_missing_numeric_cols] = df[high_missing_numeric_cols].fillna(0)

    return df

# Log transform specific features to reduce skewness
def log_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_cols = [
        'annual_inc', 'dti', 'max_bal_bc', 'total_rev_hi_lim', 'avg_cur_bal',
        'bc_open_to_buy', 'delinq_amnt', 'tot_hi_cred_lim',
        'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'revol_bal'
    ]
    existing = [c for c in log_cols if c in df.columns]
    if not existing:
        return df

    # -1 이하 값이 있는 행 제거 (명시적 copy)
    df = df.loc[(df[existing] > -1).all(axis=1)].copy()

    # 로그 변환
    df.loc[:, existing] = np.log1p(df[existing])
    return df

# Binarize count features to convert them into binary indicators
def binarize_count_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bin_cols = [
        'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
        'acc_now_delinq', 'tot_coll_amt', 'chargeoff_within_12_mths',
        'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
        'pub_rec_bankruptcies', 'tax_liens'
    ]
    present = [c for c in bin_cols if c in df.columns]
    if present:
        df.loc[:, present] = df[present].fillna(0)
        df.loc[:, present] = (df[present] >= 1).astype(int)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # 각 단계에서 copy와 loc를 사용해 체인 인덱싱 방지
    df = add_default_features(df)
    df = clean_issue_date_format(df)
    df = clean_term_format(df)
    df = clean_emp_length(df)
    df = clean_revol_util(df)
    df = encode_sub_grade(df)

    # 결측치 처리 (순서: high → low)
    df = handle_high_missing_values(df)
    df = handle_low_missing_values(df)

    # 로그 변환
    df = log_transform_features(df)

    # 이진화
    df = binarize_count_features(df)

    return df