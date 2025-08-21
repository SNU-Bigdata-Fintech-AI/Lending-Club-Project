import numpy as np
import pandas as pd
import string

# Add default features based on loan status
def add_default_features(df):
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]
    df['default'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)
    return df

# Clean issue date format to datetime
def clean_issue_date_format(df):
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    return df

# Clean term format to extract numeric values
def clean_term_format(df):
    df['term'] = df['term'].str.extract(r'(\d+)').astype(int)
    return df

# Process employment length
def process_emp_length(x):
    if pd.isna(x):
        return np.nan
    elif '< 1' in x:
        return 0.5
    elif '10+' in x:
        return 10.0
    else:
        extracted = pd.to_numeric(pd.Series(x).str.extract(r'(\d+)')[0], errors='coerce')
        return extracted.iloc[0]

def clean_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].apply(process_emp_length)
    return df

# Process revol_util
def preprocess_revol_util(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace('%', '')
    try:
        return float(x)
    except:
        return np.nan

def clean_revol_util(df: pd.DataFrame) -> pd.DataFrame:
    if 'revol_util' in df.columns:
        df['revol_util'] = df['revol_util'].apply(preprocess_revol_util)
    return df

# Encode sub_grade as numeric values
def encode_sub_grade(df: pd.DataFrame) -> pd.DataFrame:
    sub_grades = [f"{l}{n}" for l in string.ascii_uppercase[:7] for n in range(1, 6)]
    sub_grade_map = {grade: idx for idx, grade in enumerate(sub_grades, start=1)}
    
    if 'sub_grade' in df.columns:
        df['sub_grade'] = df['sub_grade'].map(sub_grade_map)
    
    return df

# Handle missing values in the DataFrame
def handle_low_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # 결측률 계산
    missing_ratio = df.isnull().mean()
    
    # 결측률이 0보다 크고 10% 미만인 변수만 선택
    low_missing_cols = missing_ratio[(missing_ratio > 0) & (missing_ratio < 0.1)].index.tolist()

    # emp_length
    if 'emp_length' in df.columns:
        df['emp_length_missing'] = df['emp_length'].isnull().astype(int)
        df['emp_length'] = df['emp_length'].fillna(0)

    # percent_bc_gt_75
    if 'percent_bc_gt_75' in df.columns:
        df['percent_bc_gt_75_missing'] = df['percent_bc_gt_75'].isnull().astype(int)
        df['percent_bc_gt_75'] = df['percent_bc_gt_75'].fillna(0)

    # 결측치 = 0으로 채운 변수들
    fillna0_cols = ['collections_12_mths_ex_med', 'tot_coll_amt', 'chargeoff_within_12_mths',
                    'mo_sin_old_il_acct', 'inq_last_6mths']
    
    for col in fillna0_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 중앙값으로 채운 변수들
    median_replacement_cols = ['tot_cur_bal', 'total_rev_hi_lim', 'acc_open_past_24mths', 'avg_cur_bal',
                               'bc_open_to_buy', 'bc_util', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
                               'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc',
                               'num_accts_ever_120_pd', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
                               'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
                               'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
                               'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 'tax_liens',
                               'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
                               'dti', 'revol_util']
    
    for col in median_replacement_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 이미 처리한 변수들
    already_processed_cols = ['emp_length', 'percent_bc_gt_75'] + fillna0_cols + median_replacement_cols

    # low_missing_cols 중 아직 안 처리한 것
    remaining_cols = [col for col in low_missing_cols if col not in already_processed_cols]

    if remaining_cols:
        df[remaining_cols] = df[remaining_cols].fillna(df[remaining_cols].median())

    return df

# Handle high missing values in numeric columns
def handle_high_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include='number').columns

    missing_ratio = df.isnull().mean()

    high_missing_numeric_cols = [
        col for col in numeric_cols
        if (missing_ratio.get(col, 0) >= 0.1) and not set(df[col].dropna().unique()).issubset({0, 1})
    ]

    for col in high_missing_numeric_cols:
        df[f'{col}_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)

    return df

# Log transform specific features to reduce skewness
def log_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    log_cols = [
        'annual_inc', 'dti', 'max_bal_bc', 'total_rev_hi_lim', 'avg_cur_bal',
        'bc_open_to_buy', 'delinq_amnt', 'tot_hi_cred_lim',
        'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'revol_bal'
    ]
    # -1 이하 값이 있는 행 제거
    df = df[(df[log_cols] > -1).all(axis=1)].copy()

    for col in log_cols:
        df[col] = np.log1p(df[col])
    return df

# Binarize count features to convert them into binary indicators
def binarize_count_features(df: pd.DataFrame) -> pd.DataFrame:
    bin_cols = [
        'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
        'acc_now_delinq', 'tot_coll_amt', 'chargeoff_within_12_mths',
        'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
        'pub_rec_bankruptcies', 'tax_liens'
    ]
    for col in bin_cols:
        df[col] = df[col].fillna(0)
        df[col] = (df[col] >= 1).astype(int)
    return df

def feature_engineering(df):
    df = add_default_features(df)
    df = clean_issue_date_format(df)
    df = clean_term_format(df)
    df = clean_emp_length(df)
    df = clean_revol_util(df)
    df = encode_sub_grade(df)
    df = handle_low_missing_values(df)
    df = handle_high_missing_values(df)

    return df

