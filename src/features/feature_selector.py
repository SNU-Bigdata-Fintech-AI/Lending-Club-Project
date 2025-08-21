import numpy as np
import pandas as pd

def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [ 
    'loan_status',
    'last_fico_range_high', 'last_fico_range_low',
    'application_type', 'grade', 'verification_status_joint', 'hardship_loan_status', 'hardship_type', 'hardship_reason', 'hardship_status',
    'deferral_term', 'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
    'hardship_length', 'hardship_dpd', 'orig_projected_additional_accrued_interest',
    'hardship_amount', 'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
    'sec_app_revol_util', 'revol_bal_joint', 'sec_app_fico_range_low', 'sec_app_fico_range_high',
    'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
    'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
    'sec_app_collections_12_mths_ex_med', 'annual_inc_joint', 'dti_joint', 'mths_since_last_record',
    'mths_since_recent_bc_dlq', 'mths_since_last_major_derog', 'next_pymnt_d', 'inq_fi',
    'total_cu_tl', 'emp_title', 'num_actv_bc_tl', 'hardship_flag', 'title',
    'earliest_cr_line', 'funded_amnt_inv', 'id',
    'initial_list_status', 'int_rate', 'last_credit_pull_d',
    'last_pymnt_amnt', 'out_prncp', 'out_prncp_inv', 'policy_code',
    'pymnt_plan', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int',
    'total_rec_late_fee', 'total_rec_prncp', 'url', 'zip_code', 'debt_settlement_flag',
    'desc', 'member_id', 'verified_status_joint', 'sec_app_mths_since_last_major_derog',
    'disbursement_method', 'debt_settlement_flag_date', 'settlement_status', 'settlement_date',
    'settlement_amount', 'settlement_percentage', 'settlement_term', 'application type'
    ]

    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # 결측치가 있는 행 제거
    return df.dropna(subset=['last_pymnt_d', 'issue_d']).copy()

def remove_inf_values(df: pd.DataFrame) -> pd.DataFrame:
    # df 전체에서 유한한 값만 남기고, inf/-inf가 포함된 행은 제거
    # 숫자형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 숫자형 컬럼 중에서 inf/-inf 있는 행 삭제
    return df[np.isfinite(df[numeric_cols]).all(axis=1)].copy()

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_unnecessary_columns(df)
    df = remove_missing_values(df)
    df = remove_inf_values(df)
    
    return df