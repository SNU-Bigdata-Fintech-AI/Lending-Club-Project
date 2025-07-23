import pandas as pd
import numpy as np

def create_cash_flow(row):
    loan_amnt = row['loan_amnt']
    installment = row['installment']
    term = int(row['term'])
    default = row['default']
    last_pymnt_num = row['last_pymnt_num']  # 계산해둔 것 (issue_d 기준 몇 번째 달에 default됐는지)
    recoveries = row['recoveries']
    collection_recovery_fee = row['collection_recovery_fee']    
   
    if pd.isna(loan_amnt) or pd.isna(installment) or pd.isna(term):
          print(f"🚨 누락된 필드: {row}")
          return np.nan
    
    cash_flow = [-loan_amnt]  # 대출 실행 시점 투자
    for month in range(1, term + 1):
        if default == 1:
            if month == last_pymnt_num + 1:
                cf = recoveries - collection_recovery_fee
                cash_flow.append(cf)
            elif month <= last_pymnt_num:
                cash_flow.append(installment)
            else:
                cash_flow.append(0)
        else:
                cash_flow.append(installment)

    return cash_flow


cash_flow_cache = {}

def get_cash_flow(row):
    
    try:
        loan_amnt = float(row['loan_amnt'])
        installment = float(row['installment'])
        term = int(row['term'])
        default = int(row['default'])
        last_pymnt_num = int(row['last_pymnt_num'])
        recoveries = float(row['recoveries'])
        collection_recovery_fee = float(row['collection_recovery_fee'])
    except Exception as e:
        print(f"⚠️ 캐시 키 생성 중 오류: {e}")
        return np.nan

    key = (loan_amnt, installment, term, default, last_pymnt_num, recoveries, collection_recovery_fee)

    if key not in cash_flow_cache:
        row = row.copy()
        row['term'] = term  # create_cash_flow 내부에서 int(row['term']) 처리 위해
        cash_flow_cache[key] = create_cash_flow(row)

    return cash_flow_cache[key]


