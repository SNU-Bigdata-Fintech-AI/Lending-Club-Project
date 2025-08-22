from data.make_dataset import load_data, make_dataset_for_test
from models.utils import calculate_sharpe, get_fred, get_cash_flow


from joblib import load
from pathlib import Path
import pandas as pd
import numpy as np

THRESHOLD = 0.15

SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "models" / "best_model_xgb_[final].pkl"
TEST_PATH = SRC_DIR / "data" / "test" / "lending_club_2020_test.csv" 

if __name__ == "__main__":
    # 1) 원본 로드 & 테스트셋 생성=
    print(f"Loading test data from: {TEST_PATH}")
    df_raw = load_data(str(TEST_PATH))
    df_test = make_dataset_for_test(df_raw)

    # 2) 전처리(학습과 동일 파이프라인 재현) + 피처 매트릭스 생성
    print("Preparing test data...")
    drop_cols = [
    'term', 'last_pymnt_d', 'installment', 'funded_amnt',
    'recoveries', 'collection_recovery_fee', 'default', 'issue_d'
    ]
    X_test = df_test.drop(columns=drop_cols)
    
    df_test = get_fred(df_test)
    df_test = get_cash_flow(df_test)

    # 3) 모델 로드
    print(f"Loading Model XGB from: {MODEL_PATH}")
    best_model_xgb = load(str(MODEL_PATH))

    # 4) 확률 예측
    y_proba = best_model_xgb.predict_proba(X_test)[:, 1]

    y_test_proba = best_model_xgb.predict_proba(X_test)[:, 1]

    # 승인 규칙: 부도확률 <= threshold
    approved_mask = y_test_proba <= THRESHOLD

    # 승인: IRR, 거절: risk-free
    returns = pd.Series(
        np.where(approved_mask, df_test['irr'].values, df_test['risk_free_rate'].values),
        index=df_test.index
    )
    risk_free = df_test['risk_free_rate']

    valid = returns.notnull() & risk_free.notnull()
    sharpe_final = calculate_sharpe(returns[valid], risk_free[valid])

    print("\n✅ Sharpe Ratio:", sharpe_final)