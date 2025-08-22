from data.make_dataset import make_dataset_for_test
from data.load_data import load_data
from models.utils import prepare_data_for_test, calculate_sharpe
from joblib import load
from pathlib import Path

THRESHOLD = 0.15

SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "models" / "best_model_xgb_[final].pkl"
TEST_PATH = SRC_DIR / "data" / "test" / "lendingclub_test.csv" 

if __name__ == "__main__":
    # 1) 원본 로드 & 테스트셋 생성=
    df_raw = load_data(str(TEST_PATH))
    df_test_src = make_dataset_for_test(df_raw)

    # 2) 전처리(학습과 동일 파이프라인 재현) + 피처 매트릭스 생성
    df_test, X_test = prepare_data_for_test(df_test_src)

    # 3) 모델 로드
    best_model_xgb = load(str(MODEL_PATH))

    # 4) 확률 예측
    y_proba = best_model_xgb.predict_proba(X_test)[:, 1]

    # 5) 임계값 적용(부도확률 <= THRESHOLD → 승인)
    approved_mask = y_proba <= THRESHOLD
    denied_mask = ~approved_mask

    # 승인 = IRR, 거절 = Risk-free 로 수익률 구성
    df_test = df_test.copy()
    df_test.loc[approved_mask, 'irr_adj'] = df_test.loc[approved_mask, 'irr']
    df_test.loc[denied_mask,  'irr_adj'] = df_test.loc[denied_mask,  'risk_free_rate']

    # 6) Sharpe Ratio 계산 (결측 제외)
    valid = df_test['irr_adj'].notnull() & df_test['risk_free_rate'].notnull()
    sharpe = calculate_sharpe(df_test.loc[valid, 'irr_adj'], df_test.loc[valid, 'risk_free_rate'])

    # 7) 결과 출력(Sharpe만)
    print(f"✅Sharpe Ratio (threshold={THRESHOLD}): {sharpe}")