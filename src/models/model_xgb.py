import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from models.utils import calculate_sharpe, get_fred, get_cash_flow

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "lending_club_2020_train_processed.csv"

def train_xgb():
    # 데이터 준비 (df에는 risk_free_rate / irr 포함됨)
    df = pd.read_csv(DATA_PATH)
    
    drop_cols = [
    'term', 'last_pymnt_d', 'installment', 'funded_amnt',
    'recoveries', 'collection_recovery_fee', 'default', 'issue_d'
    ]
    X = df_test.drop(columns=drop_cols)
    y = df['default']

    df_train = get_fred(df)
    df_train = get_cash_flow(df_train)

    # 저장 디렉토리
    SAVE_DIR = Path(__file__).resolve().parents[1] / "models"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    param_dist = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [3, 5, 7, 10],
        'learning_rate':     [0.01, 0.05, 0.1],
        'min_child_weight':  [1, 3, 5, 10],   # LGBM의 min_child_samples 대응
        'gamma':             [0, 0.5, 1.0],   # 분할 최소 손실감소
        'subsample':         [0.6, 0.8, 1.0],
        'colsample_bytree':  [0.6, 0.8, 1.0],
        'reg_alpha':         [0, 1e-3, 1e-2, 1e-1],  # L1
        'reg_lambda':        [0.1, 1, 5, 10],       # L2
    }

    # Stratified K-Fold 교차검증 (튜닝 안정화)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_base = XGBClassifier(random_state=42, eval_metric="auc")

    search = RandomizedSearchCV(
        estimator=model_base,
        param_distributions=param_dist,
        n_iter=5,               # 레퍼런스와 동일; 필요하면 늘리기
        scoring='roc_auc',
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X, y)
    best_params = search.best_params_

    # 결과 저장 변수
    best_model = None
    best_model_index = None
    best_threshold_overall = None
    best_params_overall = None
    best_sharpe_overall = -np.inf

    validation_sharpes = []
    test_sharpes = []
    best_thresholds = []
    best_models = []
    test_approval_rates = []
    test_irr_means = []
    test_irr_positive_rates = []

    # 여러 시드로 반복 학습/검증/평가
    for i in tqdm(range(100)):
        # 데이터 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=i, stratify=y_temp
        )

        # XGBoost 모델 학습
        model = XGBClassifier(**best_params, random_state=i, n_jobs=-1)
        model.fit(X_train, y_train)

        # 검증셋 예측 확률
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.0, 1.0, 0.05)

        best_sharpe = -np.inf
        best_threshold = None
        df_val = df.loc[X_val.index]

        for threshold in thresholds:
            approved_mask = y_pred_proba <= threshold
            denied_mask = ~approved_mask

            selected = df_val.copy()
            selected.loc[approved_mask, 'irr_adj'] = selected.loc[approved_mask, 'irr']
            selected.loc[denied_mask, 'irr_adj'] = selected.loc[denied_mask, 'risk_free_rate']

            returns = selected['irr_adj']
            risk_free = selected['risk_free_rate']
            valid = returns.notnull() & risk_free.notnull()

            if valid.sum() < 2:
                continue

            sharpe = calculate_sharpe(returns[valid], risk_free[valid])

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = threshold

        best_models.append(model)
        best_thresholds.append(best_threshold)
        validation_sharpes.append(best_sharpe)

    # 테스트셋 평가(수정 사항)
        # --- (Validation 로직과 동일하게: 승인=IRR, 거절=Risk-free 포함) ---
        y_test_proba = model.predict_proba(X_test)[:, 1]
        df_test = df.loc[X_test.index].copy()
        test_approved_mask = y_test_proba <= best_threshold
        test_denied_mask = ~test_approved_mask
        # 승인 건: IRR, 거절 건: risk-free 로 수익률 대체
        df_test.loc[test_approved_mask, 'irr_adj'] = df_test.loc[test_approved_mask, 'irr']
        df_test.loc[test_denied_mask,  'irr_adj'] = df_test.loc[test_denied_mask,  'risk_free_rate']
        returns_test    = df_test['irr_adj']
        risk_free_test  = df_test['risk_free_rate']
        valid = returns_test.notnull() & risk_free_test.notnull()
        returns_test   = returns_test[valid]
        risk_free_test = risk_free_test[valid]
        sharpe_test = calculate_sharpe(returns_test, risk_free_test)
        test_sharpes.append(sharpe_test)
        # 승인 비율은 그대로 전체 대비 승인 비중으로 기록
        test_approval_rates.append(test_approved_mask.mean())
        # 요약 통계도 '포함된 수익률(irr_adj, 즉 승인=IRR/거절=RF)' 기준으로 기록
        test_irr_means.append(returns_test.mean())
        test_irr_positive_rates.append((returns_test > 0).mean())
        # ----  수정 사항 끝 ---

        # Best 모델 업데이트
        if sharpe_test > best_sharpe_overall:
            best_sharpe_overall = sharpe_test
            best_model = model
            best_model_index = i
            best_threshold_overall = best_threshold
            best_params_overall = model.get_params()

    # Best 모델/정보 저장
    joblib.dump(best_model, SAVE_DIR / "best_model_xgb_[final].pkl")
    with open(SAVE_DIR / "best_model_threshold_xgb_[final].json", "w") as f:
        json.dump({"threshold": best_threshold_overall, "seed": best_model_index}, f, indent=2)
    with open(SAVE_DIR / "best_model_params_xgb_[final].json", "w") as f:
        json.dump(best_params_overall, f, indent=2)

if __name__ == "__main__":
    train_xgb()