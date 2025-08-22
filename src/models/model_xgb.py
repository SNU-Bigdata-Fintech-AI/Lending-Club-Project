import json
import numpy as np
import joblib

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from utils import prepare_data, calculate_sharpe, pick_best_threshold_by_sharpe

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "lending_club_2020_train_processed.csv"

def train_xgb():
    # 데이터 준비 (df에는 risk_free_rate / irr 포함됨)
    df, X, y = prepare_data(DATA_PATH)

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

        # 모델 학습 (XGBoost)
        model = XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="auc",     # 검증/테스트와 일관되게 유지
            tree_method="hist",
            random_state=i,
            n_jobs=-1,
            use_label_encoder=False,
        )
        model.fit(X_train, y_train)

        # ── 검증셋에서 최적 threshold (Sharpe 최대) 탐색 ──
        y_val_proba = model.predict_proba(X_val)[:, 1]
        df_val = df.loc[X_val.index].copy()
        best_thr, best_sharpe = pick_best_threshold_by_sharpe(
            y_proba=y_val_proba,
            df_subset=df_val,
            risk_col="risk_free_rate",
            irr_col="irr",
            step=0.05
        )

        best_models.append(model)
        best_thresholds.append(best_thr)
        validation_sharpes.append(best_sharpe)

        # ── 테스트셋 평가: 승인=IRR, 거절=Risk-free 대체 후 Sharpe/통계 ──
        y_test_proba = model.predict_proba(X_test)[:, 1]
        df_test = df.loc[X_test.index].copy()

        test_approved_mask = y_test_proba <= best_thr
        test_denied_mask = ~test_approved_mask

        df_test.loc[test_approved_mask, 'irr_adj'] = df_test.loc[test_approved_mask, 'irr']
        df_test.loc[test_denied_mask,  'irr_adj'] = df_test.loc[test_denied_mask,  'risk_free_rate']

        valid = df_test['irr_adj'].notnull() & df_test['risk_free_rate'].notnull()
        returns_test = df_test.loc[valid, 'irr_adj']
        risk_free_test = df_test.loc[valid, 'risk_free_rate']

        sharpe_test = calculate_sharpe(returns_test, risk_free_test)
        test_sharpes.append(sharpe_test)

        test_approval_rates.append(float(test_approved_mask.mean()))
        test_irr_means.append(float(returns_test.mean()) if not returns_test.empty else np.nan)
        test_irr_positive_rates.append(float((returns_test > 0).mean()) if not returns_test.empty else np.nan)

        # Best 모델 업데이트
        if sharpe_test > best_sharpe_overall:
            best_sharpe_overall = sharpe_test
            best_model = model
            best_model_index = i
            best_threshold_overall = best_thr
            best_params_overall = model.get_params()

    # Best 모델/정보 저장
    joblib.dump(best_model, SAVE_DIR / "best_model_xgb_[final].pkl")
    with open(SAVE_DIR / "best_model_threshold_xgb_[final].json", "w") as f:
        json.dump({"threshold": best_threshold_overall, "seed": best_model_index}, f, indent=2)
    with open(SAVE_DIR / "best_model_params_xgb_[final].json", "w") as f:
        json.dump(best_params_overall, f, indent=2)

if __name__ == "__main__":
    train_xgb()