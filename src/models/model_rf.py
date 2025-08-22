import json
import numpy as np
import joblib

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from utils import prepare_data, calculate_sharpe, pick_best_threshold_by_sharpe

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "lending_club_2020_train_processed.csv"

def train_rf():
    # 데이터 준비 (df에는 risk_free_rate / irr 포함됨)
    df, X, y = prepare_data(DATA_PATH)

    # 저장 디렉토리
    SAVE_DIR = Path(__file__).resolve().parents[1] / "models"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # RandomForest 하이퍼파라미터 탐색 공간
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Stratified K-Fold 교차검증
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # RandomizedSearchCV 설정
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,               # 필요 시 조정
        scoring="roc_auc",
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

        # 모델 학습 (RandomForest)
        model = RandomForestClassifier(
            **best_params,
            random_state=i,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # ── 검증셋에서 최적 threshold (Sharpe 최대) 탐색 ──
        y_val_proba = model.predict_proba(X_val)[:, 1]
        df_val = df.loc[X_val.index].copy()

        # (XGB 코드의 threshold grid 탐색과 동등) → util의 함수로 동일 로직 실행
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

        # ── 테스트셋 평가: 승인=IRR, 거절=Risk-free로 대체 후 Sharpe 및 통계 ──
        y_test_proba = model.predict_proba(X_test)[:, 1]
        df_test = df.loc[X_test.index].copy()

        test_approved_mask = y_test_proba <= best_thr
        test_denied_mask = ~test_approved_mask

        # 승인 건: IRR, 거절 건: risk-free
        df_test.loc[test_approved_mask, 'irr_adj'] = df_test.loc[test_approved_mask, 'irr']
        df_test.loc[test_denied_mask,  'irr_adj'] = df_test.loc[test_denied_mask,  'risk_free_rate']

        valid = df_test['irr_adj'].notnull() & df_test['risk_free_rate'].notnull()
        returns_test = df_test.loc[valid, 'irr_adj']
        risk_free_test = df_test.loc[valid, 'risk_free_rate']

        sharpe_test = calculate_sharpe(returns_test, risk_free_test)
        test_sharpes.append(sharpe_test)

        # 승인 비율 / 평균 수익률 / 양(>0) 비율
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

    # Best 모델/정보 저장 (파일명은 RF로 구분)
    joblib.dump(best_model, SAVE_DIR / "best_model_rf_[final].pkl")
    with open(SAVE_DIR / "best_model_threshold_rf_[final].json", "w") as f:
        json.dump({"threshold": best_threshold_overall, "seed": best_model_index}, f, indent=2)
    with open(SAVE_DIR / "best_model_params_rf_[final].json", "w") as f:
        json.dump(best_params_overall, f, indent=2)


if __name__ == "__main__":
    train_rf()