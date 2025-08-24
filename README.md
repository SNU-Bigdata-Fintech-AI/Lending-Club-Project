# 📈 Lending Club 대출 투자 전략 (Sharpe Ratio 최적화)

> **리스크를 고려한 수익률 최적화**를 목표로, Lending Club 데이터(2007–2020)를 활용해 **부도확률 예측 → 임계값(승인기준) 조정 → 현금흐름 기반 IRR → Sharpe Ratio 최대화** 파이프라인을 구축합니다.


## 📌 핵심 요약

- **목표지표**: 위험조정수익률인 **Sharpe Ratio** 최대화 (무위험수익률: 대출 발행 시점의 美 3/5년 국채수익률)
- **모델**: Random Forest / XGBoost / LightGBM, **임계값 기반 승인전략 + 반복 검증**
- **대표 결과**: **XGBoost**가 Out-of-sample에서 가장 높은 Sharpe Ratio(예: **≈0.116**)로 우수
- **최적 임계값**: 세 모델 공통으로 **0.15** 인근에서 최고 성과 관찰 (승인률·분산의 균형)
- **설명력**: `sub_grade`, `dti`, `acc_open_past_24mths`, `annual_inc` 등 영향 큼
- **보완 포인트**: **조기상환/듀레이션 기반 무위험수익률 보정** 시 Sharpe 계산 타당성 상승


## 📂 리포지토리 구조

```
project/
├─ data/
│  ├─ raw/           # 원본 CSV
|  ├─ test/          # Out-of-sample 테스트 데이터
│  └─ processed/     # 전처리 산출물
├─ src/
│  ├─ data/          # 로딩/스플릿/결측치·이상치 처리
│  ├─ features/      # 파생변수/인코딩/스케일링
│  └─ models/        # 학습/예측/임계값 탐색/현금흐름 시뮬
├─ notebooks/        # EDA 및 실험 기록
├─ doc/              # 결과 표/그림, 실험 로그
├─ main.py           # 데이터를 통한 Sharpe Ratio 추정
└─ README.md
```

> 전처리 원칙: **의사결정 시점**에 알 수 없는 **내생 변수(사후 정보)** 제외, 결측률 구간별 이원화 전략, 범주형 One-hot/순서형 처리, 고왜도 로그변환 등.


## 🧭 문제정의 & 목적함수

**Sharpe Ratio**

$$
Sharpe\\ Ratio = \\frac{\\overline{R - R_f}}{\\sigma(R - R_f)}
$$

- **R**: 대출별 **현금흐름 시뮬레이션 기반 IRR(연환산)**  
- **R_f**: 발행시점 美 3/5년 국채수익률 (FRED 매칭)  
- **σ**: 초과수익률의 표본표준편차(ddof=1)  
→ **예측정확도**가 아니라 **투자성과(위험대비)**를 직접 최적화하는 설계.


## 📊 데이터 개요

- 출처: **Lending Club** 공개 데이터 (2007–2020)
- 약 **175만 건**(Train/Valid) + 별도 **Hold-out Test**
- 📁 데이터 다운로드: [Google Drive 링크](https://drive.google.com/drive/folders/1TNVGtsFXAmP4cZYTgfkQxtFH6o_rEUp6k)

### 주요 처리 사항
- `loan_status` 기반 **부도 여부 이진 라벨** 생성
- **심사 시점 정보만 사용**(사후 데이터 제외)  
  예: 신용점수, 연소득 ✅ / 납입이자, 총상환금액 ❌
- 결측치/이상치 처리, 범주형 인코딩, 스케일링
- 제외 컬럼: `desc`, `member_id`, `settlement_amount` 등


## 🔧 전처리 하이라이트

- **라벨링**:  
  - Fully Paid → 0, Charged Off/Default → 1  
  - *Current/Late/Grace/Issued 등은 제외*
- **형 변환/인코딩**  
  - `term`/`revol_util` 문자열 → 수치, `emp_length`의 `< 1` → 0.5년  
  - `sub_grade`(A1~G5) → 1~35 순서형  
  - `addr_state`, `home_ownership`, `purpose`, `verification_status` → One-hot
- **결측치**  
  - 결측률 ≥10%: 0 대체 + **결측 더미 추가**  
  - 결측률 <10%: **중앙값 대체**
- **현금흐름**  
  - `funded_amnt` 초기 유출 → 월 `installment` 유입  
  - 부도 시 **추심 순유입**(recoveries−fee) 반영 후 IRR 계산


## 🧠 모델링 & 임계값 전략

- **모델**: RF / XGB / LGBM (대용량·혼합형 변수·비선형/상호작용에 강건)
- **데이터 분할**: 6:2:2 (Train/Valid/Test), **여러 회 반복**으로 안정성 확보
- **임계값(threshold)**:  
  - 예측 **부도확률 ≤ τ → 승인**  
  - **Sharpe 최대화 기준**으로 τ 탐색, 세 모델 공통 **τ≈0.15** 최적
- **튜닝**: 학습률·깊이·규제·subsample 등 **Randomized/Optuna** 기반


## 📈 대표 결과 (Out-of-sample 예시)

| Model        | 최적 임계값(공통) | Test Sharpe (예) | 메모 |
|--------------|-------------------|------------------|------|
| RandomForest | ~0.15             | ~0.094–0.134     | 승인률 낮고 Positive IRR Ratio↑, 분산효과 제한 |
| **XGBoost**  | **~0.15**         | **~0.116**       | **최고 Sharpe** (승인모수 확대+분산효과) |
| LightGBM     | ~0.15             | ~0.111–0.115     | XGB와 근접 성과 |

> 벤치마크(**Approve All**)의 Sharpe는 **음(-)** 혹은 매우 낮아 **위험 차등화** 없는 일괄승인은 비효율적임을 확인.


## 🪄 해석 가능성 (SHAP 예)

- 공통 중요 변수: **`sub_grade`, `loan_amnt`, `dti`, `acc_open_past_24mths`** 등  
- **Boosting**은 `dti`·최근 신용활동(단기)와 `emp_length`·`home_ownership`·`mort_acc`(장기)를 함께 반영해 **위험 식별**에 유리.


## 🧮 고도화 제안: 조기상환 & 듀레이션 반영

- 대출은 **원리금균등**으로 듀레이션이 만기보다 짧음 → 각 건의 **맥컬리 듀레이션**으로 실질 만기 추정  
- 듀레이션에 맞춰 **무위험수익률 보간** → **Sharpe 계산의 타당성** 향상(조기상환 영향 반영)


## 🛠 설치 & 실행 (예시)

```bash
# 1) 환경
conda create -n lc-sharpe python=3.10 -y
conda activate lc-sharpe
pip install -r requirements.txt

# 2) 데이터 배치
# data/raw/ 에 Lending Club CSV 배치

# 3) 파이프라인(예시)
python src/main.py   --model xgb   --threshold 0.15   --n_repeats 100   --train_path data/raw/LC.csv   --out_dir outputs/xgb_t015

# 4) 결과물
# outputs/ 아래 Sharpe/IRR 분포, 승인률, SHAP, ROC/CM, 로그 저장
```

> 실제 인자명/경로는 리포지토리 스크립트에 맞게 조정하세요. (본 README는 **실행 예시 템플릿**입니다.)


## 🧪 재현 체크리스트

-  **Seed 고정** 및 데이터 Split 기록  
-  **사용 변수/제외 변수 목록** 커밋  
-  **임계값 탐색 간격**(0.05 → **0.01 권장**)과 각 구간별 성과 로그  
-  **현금흐름 가정**(추심/수수료/조기상환) 명시 및 버전관리


## 🗺️ 로드맵

- **정상대출을 부도로 과예측**하는 비율 완화(정밀도 개선) → `class_weight` 등 비용민감 학습  
- **임계값 탐색** 0.01 간격으로 세분화 → 진정한 최적 Sharpe 탐색  
- **설명변수 확장**: 지역·거시 변수(예: Zillow 주택가격, 실업률 등) 추가


## 📊 데이터 & 참고

- 데이터: Lending Club 공개(2007–2020)  
- 무위험수익률: 발행시점 **美 국채수익률(3y/5y)** 매칭  
- 상세한 방법·결과·그림은 **팀 보고서**를 참조


## 👥 팀

서울대학교 빅데이터 AI 핀테크 고급 전문가 과정 11기 (2조)  
강수정 · 배기태 · 심준선 · 이강산 · 이선유 · 전상언 · 황정현 (2025.08)


## 📝 참고 문헌
[단행본]
- Fabozzi, F. J.(2016). Bond markets. anlysis, and strategies(9th ed.). Pearson.
- Gutierrez, A., & Mathieson, D.(2017). Optimizing investment strategy in peer-to-peer lending(CS229 Final Report). Stanford University.
  
[논문]
- Mo, L., & Yae, J.(2022) Lending Club meets Zillow: local housing prices and default risk of peer-to-peer loans. Applied Economics, 54(35), 4101-4112
- Sharpe, W. F.(1966). Mutual fund performance. the Journal of Business. 39(1), 119-138
- Sharpe, W. F.(1994). The Sharpe ratio. The Journal of Portfolio Management, 21(1), 49-58
