# 📈 Lending Club 대출 투자 전략 수립 프로젝트  
> 리스크를 고려한 수익률 최적화 전략
## 📌 프로젝트 개요

본 프로젝트는 미국 P2P 대출 플랫폼 **Lending Club**의 실 데이터를 활용하여,  
**대출 승인 여부 판단과 투자 전략 수립**을 위한 머신러닝 기반 모델을 구축하는 것이 목적입니다.  

단순한 연체/상환 여부 예측을 넘어서,  
**Sharpe Ratio(위험 대비 수익률)** 를 최대화하는 **투자 전략 최적화**에 초점을 둡니다.

## 🎯 목적 함수: **Sharpe Ratio 최대화**

Sharpe Ratio는 "위험 단위당 초과 수익"을 측정하는 지표로,  
높을수록 안정적인 수익 구조를 의미합니다.

- 고수익(high IRR)이더라도 **불안정한 연체**가 많으면 Sharpe Ratio는 낮아집니다.
- 상대적으로 낮은 IRR이라도 **안정적인 상환 이력**이 있으면 Sharpe Ratio가 높을 수 있습니다.

따라서, 본 프로젝트의 모델링은 단순 예측이 아닌  
**리스크 필터링 기반의 투자 전략 수립**을 지향합니다.

## 📊 데이터 개요

- 출처: Lending Club 공개 데이터 (2007–2020)
- 총 약 2.9M건 중 약 60%인 **175만 건**이 train/validation에 제공됨
- 나머지 40%는 **최종 테스트 평가용(hold-out set)** 으로 별도 제공 예정

📁 데이터 다운로드: [Google Drive 링크](https://drive.google.com/drive/folders/1TNVGtsFXAmP4cZYTgfkQxtFH6o_rEUp6k)

### 주요 처리 사항
- `loan_status`를 기반으로 부도 여부 이진 라벨 생성
- **대출 심사 시점의 정보만 사용 (미래 정보 제외)**  
  예: 신용점수, 연소득 ✅ / 납입이자, 총상환금액 ❌
- 결측치, 이상치, 범주형 인코딩 등 전처리 필수
- 제외 컬럼: `desc`, `member_id`, `settlement_amount` 등 9개 변수

## 📈 모델링 전략

### 📌 분류 및 회귀 모델 혼합 접근
- **분류 모델**: 부도 여부 예측 (`loan_status` → 0 or 1)
- **회귀 모델**: 예상 수익률 또는 IRR 예측 (Continuous target)
- 다양한 모델 실험 후, validation set에서 성능 비교 → 최적 모델 선택

### 🧠 고려 기술 및 알고리즘
- Logistic Regression / Random Forest / XGBoost / LightGBM
- Hyperparameter Tuning (GridSearchCV, Optuna 등)
- Cross Validation + Bootstrapping (100~1000회 반복)
- Feature Selection 및 Domain-based Engineering


## 💰 재무 지표 및 평가 방식

### ✅ Sharpe Ratio
- 정의: (평균 수익률 - 무위험 수익률) / 수익률의 표준편차
- 무위험 수익률: 당시의 3년/5년 만기 미 국채 수익률로 가정

### ✅ IRR (내부 수익률)
- 원리금 균등상환 기반 월별 현금 흐름 할인 후 계산
- 현금 흐름을 할인해 실제 기대 수익률 평가

## 🛠 프로젝트 구조 (예시) 
<pre><code> <code>project/
├── data/
│ ├── raw/ # 원본 Lending Club CSV
│ └── processed/ # 전처리된 데이터
├── src/
│ ├── data/ # 데이터 로딩 및 전처리
│ ├── features/ # 피처 엔지니어링
│ └── models/ # 학습 및 예측 스크립트
├── notebooks/ # EDA 및 실험 노트북
├── outputs/ # 결과물, 모델 성능 기록
└── README.md</code></code></pre>

## 🤝 협업 및 결과물

| 항목 | 설명 |
|------|------|
| 코드 협업 | GitHub (branch 기반 개발) |
| 보고서 제출 | 2025년 8월 8일 (예정) |
| 최종 결과물 | PPT 발표 자료 + 프로젝트 보고서 (15~20분 분량) |


## 📌 참고 사항

- 단순 IRR 예측이 아닌, **Sharpe Ratio 최적화**가 목표
- 데이터 전처리 및 변수 선택 시 **도메인 지식 고려 필수**
- 모델 선택, 하이퍼파라미터 튜닝, 성능 검증 과정 기록 필수
- Train/Validation/Test 분할은 **재현 가능하게 유지**


## 📎 참고 지표

| 지표 | 설명 |
|------|------|
| Sharpe Ratio | 위험 대비 수익률 |
| IRR (내부 수익률) | 대출의 실제 기대 수익률 |
| Accuracy / AUC | 분류 성능 지표 (보조적으로 활용) |
| Std of Returns | 수익률의 안정성 척도 |


## 🔗 관련 자료
- [Lending Club 데이터셋](https://drive.google.com/drive/folders/1TNVGtsFXAmP4cZYTgfkQxtFH6o_rEUp6k)
- [프로젝트 Notion 공지](https://www.notion.so/237aefd772668019b3e5f5fe6c7c795a)