from pathlib import Path
import pandas as pd
import os
print(os.getcwd())

def main():
    # 1) 파일 읽기
    df = pd.read_csv('data/raw/lendingclub.csv', low_memory = False) 
    print(df.head(3))
    
    def process_emp_length(x):
        if pd.isna(x):
            return None
        elif '< 1' in x:
            return 0.5
        elif '10+' in x:
            return 10.0
        else:
            extracted = pd.to_numeric(pd.Series(x).str.extract(r'(\d+)')[0], errors='coerce')
            return extracted.iloc[0]

    df['emp_length'] = df['emp_length'].apply(process_emp_length)

    # loan_status를 default로 변환
    # 'Fully Paid', 'Charged Off', 'Default'만 남기기
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]

    # default 컬럼 매핑
    status_mapping = {
        'Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1
    }
    df['default'] = df['loan_status'].map(status_mapping)
    df = df.drop(columns = ['loan_status'])

    # default가 NaN인 경우는 데이터셋에서 제외
    df = df[~df['default'].isnull()]
    print(f"전처리 후 데이터 크기: {df.shape}")

    # 3) interim 폴더에 저장
    out_path = Path('data/interim/lendingclub_clean.csv')
    out_path.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(out_path, index = False)
    print(f"저장 완료: {out_path}")

if __name__ == '__main__':
    main()