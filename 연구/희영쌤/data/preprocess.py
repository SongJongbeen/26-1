import pandas as pd
import os

# 연도별 분할 파일 리스트
files = [
    "data/2020년_이상데이터 변경.csv",
    "data/2021년_이상데이터 변경.csv",
    "data/2022년_이상데이터변경.csv",
    "data/2023년_이상데이터 변경.csv",
    "data/2024년_이상데이터 변경.csv"
]

dfs = []
for f in files:
    if os.path.exists(f):
        # 한글 인코딩 자동 처리
        try:
            df = pd.read_csv(f, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding='cp949')
            
        # 파일명에서 연도 추출 후 누락 방지
        year = int(f.split('/')[-1][:4])
        if '연도' not in df.columns:
            df['연도'] = year
        dfs.append(df)

# 5개년 패널 데이터 생성 완료
combined_df = pd.concat(dfs, ignore_index=True)

# 패널 데이터 저장
combined_df.to_csv('data/concat_data.csv', index=False, encoding='utf-8')
