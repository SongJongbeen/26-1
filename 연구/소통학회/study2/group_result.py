import pandas as pd
import numpy as np

# 1. 데이터 불러오기
df = pd.read_csv('study2_risk_perception_results.csv')

# 2. 모델명에 따른 그룹 분류 함수 정의
def map_group(model_name):
    model_name = str(model_name).lower()
    if 'grok' in model_name:
        return 'individualism'
    elif 'claude' in model_name:
        return 'hierarchy'
    else:
        return 'egalitarianism'

# 새로운 'Group' 컬럼 생성
df['Group'] = df['Model'].apply(map_group)

# 3. 평균을 낼 숫자형 데이터 컬럼 추출
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 4. 'Group'과 'Risk_Issue'를 기준으로 그룹화하고 각 항목들의 평균값 계산
grouped_df = df.groupby(['Group', 'Risk_Issue'])[numeric_cols].mean().reset_index()

# 5. 새로운 CSV 파일로 저장
output_file = 'grouped_risk_perception.csv'
grouped_df.to_csv(output_file, index=False)