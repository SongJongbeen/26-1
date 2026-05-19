import pandas as pd

def normalize_wvs_data(input_file, output_file):
    # 1. 데이터 로드
    df = pd.read_csv(input_file)
    df_norm = df.copy()

    # 반올림할 소수점 자리수 설정
    decimals = 3

    # 2. Grid (그리드) 영역 정규화 (High Grid = 1) 및 반올림
    df_norm['Grid_Q1'] = ((4 - df['Grid_Q1']) / 3).round(decimals)
    df_norm['Grid_Q2'] = ((5 - df['Grid_Q2']) / 4).round(decimals)
    df_norm['Grid_Q3'] = ((4 - df['Grid_Q3']) / 3).round(decimals)
    df_norm['Grid_Q4'] = (2 - df['Grid_Q4']).round(decimals)
    df_norm['Grid_Q5'] = ((5 - df['Grid_Q5']) / 4).round(decimals)
    df_norm['Grid_Q6'] = ((3 - df['Grid_Q6']) / 2).round(decimals)
    df_norm['Grid_Q7'] = ((df['Grid_Q7'] - 1) / 9).round(decimals)
    df_norm['Grid_Q8'] = ((10 - df['Grid_Q8']) / 9).round(decimals)
    df_norm['Grid_Q9'] = ((10 - df['Grid_Q9']) / 9).round(decimals)
    df_norm['Grid_Q10'] = ((10 - df['Grid_Q10']) / 9).round(decimals)
    df_norm['Grid_Q11'] = ((10 - df['Grid_Q11']) / 9).round(decimals)

    # 3. Group (그룹) 영역 정규화 (High Group = 1) 및 반올림
    df_norm['Group_Q1'] = ((4 - df['Group_Q1']) / 3).round(decimals)
    df_norm['Group_Q2'] = ((4 - df['Group_Q2']) / 3).round(decimals)
    df_norm['Group_Q3'] = ((5 - df['Group_Q3']) / 4).round(decimals)
    df_norm['Group_Q4'] = (2 - df['Group_Q4']).round(decimals)
    df_norm['Group_Q5'] = ((10 - df['Group_Q5']) / 9).round(decimals)
    df_norm['Group_Q6'] = ((df['Group_Q6'] - 1) / 9).round(decimals)
    df_norm['Group_Q7'] = ((df['Group_Q7'] - 1) / 9).round(decimals)
    df_norm['Group_Q8'] = ((10 - df['Group_Q8']) / 9).round(decimals)
    df_norm['Group_Q9'] = ((3 - df['Group_Q9']) / 2).round(decimals)
    df_norm['Group_Q10'] = ((df['Group_Q10'] - 1) / 2).round(decimals)

    # Group_Q11은 제외
    if 'Group_Q11' in df_norm.columns:
        df_norm = df_norm.drop(columns=['Group_Q11'])

    # 4. 각 모델별 최종 Grid Index와 Group Index 계산 후 반올림
    grid_cols = [f'Grid_Q{i}' for i in range(1, 12)]
    group_cols = [f'Group_Q{i}' for i in range(1, 11)]

    df_norm['Grid_Index'] = df_norm[grid_cols].mean(axis=1).round(decimals)
    df_norm['Group_Index'] = df_norm[group_cols].mean(axis=1).round(decimals)

    # 5. 결과 저장
    df_norm.to_csv(output_file, index=False, encoding='utf-8')
    print("정규화 및 반올림 처리 완료! 처음 몇 개의 결과를 확인합니다:")
    print(df_norm[['Model', 'Grid_Index', 'Group_Index']])

# 함수 실행
if __name__ == "__main__":
    input_filename = "ai_cultural_prototypes_results.csv"
    output_filename = "ai_cultural_prototypes_normalized.csv"
    
    normalize_wvs_data(input_filename, output_filename)