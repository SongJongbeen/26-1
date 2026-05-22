import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드
df = pd.read_csv('grouped_risk_perception.csv')

# ==========================================
# 출력 결과를 저장할 텍스트 파일 열기
# ==========================================
with open('analysis_results.txt', 'w', encoding='utf-8') as f:
    
    # ==========================================
    # 1단계: 문화 이론 예측 검증 (Riskiness 및 Unacceptability 비교)
    # ==========================================
    issues_step1 = ['Nuclear power', 'Ozone depletion', 'Genetic engineering', 'Food colourings', 
                    'Mugging', 'Terrorism', 'Car driving', 'Alcoholic drinks']
    
    # SettingWithCopyWarning을 방지하기 위해 .copy() 사용
    df_step1 = df[df['Risk_Issue'].isin(issues_step1)].copy()

    # [추가] Riskiness와 Unacceptability의 평균을 담은 새로운 칼럼 생성
    df_step1['Average_Perception'] = (df_step1['Riskiness'] + df_step1['Unacceptability']) / 2

    # 1-1. Riskiness 막대 그래프
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_step1, x='Risk_Issue', y='Riskiness', hue='Group', order=issues_step1)
    plt.title('Step 1: Riskiness across Groups and Issues')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('step1_riskiness.png', dpi=300)
    plt.close()

    # 1-2. Unacceptability 막대 그래프
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_step1, x='Risk_Issue', y='Unacceptability', hue='Group', order=issues_step1)
    plt.title('Step 1: Unacceptability across Groups and Issues')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('step1_unacceptability.png', dpi=300)
    plt.close()

    # 1-3. [추가] Riskiness & Unacceptability 평균 막대 그래프
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_step1, x='Risk_Issue', y='Average_Perception', hue='Group', order=issues_step1)
    plt.title('Step 1: Average of Riskiness & Unacceptability')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('step1_average.png', dpi=300)
    plt.close()

    # 집단별 평균 수치 확인 (텍스트 파일에 저장)
    f.write("--- Step 1: Riskiness, Unacceptability & Average Mean ---\n")
    f.write(df_step1.groupby(['Risk_Issue', 'Group'])[['Riskiness', 'Unacceptability', 'Average_Perception']].mean().to_string())
    f.write("\n\n")
