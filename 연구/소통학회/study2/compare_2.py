import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드
df = pd.read_csv('grouped_risk_perception.csv')

# 주석(Annotation)으로 강조할 핵심 데이터 포인트 설정
points_to_annotate_group = [
    ('egalitarianism', 'War'),
    ('individualism', 'Microwave ovens'),
    ('individualism', 'Ozone depletion')
]
# 그룹 없이 전체 평균을 낼 때 강조할 이슈 이름
points_to_annotate_agg = ['War', 'Microwave ovens', 'Ozone depletion']


# =====================================================================
# 시각화 1: 통합 변수 (핵심 추동 요인 vs 최종 위험 인식) - Group별 구분
# =====================================================================
# 두 변수의 평균을 새로운 컬럼으로 생성
df['Core_Factors'] = (df['Dread'] + df['Harm_to_future_generations']) / 2
df['Overall_Perception'] = (df['Riskiness'] + df['Unacceptability']) / 2

plt.figure(figsize=(10, 7))
sns.regplot(data=df, x='Core_Factors', y='Overall_Perception', scatter=False, color='grey', line_kws={'linestyle':'--', 'alpha':0.6})
sns.scatterplot(data=df, x='Core_Factors', y='Overall_Perception', hue='Group', s=100, alpha=0.8)

for index, row in df.iterrows():
    if (row['Group'], row['Risk_Issue']) in points_to_annotate_group:
        plt.annotate(f"{row['Group'].capitalize()}:\n{row['Risk_Issue']}", 
                     (row['Core_Factors'], row['Overall_Perception']),
                     textcoords="offset points", xytext=(0, 15), ha='center', fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color='black'))

plt.title('1. Combined Variables: Core Factors vs Overall Perception (by Group)', fontsize=14, pad=15)
plt.xlabel('Core Factors (Average of Dread & Harm)')
plt.ylabel('Overall Perception (Average of Riskiness & Unacceptability)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title='Cultural Group', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('step2_1_combined_by_group.png', dpi=300, bbox_inches='tight')  # 파일 1 저장
plt.close()


# =====================================================================
# 시각화 2 수정: 개별 변수 4분할 (Dread/Harm vs Riskiness/Unacceptability) 
# -> 핵심 3개 점만 표시 & 1~5 스케일 축 고정
# =====================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
plots_info = [
    ('Dread', 'Riskiness', axs[0, 0]),
    ('Dread', 'Unacceptability', axs[0, 1]),
    ('Harm_to_future_generations', 'Riskiness', axs[1, 0]),
    ('Harm_to_future_generations', 'Unacceptability', axs[1, 1])
]

# 우리가 관심 있는 3개의 데이터 포인트만 마스킹하여 필터링
mask = df.apply(lambda row: (row['Group'], row['Risk_Issue']) in points_to_annotate_group, axis=1)
df_filtered = df[mask]

# 다른 산점도와 색상 톤을 맞추기 위한 커스텀 팔레트 (Seaborn 기본 색상 매핑)
custom_palette = {'egalitarianism': '#0173b2', 'individualism': '#029e73', 'hierarchy': '#de8f05'}

for x_col, y_col, ax in plots_info:
    # 1. 전체 데이터(df)를 사용해 회귀선(추세선) 그리기 (점은 숨김 처리)
    sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color='grey', line_kws={'linestyle':'--', 'alpha':0.6}, ax=ax)
    
    # 2. 필터링된 데이터(df_filtered)만 사용해 강조할 점 3개만 그리기
    sns.scatterplot(data=df_filtered, x=x_col, y=y_col, hue='Group', palette=custom_palette, s=150, alpha=0.9, ax=ax)
    
    # 3. Y축 왜곡 방지를 위해 1~5 스케일에 맞게 축 범위 강제 고정
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    
    # 4. 강조된 점들에 텍스트 주석 추가
    for index, row in df_filtered.iterrows():
        # 오존층 파괴 라벨이 차트 바깥으로 나가는 것을 방지
        y_offset = -25 if (row['Risk_Issue'] == 'Ozone depletion' and y_col == 'Riskiness') else 15
        ax.annotate(f"{row['Group'].capitalize()}:\n{row['Risk_Issue']}", 
                     (row[x_col], row[y_col]), textcoords="offset points", xytext=(0, y_offset), 
                     ha='center', fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
                     
    ax.set_title(f"{x_col.replace('_', ' ')} vs {y_col}", fontsize=12, fontweight='bold')
    ax.set_xlabel(x_col.replace('_', ' '))
    ax.set_ylabel(y_col)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    if ax.get_legend():
        ax.get_legend().remove()

# 전체 범례 추가 (필터링된 데이터셋에 존재하는 그룹만 범례에 표시됨)
handles, labels = axs[0,0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, title='Cultural Group', loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)
    
fig.suptitle('2. Individual Relationships (Highlighting Key Points Only)', fontsize=16, fontweight='bold', y=1.06)

plt.tight_layout()
plt.savefig('step2_2_individual_by_group_highlighted.png', dpi=300, bbox_inches='tight')
plt.close()


# =====================================================================
# 시각화 3: 그룹을 무시하고(깡 평균) '이슈별' 개별 변수 4분할 분석
# =====================================================================
# Group 변수를 배제하고 Risk_Issue를 기준으로 각 항목들의 전체 평균 계산
df_agg = df.groupby('Risk_Issue')[['Dread', 'Harm_to_future_generations', 'Riskiness', 'Unacceptability']].mean().reset_index()

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
plots_info_agg = [
    ('Dread', 'Riskiness', axs[0, 0]),
    ('Dread', 'Unacceptability', axs[0, 1]),
    ('Harm_to_future_generations', 'Riskiness', axs[1, 0]),
    ('Harm_to_future_generations', 'Unacceptability', axs[1, 1])
]

for x_col, y_col, ax in plots_info_agg:
    sns.regplot(data=df_agg, x=x_col, y=y_col, scatter=False, color='grey', line_kws={'linestyle':'--', 'alpha':0.6}, ax=ax)
    sns.scatterplot(data=df_agg, x=x_col, y=y_col, s=120, color='teal', alpha=0.8, ax=ax)
    
    # ==========================================
    # [핵심 수정] 원래 척도(1~5)에 맞게 축 범위 강제 고정
    # 점들이 테두리에 잘리지 않게 0.8 ~ 5.2 정도로 여백만 살짝 줍니다.
    # ==========================================
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    
    for index, row in df_agg.iterrows():
        if row['Risk_Issue'] in points_to_annotate_agg:
            ax.annotate(f"{row['Risk_Issue']}", 
                         (row[x_col], row[y_col]), textcoords="offset points", xytext=(0, 12), 
                         ha='center', fontsize=10, fontweight='bold', color='darkred',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkred", lw=1, alpha=0.9),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color='darkred'))
            
    ax.set_title(f"Aggregated: {x_col.replace('_', ' ')} vs {y_col}", fontsize=12, fontweight='bold')
    ax.set_xlabel(f"Average {x_col.replace('_', ' ')}")
    ax.set_ylabel(f"Average {y_col}")
    ax.grid(True, linestyle=':', alpha=0.6)

fig.suptitle('3. Base Psychometric Paradigm (Fixed Axes to 1-5 Scale)', fontsize=16, fontweight='bold', y=1.03)

plt.tight_layout()
plt.savefig('step2_3_individual_aggregated.png', dpi=300, bbox_inches='tight')
plt.close()