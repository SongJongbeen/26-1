import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 (실제 작업 환경의 파일명에 맞게 수정 필요)
# (만약 파일명이 grouped_risk_perception_3.csv 라면 아래 파일명을 수정해주세요)
df = pd.read_csv('grouped_risk_perception.csv')

# 일관된 색상 테마 적용 (평등주의: 파랑, 개인주의: 초록, 계층주의: 주황)
custom_palette = {'egalitarianism': '#0173b2', 'individualism': '#029e73', 'hierarchy': '#de8f05'}
sns.set_theme(style="whitegrid")

# =========================================================
# 2-1. '미래 세대에 미치는 해악'과 '파국적 잠재성' (Mugging vs Environment)
# =========================================================
issues_2_1 = ['Mugging', 'Nuclear power', 'Ozone depletion']
df_2_1 = df[df['Risk_Issue'].isin(issues_2_1)]
df_melt_2_1 = df_2_1.melt(id_vars=['Group', 'Risk_Issue'], 
                          value_vars=['Harm_to_future_generations', 'Catastrophic_potential'],
                          var_name='Factor', value_name='Score')

# 변수명 가독성 개선
df_melt_2_1['Factor'] = df_melt_2_1['Factor'].str.replace('_', ' ').str.title()

g = sns.catplot(data=df_melt_2_1, kind='bar', x='Risk_Issue', y='Score', hue='Group', 
                col='Factor', palette=custom_palette, height=5, aspect=1.2, edgecolor='black')

g.set(ylim=(0.8, 5.2)) # 1~5 척도에 맞게 축 고정
g.fig.suptitle("3-1. 'Harm' & 'Catastrophic Potential': Mugging vs. Environmental Threats", y=1.08, fontweight='bold', fontsize=14)

for ax in g.axes.flat:
    ax.set_ylabel('Score (1-5)')
    ax.set_xlabel('Risk Issue')
    
plt.savefig('step3_1_harm_catastrophic.png', dpi=300, bbox_inches='tight')
plt.close()


# =========================================================
# 2-2. 자동차 운전(Car driving)에 대한 시각 차이
# =========================================================
df_car = df[df['Risk_Issue'] == 'Car driving']
factors_car = ['Delayed_effects', 'Catastrophic_potential', 'Harm_to_future_generations']
df_melt_car = df_car.melt(id_vars='Group', value_vars=factors_car, var_name='Factor', value_name='Score')

plt.figure(figsize=(9, 6))
ax = sns.barplot(data=df_melt_car, x='Factor', y='Score', hue='Group', palette=custom_palette, edgecolor='black')

plt.ylim(0.8, 5.2) # 1~5 척도에 맞게 축 고정
plt.title("3-2. Different Perspectives on 'Car Driving'", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Score (1-5)")
plt.xlabel("Risk Characteristics (Outrage Factors)")

# 텍스트 분석 내용이 한눈에 보이게 X축 라벨 커스텀
custom_labels = [
    'Delayed Effects\n(Match: Egal > Ind)', 
    'Catastrophic Potential\n(Match: Egal > Ind)', 
    'Harm to Future Gen\n(Mismatch: Egal < Others)'
]
# [경고 해결] 라벨을 지정하기 전에 눈금(ticks) 위치를 먼저 명시적으로 고정해줍니다.
ax.set_xticks(range(len(custom_labels)))
ax.set_xticklabels(custom_labels)

plt.legend(title='Cultural Group', loc='upper right')

plt.tight_layout()
plt.savefig('step3_2_car_driving.png', dpi=300, bbox_inches='tight')
plt.close()


# =========================================================
# 2-3. 원자력(Nuclear power)에 대한 시각 차이 (합치 vs 어긋남 분리)
# =========================================================
df_nuc = df[df['Risk_Issue'] == 'Nuclear power']

# 서사를 극대화하기 위해 '예측 합치 요인'과 '예측 빗나간 요인'을 분리
factors_match = ['Dread', 'Involuntariness', 'Severity']
factors_mismatch = ['Catastrophic_potential', 'Harm_to_future_generations', 'Delayed_effects']

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: 예측 합치 (Egalitarianism이 압도적으로 높음)
df_nuc_match = df_nuc.melt(id_vars='Group', value_vars=factors_match, var_name='Factor', value_name='Score')
df_nuc_match['Factor'] = df_nuc_match['Factor'].str.replace('_', ' ').str.title()
sns.barplot(data=df_nuc_match, x='Factor', y='Score', hue='Group', palette=custom_palette, edgecolor='black', ax=axs[0])

axs[0].set_ylim(0.8, 5.2)
axs[0].set_title("Matches Expectation\n(Egalitarianism > Others)", fontsize=13, fontweight='bold', color='darkgreen')
axs[0].set_xlabel("")
axs[0].set_ylabel("Score (1-5)")

# Subplot 2: 예측 어긋남 (Egalitarianism이 같거나 더 낮음)
df_nuc_mismatch = df_nuc.melt(id_vars='Group', value_vars=factors_mismatch, var_name='Factor', value_name='Score')
df_nuc_mismatch['Factor'] = df_nuc_mismatch['Factor'].str.replace('_', ' ').str.title()
sns.barplot(data=df_nuc_mismatch, x='Factor', y='Score', hue='Group', palette=custom_palette, edgecolor='black', ax=axs[1])

axs[1].set_ylim(0.8, 5.2)
axs[1].set_title("Violates Expectation\n(Egalitarianism <= Others)", fontsize=13, fontweight='bold', color='darkred')
axs[1].set_xlabel("")
axs[1].set_ylabel("")

# 범례 정리 (개별 그래프의 범례를 지우고 최상단 중앙에 하나만 배치)
axs[0].get_legend().remove()
axs[1].get_legend().remove()
handles, labels = axs[0].get_legend_handles_labels()

# 범례 라벨 첫 글자 대문자 처리
labels = [label.capitalize() for label in labels]
fig.legend(handles, labels, title='Cultural Group', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)

fig.suptitle("3-3. Detailed Perspectives on 'Nuclear Power'", fontsize=16, fontweight='bold', y=1.12)

plt.tight_layout()
plt.savefig('step3_3_nuclear_power.png', dpi=300, bbox_inches='tight')
plt.close()
