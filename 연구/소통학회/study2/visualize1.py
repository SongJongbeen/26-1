import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# 1. 데이터 로드 및 Prototype 매핑
df = pd.read_csv('study2_risk_perception_results.csv')

def map_prototype(model_name):
    if 'grok' in model_name.lower(): return 'Individualism (Grok)'
    elif 'claude' in model_name.lower(): return 'Hierarchy (Claude)'
    else: return 'Egalitarianism (Others)'

df['Prototype'] = df['Model'].apply(map_prototype)

outrage_cols = ['Involuntariness', 'Delayed_effects', 'Severity', 'Dread',
                'Catastrophic_potential', 'Harm_to_future_generations',
                'Lack_of_knowledge_exposed', 'Lack_of_knowledge_scientists', 'Unfairness']

plt.style.use('ggplot')

# ==========================================
# 시각화 1: 상관계수 히트맵 (Correlation Heatmap)
# ==========================================
corr_data = []
for proto in df['Prototype'].unique():
    sub_df = df[df['Prototype'] == proto]
    # 위험성(Riskiness)과 9개 아웃레이지 요인 간의 피어슨 상관계수 추출
    corr = sub_df[outrage_cols + ['Riskiness']].corr(numeric_only=True)['Riskiness'].drop('Riskiness')
    corr.name = proto
    corr_data.append(corr)

corr_df = pd.DataFrame(corr_data).T

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
plt.title("Correlation between Outrage Factors and Riskiness by Cultural Prototype", fontsize=14, pad=20)
plt.ylabel("Outrage Factors")
plt.tight_layout()
plt.savefig('correlation_heatmap_prototype.png', dpi=300)
plt.close()

# ==========================================
# 시각화 2: 아웃레이지 프로파일 방사형 차트 (Radar Chart)
# ==========================================
mean_df = df.groupby('Prototype')[outrage_cols].mean()
categories = [col.replace('_', '\n') for col in outrage_cols]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, size=10)
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
plt.ylim(0, 5.5)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for idx, (proto, row) in enumerate(mean_df.iterrows()):
    values = row.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=proto, color=colors[idx])
    ax.fill(angles, values, color=colors[idx], alpha=0.1)

plt.title("Mean Outrage Profiles by Cultural Prototype", size=16, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('radar_chart_prototype.png', dpi=300)
plt.close()

# ==========================================
# 시각화 3: 극성 변수 대조 막대그래프 (Bar Chart)
# ==========================================
# 이론적으로 가장 차이가 두드러지는 3개 핵심 변수만 추출
focus_cols = ['Dread', 'Severity', 'Lack_of_knowledge_scientists']
focus_mean = mean_df[focus_cols].reset_index()
focus_melt = focus_mean.melt(id_vars='Prototype', var_name='Factor', value_name='Mean Score')

plt.figure(figsize=(10, 6))
sns.barplot(data=focus_melt, x='Factor', y='Mean Score', hue='Prototype', palette='Set2')
plt.title("Contrasting Key Outrage Factors Across Cultural Prototypes", fontsize=14)
plt.ylim(1, 5)
plt.tight_layout()
plt.savefig('key_factors_bar_prototype.png', dpi=300)
plt.close()