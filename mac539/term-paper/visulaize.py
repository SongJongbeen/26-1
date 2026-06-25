import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# 1. 데이터 불러오기 및 폰트 설정
df = pd.read_csv('sample.csv')

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 깨짐 방지용 폰트 설정 가능

# ==========================================
# H1a: 일반적 신뢰에서의 주류 미디어 평균화 휴리스틱 모방
# 검정 방법: "News media in general"의 점수와 Mainstream 평균 점수 간의 차이, HAC 평균 간의 차이 비교
# ==========================================
plt.figure(figsize=(8, 6))

# News media in general의 평균 신뢰도 (기준선)
general_trust_baseline = df[df['Trustee'] == 'News media in general']['General_Trust'].mean()

# Mainstream과 HAC 데이터 분리
df_h1a = df[df['Category'].isin(['Mainstream', 'HAC'])]

sns.barplot(data=df_h1a, x='Category', y='General_Trust', capsize=0.1, errorbar='ci', palette='muted')
plt.axhline(y=general_trust_baseline, color='red', linestyle='--', label=f'"News media in general" (Mean: {general_trust_baseline:.2f})')

# 통계 검정 (Independent t-test)
mainstream_scores = df[df['Category'] == 'Mainstream']['General_Trust']
hac_scores = df[df['Category'] == 'HAC']['General_Trust']
t_stat, p_val = stats.ttest_ind(mainstream_scores, hac_scores)

plt.title(f'H1a: General Trust Heuristic\n(Mainstream vs HAC p-value: {p_val:.3e})')
plt.legend()
plt.tight_layout()
plt.savefig('H1a_Result.png')
plt.show()

# ==========================================
# H1b: 정보 기반의 다차원적 신뢰성 차등 평가
# 검정 방법: (정확성, 완전성) 그룹 vs (공정성, 비편향성) 그룹 간의 T-test
# ==========================================
plt.figure(figsize=(8, 6))

info_based = df[['Accuracy', 'Completeness']].mean(axis=1)
norm_based = df[['Fairness', 'Unbiasedness']].mean(axis=1)

# 데이터프레임 재구성
df_h1b = pd.DataFrame({
    'Score': pd.concat([info_based, norm_based]),
    'Dimension_Group': ['Accuracy & Completeness']*len(info_based) + ['Fairness & Unbiasedness']*len(norm_based)
})

sns.boxplot(data=df_h1b, x='Dimension_Group', y='Score', palette='Set2')

# 통계 검정 (Paired t-test)
t_stat_1b, p_val_1b = stats.ttest_rel(info_based, norm_based)

# 유의성 표시 (별표)
sig_symbol = "***" if p_val_1b < 0.001 else ("**" if p_val_1b < 0.01 else ("*" if p_val_1b < 0.05 else "ns"))
plt.plot([0, 0, 1, 1], [df_h1b['Score'].max()+0.2, df_h1b['Score'].max()+0.5, df_h1b['Score'].max()+0.5, df_h1b['Score'].max()+0.2], lw=1.5, c='black')
plt.text(0.5, df_h1b['Score'].max()+0.6, f'T-test: p={p_val_1b:.3e} ({sig_symbol})', ha='center', color='black')

plt.title('H1b: Multidimensional Trust Evaluation')
plt.ylim(0, df_h1b['Score'].max() + 1.5)
plt.tight_layout()
plt.savefig('H1b_Result.png')
plt.show()

# ==========================================
# H2a: 결정론적 자기이익 동기 (냉소)
# 검정 방법: AI의 Cynicism_Score가 중립 척도(예: 4.0, 7점 척도 기준)보다 유의미하게 높은지 One-sample t-test
# ==========================================
plt.figure(figsize=(8, 6))

neutral_midpoint = 4.0 # 척도의 중립값 (설문 구성에 따라 수정 필요)
sns.histplot(df['Cynicism_Score'], kde=True, color='purple', bins=10)
plt.axvline(neutral_midpoint, color='black', linestyle='--', label='Neutral Midpoint (4.0)')
mean_cynicism = df['Cynicism_Score'].mean()
plt.axvline(mean_cynicism, color='red', linestyle='-', label=f'AI Mean ({mean_cynicism:.2f})')

# 통계 검정 (One-sample t-test)
t_stat_2a, p_val_2a = stats.ttest_1samp(df['Cynicism_Score'].dropna(), neutral_midpoint)
# 단측 검정으로 해석 (크다)
p_val_2a_one_sided = p_val_2a / 2 if t_stat_2a > 0 else 1.0

plt.title(f'H2a: Deterministic Self-serving Motives\n(One-sample t-test p-value: {p_val_2a_one_sided:.3e})')
plt.xlabel('Cynicism Score')
plt.legend()
plt.tight_layout()
plt.savefig('H2a_Result.png')
plt.show()

# ==========================================
# H2b: 미디어 행위자에 대한 무차별적 인식 (낮은 분화도)
# 검정 방법: 모델별 냉소 점수 vs 매체간 신뢰도 표준편차(분화도)의 상관분석
# ==========================================
plt.figure(figsize=(8, 6))

# 모델별로 냉소 점수 평균과 주류 미디어 신뢰도의 표준편차(분화도) 계산
df_mainstream = df[df['Category'] == 'Mainstream']
differentiation_df = df_mainstream.groupby('Model').agg(
    Mean_Cynicism=('Cynicism_Score', 'mean'),
    Trust_Std=('General_Trust', 'std') # 개별 매체간 차이를 얼마나 두는지(분화도)
).dropna()

sns.regplot(data=differentiation_df, x='Mean_Cynicism', y='Trust_Std', color='crimson')

# 통계 검정 (Pearson Correlation)
if len(differentiation_df) > 1:
    corr, p_val_2b = stats.pearsonr(differentiation_df['Mean_Cynicism'], differentiation_df['Trust_Std'])
    plt.text(differentiation_df['Mean_Cynicism'].min(), differentiation_df['Trust_Std'].max(), 
             f'Pearson r: {corr:.3f}\np-value: {p_val_2b:.3f}', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
else:
    plt.text(0.5, 0.5, 'Not enough model variance for correlation', ha='center', transform=plt.gca().transAxes)

plt.title('H2b: Cynicism vs Low Differentiation (Trust Std. Dev.)')
plt.xlabel('Cynicism Score (Mean per Model)')
plt.ylabel('Differentiation (Standard Deviation of Trust)')
plt.tight_layout()
plt.savefig('H2b_Result.png')
plt.show()

# ==========================================
# H3a: 반기득권 정서에 기반한 HAC 미디어의 단일 범주화
# 검정 방법: HAC 내부 매체들 간의 One-way ANOVA (차이가 없어야 함) + Mainstream 대비 T-test (유의미하게 낮아야 함)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# 데이터 준비
df_hac = df[df['Category'] == 'HAC']
hac_trustees = df_hac['Trustee'].unique()

sns.boxplot(data=df_hac, x='Trustee', y='General_Trust', palette='pastel', ax=ax)

# 통계 검정 1: HAC 매체 간 ANOVA (유의미한 차이가 없음을 증명)
hac_groups = [df_hac[df_hac['Trustee'] == t]['General_Trust'].dropna() for t in hac_trustees]
f_stat, p_val_anova = stats.f_oneway(*hac_groups)

# 통계 검정 2: Mainstream vs HAC 비교를 위한 Mainstream 평균선 추가
mean_mainstream = df[df['Category'] == 'Mainstream']['General_Trust'].mean()
ax.axhline(mean_mainstream, color='green', linestyle='--', label=f'Mainstream Mean ({mean_mainstream:.2f})')

ax.set_title(f'H3a: Single Categorization of HAC Media\n(ANOVA among HAC p-value: {p_val_anova:.3f} -> If >0.05, no significant difference)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig('H3a_Result.png')
plt.show()