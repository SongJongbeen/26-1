import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1단계: 데이터 전처리 및 패널 구조화
# ==========================================
print("=== [1단계] 데이터 로드 및 전처리 ===")
df = pd.read_csv("PROCESSED_DATA.csv")

# 종속변수 생성: 사업화수익 또는 투자수익 중 하나라도 있으면 1
df['사업화_투자수익_발생여부'] = ((df['사업화수익보유여부'] == 1) | (df['투자수익보유여부'] == 1)).astype(int)

# 패널 정렬 (회사 식별자와 연도 기준)
df = df.sort_values(by=['기술지주회사_익명', '연도'])

# 시차 변수(Lag & Lead) 생성
# t+1 기의 결과 (Lead 변수)
df['전담인력_변환_t_plus_1'] = df.groupby('기술지주회사_익명')['전담인력_변환'].shift(-1)
df['자회사수_변환_t_plus_1'] = df.groupby('기술지주회사_익명')['자회사수_변환'].shift(-1)
df['사업화_투자수익_t_plus_1'] = df.groupby('기술지주회사_익명')['사업화_투자수익_발생여부'].shift(-1)
df['국책사업비수익_t_plus_1'] = df.groupby('기술지주회사_익명')['국책사업비수익보유여부'].shift(-1)

# 통제용 기초값 (t-1 기의 상태)
df['사업화_투자수익_t_minus_1'] = df.groupby('기술지주회사_익명')['사업화_투자수익_발생여부'].shift(1)

df_clean = df.dropna().copy() # 결측치 제거 (시차 분석을 위한 밸리드 데이터만 보존)


# ==========================================
# 2단계: 기술통계 및 상관관계 산출
# ==========================================
print("\n=== [2단계] 기술통계 및 상관관계 분석 ===")
vars_to_describe = ['업력년수', '자본금규모_변환', '전담인력_변환', '자회사수_변환', 
                    '국책사업비수익보유여부', '사업화_투자수익_발생여부']

print("\n[기술통계량]")
print(df_clean[vars_to_describe].describe().T[['count', 'mean', 'std', 'min', 'max']])

print("\n[상관관계 및 유의성(p-value)]")
corr_matrix = pd.DataFrame(index=vars_to_describe, columns=vars_to_describe)
for col1 in vars_to_describe:
    for col2 in vars_to_describe:
        corr, pval = pearsonr(df_clean[col1], df_clean[col2])
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        corr_matrix.loc[col1, col2] = f"{corr:.3f}{stars}"
print(corr_matrix)


# ==========================================
# 3단계: RQ2 이중형/시장형 수익 기반 형성조건
# ==========================================
print("\n=== [3단계] RQ2: 시장형 수익 형성조건 (Logit Model) ===")
# 상수항 추가
X_rq2 = df_clean[['업력년수', '설립유형_변환', '지역구분_변환', '자본금규모_변환', '전담인력_변환', '자회사규모_변환']]
X_rq2 = sm.add_constant(X_rq2)
y_rq2 = df_clean['사업화_투자수익_발생여부']

logit_rq2 = sm.Logit(y_rq2, X_rq2).fit(disp=0)
print(logit_rq2.summary())


# ==========================================
# 4단계: RQ3 시차모형 (추가성 및 보완성)
# ==========================================
print("\n=== [4단계] RQ3: 정책사업 수익의 후속 결과 (Time-lagged Models) ===")
X_rq3 = sm.add_constant(df_clean[['국책사업비수익보유여부', '업력년수', '자본금규모_변환']])

print("\n1) 투입 추가성: t기 국책사업수익 -> t+1기 전담인력 (OLS)")
y_input = df_clean['전담인력_변환_t_plus_1']
ols_input = sm.OLS(y_input, X_rq3).fit()
print(ols_input.summary().tables[1]) # 수치 요약표만 간결하게 출력

print("\n2) 산출 추가성: t기 국책사업수익 -> t+1기 자회사수 (OLS)")
y_output = df_clean['자회사수_변환_t_plus_1']
ols_output = sm.OLS(y_output, X_rq3).fit()
print(ols_output.summary().tables[1])

print("\n3) 보완성: t기 국책사업수익 -> t+1기 사업화·투자수익 발생 (Logit)")
# 통제변수로 기존 t기의 수익 상태 통제 (상태의존성)
X_rq3_comp = X_rq3.copy()
X_rq3_comp['사업화_투자수익_발생여부(t기)'] = df_clean['사업화_투자수익_발생여부']
y_comp = df_clean['사업화_투자수익_t_plus_1']
logit_comp = sm.Logit(y_comp, X_rq3_comp).fit(disp=0)
print(logit_comp.summary().tables[1])


# ==========================================
# 5단계: 역인과성 검증 (Robustness Check)
# ==========================================
print("\n=== [5단계] 역인과성 검증 (Reverse Causality Check) ===")
print("과거의 시장형 성과(t)가 미래의 정부 지원(t+1)을 예측하는가?")
X_rev = sm.add_constant(df_clean[['사업화_투자수익_발생여부', '자회사수_변환', '업력년수']])
y_rev = df_clean['국책사업비수익_t_plus_1']

logit_rev = sm.Logit(y_rev, X_rev).fit(disp=0)
print(logit_rev.summary().tables[1])
print("\n* 해석: 사업화_투자수익_발생여부의 계수가 유의하지 않다면, 역인과성 우려를 덜 수 있습니다.")