import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. 결과 추출용 헬퍼 함수 정의
# ==========================================
def export_model_results(model_fit, model_name, filename, is_logit=False):
    """
    statsmodels의 회귀분석 결과를 논문 표 작성을 위한 데이터프레임으로 변환하여 CSV로 저장합니다.
    """
    results_df = pd.DataFrame({
        'Coefficient': model_fit.params,
        'Std_Error': model_fit.bse,
        'p_value': model_fit.pvalues
    })
    
    results_df['Significance'] = results_df['p_value'].apply(
        lambda x: '***' if x < 0.01 else '**' if x < 0.05 else '*' if x < 0.1 else ''
    )
    
    n_obs = model_fit.nobs
    llf = model_fit.llf if hasattr(model_fit, 'llf') else np.nan
    
    if is_logit:
        r2 = model_fit.prsquared if hasattr(model_fit, 'prsquared') else np.nan
        r2_label = 'Pseudo_R2'
    else:
        r2 = model_fit.rsquared if hasattr(model_fit, 'rsquared') else np.nan
        r2_label = 'R-squared'
        
    stats_df = pd.DataFrame({
        'Coefficient': [n_obs, r2, llf],
        'Std_Error': [np.nan, np.nan, np.nan],
        'p_value': [np.nan, np.nan, np.nan],
        'Significance': ['', '', '']
    }, index=['N_obs', r2_label, 'Log_Likelihood'])
    
    final_df = pd.concat([results_df, stats_df])
    final_df.to_csv(filename, encoding='utf-8-sig')
    print(f"[{model_name}] -> '{filename}' 저장 완료")

# ==========================================
# 1단계: 데이터 로드 및 전처리
# ==========================================
print("=== [1단계] 데이터 전처리 시작 ===")
df = pd.read_csv("PROCESSED_DATA.csv")

# 기존 시장형 수익 변수 생성
df['사업화_투자수익_발생여부'] = ((df['사업화수익보유여부'] == 1) | (df['투자수익보유여부'] == 1)).astype(int)

# [수정] 엄밀한 의미의 '이중형 여부' 더미 변수 생성 (정책사업 수익 AND 시장형 수익)
df['이중형_여부'] = ((df['국책사업비수익보유여부'] == 1) & (df['사업화_투자수익_발생여부'] == 1)).astype(int)

df = df.sort_values(by=['기술지주회사_익명', '연도'])

# 시차 변수 생성
df['전담인력_변환_t_plus_1'] = df.groupby('기술지주회사_익명')['전담인력_변환'].shift(-1)
df['자회사수_변환_t_plus_1'] = df.groupby('기술지주회사_익명')['자회사수_변환'].shift(-1)
df['국책사업비수익_t_plus_1'] = df.groupby('기술지주회사_익명')['국책사업비수익보유여부'].shift(-1)

# [수정] 이중형 전환 분석을 위한 시차 변수 생성
df['이중형_여부_t_plus_1'] = df.groupby('기술지주회사_익명')['이중형_여부'].shift(-1)
df['이중형_여부_t_minus_1'] = df.groupby('기술지주회사_익명')['이중형_여부'].shift(1)

df_clean = df.dropna().copy()

# ==========================================
# 2단계: 기술통계 및 상관관계 추출
# ==========================================
print("\n=== [2단계] 기술통계 및 상관관계 파일 생성 ===")
# [수정] 기술통계 변수에 '이중형_여부' 포함
vars_to_describe = ['업력년수', '자본금규모_변환', '전담인력_변환', '자회사수_변환', 
                    '국책사업비수익보유여부', '사업화_투자수익_발생여부', '이중형_여부']

desc_df = df_clean[vars_to_describe].describe().T[['count', 'mean', 'std', 'min', 'max']]
desc_df.to_csv("result_1_descriptive_stats.csv", encoding='utf-8-sig')
print("[기술통계량] -> 'result_1_descriptive_stats.csv' 저장 완료")

corr_matrix = pd.DataFrame(index=vars_to_describe, columns=vars_to_describe)
for col1 in vars_to_describe:
    for col2 in vars_to_describe:
        corr, pval = pearsonr(df_clean[col1], df_clean[col2])
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        corr_matrix.loc[col1, col2] = f"{corr:.3f}{stars}"
corr_matrix.to_csv("result_2_correlation_matrix.csv", encoding='utf-8-sig')
print("[상관관계] -> 'result_2_correlation_matrix.csv' 저장 완료")

# ==========================================
# 3단계: RQ2 이중형 운영모델 형성조건 (Logit)
# ==========================================
print("\n=== [3단계] RQ2 모형 분석 ===")
X_rq2 = df_clean[['업력년수', '설립유형_변환', '지역구분_변환', '자본금규모_변환', '전담인력_변환', '자회사규모_변환']]
X_rq2 = sm.add_constant(X_rq2)
# [수정] 종속변수를 이중형_여부로 변경
y_rq2 = df_clean['이중형_여부']

logit_rq2 = sm.Logit(y_rq2, X_rq2).fit(disp=0)
export_model_results(logit_rq2, "RQ2_이중형_Logit", "result_3_rq2_logit.csv", is_logit=True)

# ==========================================
# 4단계: RQ3 시차모형 (추가성 및 보완성)
# ==========================================
print("\n=== [4단계] RQ3 시차모형 분석 ===")
X_rq3 = sm.add_constant(df_clean[['국책사업비수익보유여부', '업력년수', '자본금규모_변환']])

y_input = df_clean['전담인력_변환_t_plus_1']
ols_input = sm.OLS(y_input, X_rq3).fit()
export_model_results(ols_input, "RQ3_투입추가성_OLS", "result_4_rq3_input_ols.csv", is_logit=False)

y_output = df_clean['자회사수_변환_t_plus_1']
ols_output = sm.OLS(y_output, X_rq3).fit()
export_model_results(ols_output, "RQ3_산출추가성_OLS", "result_5_rq3_output_ols.csv", is_logit=False)

# 3) 보완성 (Logit) - 직접적인 이중형 전환 효과
X_rq3_comp = X_rq3.copy()
# [수정] 상태의존성 통제를 기존 이중형 여부로 변경
X_rq3_comp['이중형_여부(t기)'] = df_clean['이중형_여부']
# [수정] 결과변수를 t+1기 이중형 전환 여부로 변경
y_comp = df_clean['이중형_여부_t_plus_1']
logit_comp = sm.Logit(y_comp, X_rq3_comp).fit(disp=0)
export_model_results(logit_comp, "RQ3_보완성(이중형전환)_Logit", "result_6_rq3_complement_logit.csv", is_logit=True)

# ==========================================
# 5단계: 역인과성 검증
# ==========================================
print("\n=== [5단계] 강건성 검증 (역인과성) ===")
# [수정] 독립변수로 사업화 수익 대신 이중형 여부 투입
X_rev = sm.add_constant(df_clean[['이중형_여부', '자회사수_변환', '업력년수']])
y_rev = df_clean['국책사업비수익_t_plus_1']

logit_rev = sm.Logit(y_rev, X_rev).fit(disp=0)
export_model_results(logit_rev, "역인과성검증_이중형_Logit", "result_7_robustness_reverse_logit.csv", is_logit=True)

print("\n모든 분석 결과가 개별 CSV 파일로 성공적으로 추출되었습니다.")