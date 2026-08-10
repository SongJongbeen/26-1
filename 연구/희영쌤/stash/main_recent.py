import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 1. 데이터 로드 (인코딩 에러 방지)
try:
    df = pd.read_csv("D:\\cursor\\26-1\\연구\\희영쌤\\PROCESSED_DATA.csv", encoding='utf-8')
except:
    df = pd.read_csv("D:\\cursor\\26-1\\연구\\희영쌤\\PROCESSED_DATA.csv", encoding='cp949')

# 2. 데이터 전처리 및 변수 생성
# 사업화수익 또는 투자수익 중 하나라도 있으면 '사업화_투자수익' 보유로 간주
df['사업화_투자수익'] = df[['사업화수익보유여부', '투자수익보유여부']].max(axis=1)

# 수익구조 유형 정의 함수
def categorize_type(row):
    if row['국책사업비수익보유여부'] == 1 and row['사업화_투자수익'] == 1:
        return '이중형'
    elif row['국책사업비수익보유여부'] == 1 and row['사업화_투자수익'] == 0:
        return '정책형'
    elif row['국책사업비수익보유여부'] == 0 and row['사업화_투자수익'] == 1:
        return '사업형'
    else:
        return '무수익형'

# 문자열 유형 및 숫자형(다항 로지스틱용) 유형 생성
df['수익구조유형'] = df.apply(categorize_type, axis=1)
df['수익구조유형_num'] = df['수익구조유형'].map({'정책형': 0, '사업형': 1, '이중형': 2, '무수익형': 3})
df['이중형여부'] = (df['수익구조유형'] == '이중형').astype(int)

# 사용자 정의 기준 매핑
df['설립유형_명칭'] = df['설립유형_변환'].map({1: '단독형', 2: '공동형'})
df['수도권여부'] = df['지역구분_변환'].apply(lambda x: 1 if x == 1 else 0)
df['조직단계'] = df['업력년수'].apply(lambda x: '초기' if x <= 7 else '성숙')


print("="*50)
print("RQ1. 2020~2024년 수익구조 변화 분석")
print("="*50)

# 1-1. 연도별 비중 추이 및 로지스틱 회귀 검증
trend_data = df.groupby('연도').agg(
    국책수익보유비중=('국책사업비수익보유여부', 'mean'),
    이중형비중=('이중형여부', 'mean')
).reset_index()
print("[연도별 비중 추이]\n", trend_data)

logit_rq1_1 = smf.logit("국책사업비수익보유여부 ~ 연도", data=df).fit(disp=0)
logit_rq1_2 = smf.logit("이중형여부 ~ 연도", data=df).fit(disp=0)
print(f"\n국책사업비 수익 연도별 변화 P-value: {logit_rq1_1.pvalues['연도']:.4f}")
print(f"이중형 비중 연도별 변화 P-value: {logit_rq1_2.pvalues['연도']:.4f}")

# 1-2. 유형 간 전환 확률 행렬 (Markov Transition)
df_sorted = df.sort_values(by=['기술지주회사_익명', '연도'])
df_sorted['다음해_수익구조유형'] = df_sorted.groupby('기술지주회사_익명')['수익구조유형'].shift(-1)
trans_matrix = pd.crosstab(df_sorted['수익구조유형'], df_sorted['다음해_수익구조유형'], normalize='index') * 100
print("\n[유형 간 전환 행렬 (%)]\n", trans_matrix.round(1))


print("\n" + "="*50)
print("RQ2. 업력, 설립유형, 지역에 따른 수익구조 분화")
print("="*50)

# 2-1. 업력에 따른 분화 (Multinomial Logit)
# 분석 대상: 정책형(0), 사업형(1), 이중형(2)
df_rq2 = df[df['수익구조유형_num'].isin([0, 1, 2])].copy()
mnlogit_rq2 = smf.mnlogit("수익구조유형_num ~ 업력년수", data=df_rq2).fit(disp=0)
print("[업력에 따른 수익유형 분화 MNLogit 결과 (Ref=0:정책형, 1=사업형, 2=이중형)]\n")
print(mnlogit_rq2.summary().tables[1])

# 2-2. 설립유형(단독 vs 공동)에 따른 이중형 수익구조 차이
ct = pd.crosstab(df['설립유형_명칭'], df['이중형여부'])
chi2, p, _, _ = chi2_contingency(ct)
print(f"\n[설립유형-이중형 교차분석 P-value]: {p:.4f}")
print("[설립유형별 이중형 비율]\n", pd.crosstab(df['설립유형_명칭'], df['이중형여부'], normalize='index').round(3))

# 2-3 & 2-4. 지역(수도권) 차이 및 자원(자본, 인력, 자회사) 통제 효과
logit_rq2_3 = smf.logit("국책사업비수익보유여부 ~ 수도권여부", data=df).fit(disp=0)
print("\n[Model 1: 수도권여부 -> 국책사업비수익 (자원 통제 전)]\n")
print(logit_rq2_3.summary().tables[1])

logit_rq2_4 = smf.logit("국책사업비수익보유여부 ~ 수도권여부 + 자본금규모_변환 + 전담인력_변환 + 자회사규모_변환", data=df).fit(disp=0)
print("\n[Model 2: 수도권여부 -> 국책사업비수익 (자원 통제 후)]\n")
print(logit_rq2_4.summary().tables[1])


print("\n" + "="*50)
print("RQ3. 정책사업 수익의 후속 효과 (시차 분석)")
print("="*50)

# 시차 변수(Lag) 생성 - 패널 데이터 형태
df_sorted['다음해_전담인력'] = df_sorted.groupby('기술지주회사_익명')['전담인력_변환'].shift(-1)
# 자회사수가 누적이므로 신규 자회사수는 (t+1년 자회사수 - t년 자회사수)
df_sorted['다음해_신규자회사수'] = df_sorted.groupby('기술지주회사_익명')['자회사수_변환'].shift(-1) - df_sorted['자회사수_변환']
df_sorted['다음해_사업화투자수익'] = df_sorted.groupby('기술지주회사_익명')['사업화_투자수익'].shift(-1)

# 결측치 제거
df_rq3 = df_sorted.dropna(subset=['다음해_전담인력', '다음해_신규자회사수', '다음해_사업화투자수익']).copy()

# 3-1. 국책사업비수익(t) -> 전담인력 증가(t+1) (당해 연도 전담인력 통제)
ols_rq3_1 = smf.ols("다음해_전담인력 ~ 국책사업비수익보유여부 + 전담인력_변환 + 연도", data=df_rq3).fit()
print("[국책수익(t) -> 다음해 전담인력(t+1) OLS 결과]\n")
print(ols_rq3_1.summary().tables[1])

# 3-2. 국책사업비수익(t) -> 신규 자회사 증가(t+1) (당해 연도 누적 자회사수 통제)
ols_rq3_2 = smf.ols("다음해_신규자회사수 ~ 국책사업비수익보유여부 + 자회사수_변환 + 연도", data=df_rq3).fit()
print("\n[국책수익(t) -> 다음해 신규 자회사수(t+1) OLS 결과]\n")
print(ols_rq3_2.summary().tables[1])

# 3-3 & 3-4. 국책사업비수익(t) -> 사업화투자수익(t+1) 및 초기/성숙 단계 조절효과
# C(조직단계, Treatment(reference='초기'))를 사용하여 '초기' 조직을 기준점으로 상호작용 분석
logit_rq3_3 = smf.logit("다음해_사업화투자수익 ~ 국책사업비수익보유여부 * C(조직단계, Treatment(reference='초기')) + 연도 + 사업화_투자수익", data=df_rq3).fit(disp=0)
print("\n[국책수익(t) 및 조직단계 상호작용 -> 다음해 사업화/투자수익(t+1) Logit 결과]\n")
print(logit_rq3_3.summary().tables[1])
