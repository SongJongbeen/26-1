import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency

# 데이터 로드 및 수익구조 파생변수 생성
df = pd.read_csv('PROCESSED_DATA.csv')

df['시장수익보유여부'] = ((df['사업화수익보유여부'] == 1) | (df['투자수익보유여부'] == 1)).astype(int)
df['이중형'] = ((df['국책사업비수익보유여부'] == 1) & (df['시장수익보유여부'] == 1)).astype(int)

# 조직유형 정의 및 부여
def categorize_org(row):
    if row['국책사업비수익보유여부'] == 1 and row['시장수익보유여부'] == 0:
        return '정책형'
    elif row['국책사업비수익보유여부'] == 0 and row['시장수익보유여부'] == 1:
        return '사업형'
    elif row['국책사업비수익보유여부'] == 1 and row['시장수익보유여부'] == 1:
        return '이중형'
    else:
        return '무수익형'

df['조직유형'] = df.apply(categorize_org, axis=1)

# 1. 업력년수에 따른 조직유형 변화 분석
# 직관적 추이 확인을 위한 업력 구간화 (초기, 성장기, 성숙기, 장기)
df['업력구간'] = pd.cut(
    df['업력년수'], 
    bins=[0, 3, 7, 15, 100], 
    labels=['초기(1~3년)', '성장기(4~7년)', '성숙기(8~15년)', '장기(16년 이상)']
)
tenure_group_crosstab = pd.crosstab(df['업력구간'], df['조직유형'], normalize='index') * 100

# 2. 설립유형(공동형 vs 단독형)에 따른 이중형 수익구조 차이 분석
# 설립유형_변환 값 활용, 카이제곱 검정을 통한 통계적 유의성 확인
type_dual_crosstab = pd.crosstab(df['설립유형_변환'], df['이중형'], normalize='index') * 100
chi2_val, p_val, dof, expected = chi2_contingency(pd.crosstab(df['설립유형_변환'], df['이중형']))

# 3. 지역별(수도권 vs 비수도권) 정책의존도 및 사업화수익 발생 차이 분석
# 지역구분_변환 값을 기반으로 수도권 더미 변수 생성 (코드값 1을 수도권으로 전제)
df['수도권여부'] = (df['지역구분_변환'] == 1).astype(int)

region_policy_ratio = df.groupby('수도권여부')['국책사업비수익보유여부'].mean() * 100
region_market_ratio = df.groupby('수도권여부')['사업화수익보유여부'].mean() * 100

# 4. 지역 차이와 자원 요인(자본금, 전담인력, 자회사규모) 간의 관계 분석
# 지역별 평균 자원 규모 비교
resource_vars = ['자본금규모_변환', '전담인력_변환', '자회사규모_변환']
region_resource_mean = df.groupby('수도권여부')[resource_vars].mean()

# 로지스틱 회귀분석 (자원 요인 통제 시 지역 차이의 유의성 소멸 여부 확인)
reg_data = df[['사업화수익보유여부', '수도권여부'] + resource_vars].dropna()
X = sm.add_constant(reg_data[['수도권여부'] + resource_vars])
y = reg_data['사업화수익보유여부']
logit_model = sm.Logit(y, X).fit(disp=0)

# 분석 결과 출력
print("1. 업력년수 구간별 조직유형 비중(%)\n", tenure_group_crosstab, "\n")
print(f"2. 설립유형별 이중형 비중(%) [p-value: {p_val:.4f}]\n", type_dual_crosstab, "\n")
print("3. 지역별 수익 발생 비율(%)\n", pd.DataFrame({'국책의존도': region_policy_ratio, '사업화수익': region_market_ratio}), "\n")
print("4. 지역별 조직 자원 평균\n", region_resource_mean, "\n")
print("5. 사업화수익 결정요인 로지스틱 회귀분석 결과\n", logit_model.summary())