import pandas as pd

# 데이터 로드
df = pd.read_csv('PROCESSED_DATA.csv')

# 1. 연도별 국책사업비 수익 보유 기업 비중 산출
policy_revenue_trend = df.groupby('연도')['국책사업비수익보유여부'].mean() * 100

# 시장수익(사업화수익 또는 투자수익) 보유 여부 파악을 위한 파생변수 생성
df['시장수익보유여부'] = ((df['사업화수익보유여부'] == 1) | (df['투자수익보유여부'] == 1)).astype(int)

# 2. 이중형 조직(국책사업비 및 시장수익 모두 보유) 변수 생성 및 연도별 비중 산출
df['이중형'] = ((df['국책사업비수익보유여부'] == 1) & (df['시장수익보유여부'] == 1)).astype(int)
dual_type_trend = df.groupby('연도')['이중형'].mean() * 100

# 3. 조직 유형 분류 조건 설정 (정책형, 사업형, 이중형, 무수익형)
def categorize_org(row):
    if row['국책사업비수익보유여부'] == 1 and row['시장수익보유여부'] == 0:
        return '정책형'
    elif row['국책사업비수익보유여부'] == 0 and row['시장수익보유여부'] == 1:
        return '사업형'
    elif row['국책사업비수익보유여부'] == 1 and row['시장수익보유여부'] == 1:
        return '이중형'
    else:
        return '무수익형'

# 전체 데이터에 조직 유형 부여
df['조직유형'] = df.apply(categorize_org, axis=1)

# 기업별 시계열 변화 추적을 위한 데이터 정렬 (기업명, 연도 기준)
df = df.sort_values(by=['기술지주회사_익명', '연도'])

# 기준 연도 대비 직전 연도 조직 유형 파악
df['이전_조직유형'] = df.groupby('기술지주회사_익명')['조직유형'].shift(1)

# 전환 분석을 위한 결측치(각 기업의 최초 관측 연도) 제거
transition_df = df.dropna(subset=['이전_조직유형'])

# 전년도 대비 당해연도 조직 유형 전환 행렬(교차표, 비율%) 도출
transition_matrix = pd.crosstab(
    transition_df['이전_조직유형'], 
    transition_df['조직유형'], 
    normalize='index'
) * 100

# 결과 확인용 출력
print("1. 연도별 국책사업비 수익 보유 비중(%)\n", policy_revenue_trend, "\n")
print("2. 연도별 이중형 조직 비중(%)\n", dual_type_trend, "\n")
print("3. 조직 유형 간 전환 행렬(%)\n", transition_matrix)
