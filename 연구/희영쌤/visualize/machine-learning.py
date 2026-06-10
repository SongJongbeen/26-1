import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram

# 한글 폰트 및 경고 무시 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # 맥 유저라면 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 데이터 로드 및 전처리 (문자열 -> 숫자 변환)
# ==========================================
# 파일명은 상황에 맞게 변경하세요.
df = pd.read_csv('concat_data.csv')

# --- 파생 변수 1: 수익 '여부' (0 또는 1) 만들기 ---
# 국책사업비 수익 여부
df['정책수익여부'] = df['국책사업비(정부사업비)'].apply(lambda x: 0 if pd.isna(x) or x == '0원' else 1)

# 사업화수익 여부 (세 가지 중 하나라도 0원이 아니면 1)
def check_biz_revenue(row):
    div1 = row['자회사배당수익']
    div2 = row['투자조합배당수익']
    exit_rev = row['자회사지분매각(일부+전부)']
    
    if (div1 != '0원' and pd.notna(div1)) or \
       (div2 != '0원' and pd.notna(div2)) or \
       (exit_rev != '0원' and pd.notna(exit_rev)):
        return 1
    return 0

df['사업화수익여부'] = df.apply(check_biz_revenue, axis=1)

# 이중형 여부 (정책수익과 사업화수익 둘 다 있는 경우)
df['이중형여부'] = ((df['정책수익여부'] == 1) & (df['사업화수익여부'] == 1)).astype(int)

# --- 파생 변수 2: 문자열에서 '최소 숫자' 추출 (PCA 및 ML용) ---
def extract_numeric(text):
    """ '10억이상~20억미만' -> 10, '3~5명' -> 3 으로 추출하는 함수 """
    if pd.isna(text) or text == '0원' or text == '0':
        return 0.0
    text = str(text).replace(',', '')
    # 숫자 패턴 찾기
    numbers = re.findall(r'\d+', text)
    if numbers:
        return float(numbers[0]) # 구간 중 첫 번째(최소값) 숫자를 해당 지표로 사용
    return 0.0

df['자본금_숫자'] = df['자본금 규모 구분'].apply(extract_numeric)
df['전담인력_숫자'] = df['전담인력'].apply(extract_numeric)


# ==========================================
# 2. 데이터 시각화 (Lineplot)
# ==========================================
plt.figure(figsize=(10, 6))
# 0과 1로 된 컬럼을 넣으면 sns.lineplot이 연도별 '평균(비중)'을 자동으로 계산합니다.
sns.lineplot(data=df, x='연도', y='정책수익여부', label='정책수익 비중', marker='o', errorbar=None)
sns.lineplot(data=df, x='연도', y='사업화수익여부', label='사업화·투자수익 비중', marker='s', errorbar=None)
sns.lineplot(data=df, x='연도', y='이중형여부', label='이중형 조직 비중', marker='^', errorbar=None)

plt.title('연도별 기술지주회사 수익 구조 및 운영모델 비중 변화 (2020-2024)')
plt.xlabel('연도')
plt.ylabel('비중 (Proportion)')
plt.ylim(0, 0.8) # 필요에 따라 1로 조정
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ==========================================
# 3. 주성분 분석 (PCA) - 역량 지수 도출
# ==========================================
# 머신러닝에는 숫자형 컬럼만 들어가야 합니다.
pca_features = ['자본금_숫자', '전담인력_숫자', '업력년수']
df_pca = df.dropna(subset=pca_features).copy()

# 스케일링 (단위 통일)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_pca[pca_features])

# PCA 수행
pca = PCA(n_components=1)
df_pca['역량지수(Capability_Index)'] = pca.fit_transform(scaled_data)
print(f"\n[PCA] 주성분 1이 설명하는 데이터 분산 비율: {pca.explained_variance_ratio_[0]:.2%}\n")


# ==========================================
# 4. 계층적 군집화 (Dendrogram)
# ==========================================
# 가장 최근 연도(2024년) 데이터만 필터링
df_2024 = df[df['연도'] == 2024].copy()

cluster_features = ['정책수익여부', '사업화수익여부', '자본금_숫자', '전담인력_숫자']
df_cluster = df_2024.dropna(subset=cluster_features)

scaled_cluster_data = scaler.fit_transform(df_cluster[cluster_features])
linked = linkage(scaled_cluster_data, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked, labels=df_cluster['기술지주회사(코드)'].values, distance_sort='descending')
plt.title('2024년 기준 기술지주회사 계층적 군집화 (Dendrogram)')
plt.xlabel('기술지주회사(코드)')
plt.ylabel('군집 간 거리 (Ward)')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()


# ==========================================
# 5. 머신러닝 (Random Forest) - 피처 중요도
# ==========================================
# t+1 시점의 이중형 전환 여부를 예측하기 위해 데이터 정렬 후 타겟(Label) 생성
df = df.sort_values(by=['기술지주회사(코드)', '연도'])
df['target_이중형전환_t1'] = df.groupby('기술지주회사(코드)')['이중형여부'].shift(-1)

# 범주형 변수(설립유형 등)를 컴퓨터가 이해할 수 있게 더미(0, 1) 변수로 변환
df_encoded = pd.get_dummies(df, columns=['설립유형'], drop_first=True)

# 머신러닝에 사용할 입력 피처(숫자형)
rf_features = ['업력년수', '자본금_숫자', '전담인력_숫자', '정책수익여부']
# 설립유형 더미가 생성되었으면 피처 리스트에 추가
type_cols = [c for c in df_encoded.columns if '설립유형_' in c]
rf_features.extend(type_cols)

# 결측치(가장 마지막 연도는 t+1 데이터가 없으므로 NaN) 제거
df_rf = df_encoded.dropna(subset=rf_features + ['target_이중형전환_t1'])

X = df_rf[rf_features]
y = df_rf['target_이중형전환_t1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)
print(f"Random Forest Accuracy: {rf_model.score(X_test, y_test):.2f}")

importance_df = pd.DataFrame({
    'Feature': rf_features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('다음 해(t+1) 이중형 모델 전환을 결정하는 핵심 요인 (Feature Importance)')
plt.xlabel('중요도 (Gini Importance)')
plt.ylabel('입력 변수')
plt.show()