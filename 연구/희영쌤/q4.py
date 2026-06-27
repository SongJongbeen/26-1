import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드 및 전처리
df = pd.read_csv('PROCESSED_DATA.csv')

# 분석에 사용할 독립변수(X)와 종속변수(y) 설정
# 종속변수: 자립적 수익 창출 능력 지표인 '사업화수익보유여부'
# 독립변수: 강의 자료 및 가설 기반의 조직 자원 및 환경 특성
features = [
    '업력년수', '자본금규모_변환', '전담인력_변환', 
    '자회사수_변환', '자회사규모_변환', '설립유형_변환', 
    '지역구분_변환', '국책사업비수익보유여부'
]

# 결측치 제거 후 변수 할당
model_df = df[features + ['사업화수익보유여부']].dropna()
X = model_df[features]
y = model_df['사업화수익보유여부']

# 2. 학습/테스트 데이터 분할 (8:2 비율 적용, 클래스 불균형 방지를 위한 stratify 설정)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 모델 구축 및 하이퍼파라미터 튜닝 (Random Forest 모델 활용)
rf_model = RandomForestClassifier(random_state=42)

# 탐색할 하이퍼파라미터 그리드 설정 (wk11.ipynb 방식 준수)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# 5-Fold 교차 검증을 통한 최적 모델 탐색
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 4. 최적 모델 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 평가지표 산출 (정확도 및 분류 성능 요약)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 5. 변수 중요도(Feature Importance) 추출
# 의사결정에 활용하기 위한 핵심 지표 (어떤 요인이 사업화 수익 창출을 결정짓는지 파악)
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 6. 분석 결과 출력
print("1. 최적 하이퍼파라미터 조합:\n", grid_search.best_params_, "\n")
print(f"2. 테스트 데이터 예측 정확도: {accuracy:.4f}\n")
print("3. 상세 분류 성능(Classification Report):\n", report, "\n")
print("4. 변수 중요도(Feature Importance):\n", feature_importances)