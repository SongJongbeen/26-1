import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 난수 고정
np.random.seed(42)
random.seed(42)

# 총 1,000개의 사업(프로젝트)을 생성한다고 가정
TOTAL_PROJECTS = 1000
# 자회사는 이전 코드에서 생성한 S0001 ~ S1889 범위 사용
SUBSIDIARY_COUNT = 1889 

# ==========================================
# 1. 항목별 옵션 정의 (가중치 적용)
# ==========================================
sponsors = ['중앙정부', '지자체', '공공기관', '민간기업', '대학']
sponsor_weights = [0.5, 0.2, 0.15, 0.1, 0.05] # 중앙정부 사업 비중이 가장 높도록 설정

project_types = ['R&D지원', '기술사업화', '인력양성', '펀드운용', '위탁용역']
type_weights = [0.4, 0.3, 0.1, 0.05, 0.15]

# 사업유형별 그럴싸한 사업명 프리셋 (현실 고증)
project_names_map = {
    'R&D지원': ['창업성장기술개발사업', '디딤돌 기술개발', '산학연 Collabo R&D', '미래유망 R&D 사업'],
    '기술사업화': ['TIPS 프로그램', '초기창업패키지', '실험실 특화형 창업선도대학', '기술이전 사업화 지원'],
    '인력양성': ['청년 디지털 일자리 사업', '산학연계 인력양성', '첨단산업 인재양성 부트캠프'],
    '펀드운용': ['대학창업펀드 조성사업', '지역혁신 벤처펀드', '모태펀드 자펀드 운용'],
    '위탁용역': ['신기술 동향 분석 용역', '지자체 특화산업 연구용역', '공공기관 시스템 구축 용역']
}

# ==========================================
# 2. 데이터 생성 로직
# ==========================================
data = []
current_date = datetime(2024, 1, 1) # 현황(완료/진행중)을 판단할 기준일

for i in range(1, TOTAL_PROJECTS + 1):
    p_id = f"P{str(i).zfill(4)}"
    
    # 1889개의 자회사 중 무작위로 하나 선택 (하나의 자회사가 여러 사업을 수주할 수도 있음)
    s_id = f"S{str(random.randint(1, SUBSIDIARY_COUNT)).zfill(4)}"
    
    sponsor = random.choices(sponsors, weights=sponsor_weights)[0]
    p_type = random.choices(project_types, weights=type_weights)[0]
    p_name = random.choice(project_names_map[p_type])
    
    # 지원금액: 깔끔한 정수 (예: 10 ~ 1000 사이, 암묵적 단위는 백만원)
    # 결측치나 이상치 없이 깨끗한 데이터만 생성
    amount = random.randint(1, 100) * 10 
    
    # 시작일자: 2021년 ~ 2023년 사이의 무작위 날짜
    start_date = datetime(2021, 1, 1) + timedelta(days=random.randint(0, 1000))
    
    # 종료일자: 시작일로부터 6개월(180일) ~ 3년(1095일) 사이 무작위 기간 추가
    duration_days = random.randint(180, 1095)
    end_date = start_date + timedelta(days=duration_days)
    
    # 진행현황: 기준일(2024-01-01)을 넘겼으면 완료, 아니면 진행중 (결측치 없음)
    status = '완료' if end_date < current_date else '진행중'
    
    data.append([
        p_id, s_id, sponsor, p_type, p_name, amount, 
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), status
    ])

# ==========================================
# 3. 데이터프레임 생성 및 저장
# ==========================================
df_projects = pd.DataFrame(data, columns=[
    '사업ID', '자회사ID', '사업주체', '사업유형', '사업명', 
    '지원금액', '시작일자', '종료일자', '진행현황'
])

print("📊 [사업 데이터 (Clean Version) 생성 결과] 📊")

print(f"\n✅ 총 {len(df_projects)}개의 사업 데이터가 결측치/이상치 없이 생성되었습니다.")
print(f"✅ 진행현황 분포: \n{df_projects['진행현황'].value_counts()}")

# 저장
df_projects.to_csv('projects.csv', index=False, encoding='utf-8-sig')
print("\n🎉 'projects.csv' 파일이 성공적으로 저장되었습니다!")