import pandas as pd
import numpy as np

def make_dirty_data(holding_file, sub_file, proj_file):
    # 1. 원본 데이터 로드
    holding_df = pd.read_csv(holding_file)
    sub_df = pd.read_csv(sub_file)
    proj_df = pd.read_csv(proj_file)

    np.random.seed(42) # 실행할 때마다 동일하게 오염되도록 시드 고정

    # ==========================================
    # 2. 지주회사 데이터 (holding_companies.csv) 오염
    # ==========================================
    # [노이즈 1] 날짜 포맷 파편화 (10% 확률로 'YYYY년 MM월', 5% 확률로 'YY/MM')
    idx_date1 = holding_df.sample(frac=0.1).index
    holding_df.loc[idx_date1, '설립인가연월'] = holding_df.loc[idx_date1, '설립인가연월'].astype(str).str.replace('-', '년 ') + '월'
    
    idx_date2 = holding_df.drop(idx_date1).sample(frac=0.05).index
    holding_df.loc[idx_date2, '설립인가연월'] = '08/11' # 임의의 훼손된 문자열 주입

    # [노이즈 2] 텍스트 공백 삽입 (10% 확률)
    idx_space = holding_df.sample(frac=0.1).index
    holding_df.loc[idx_space, '소재지'] = ' ' + holding_df.loc[idx_space, '소재지'] + '  '

    # [노이즈 3] 결측치(NaN) 및 이상치(음수) 주입
    idx_nan = holding_df.sample(frac=0.05).index
    holding_df.loc[idx_nan, '전담인력_수'] = np.nan
    idx_neg = holding_df.drop(idx_nan).sample(frac=0.05).index
    holding_df.loc[idx_neg, '전담인력_수'] = -5

    # [노이즈 4] 숫자형 컬럼의 문자열화 (전체 데이터에 '억', 콤마 추가)
    holding_df['자본금_총자본금'] = holding_df['자본금_총자본금'].apply(
        lambda x: f"{x:,.1f}억" if pd.notna(x) and isinstance(x, (int, float)) else x
    )

    # ==========================================
    # 3. 자회사 데이터 (subsidiaries.csv) 오염
    # ==========================================
    # [노이즈 1] 주요 재무 지표 결측치 주입 (매출액 10%, 당기순이익 10%)
    idx_rev_nan = sub_df.sample(frac=0.1).index
    idx_net_nan = sub_df.sample(frac=0.1).index
    sub_df.loc[idx_rev_nan, '매출액'] = np.nan
    sub_df.loc[idx_net_nan, '당기순이익'] = np.nan

    # [노이즈 2] 매출액 컬럼을 '문자열(Object)' 타입으로 훼손 ('원' 및 콤마 추가)
    sub_df['매출액'] = sub_df['매출액'].apply(
        lambda x: f"{int(x):,}백만원" if pd.notna(x) and str(x).replace('.','',1).isdigit() else x
    )

    # [노이즈 3] 통계를 망가뜨리는 극단적 이상치 주입 (1% 확률)
    idx_outlier = sub_df.sample(frac=0.01).index
    sub_df.loc[idx_outlier, '매출액'] = '99,999,999,999원'

    # [노이즈 4] 범주형 변수의 대소문자 혼용 및 띄어쓰기 훼손 (5~10% 확률)
    idx_cat1 = sub_df.sample(frac=0.05).index
    sub_df.loc[idx_cat1, '업종별현황'] = ' 제 조 업 '
    idx_cat2 = sub_df.sample(frac=0.05).index
    sub_df.loc[idx_cat2, '분야별현황'] = ' iT '

    # ==========================================
    # 4. 지원사업 데이터 (projects.csv) 오염
    # ==========================================
    # [노이즈 1] 종료일자 결측치(진행중인 사업이라 가정) 10% 주입
    idx_end_nan = proj_df.sample(frac=0.1).index
    proj_df.loc[idx_end_nan, '종료일자'] = np.nan

    # [노이즈 2] 지원금액 문자열화 (' 백만원', 콤마 추가)
    proj_df['지원금액'] = proj_df['지원금액'].apply(
        lambda x: f"{int(x):,} 백만원" if pd.notna(x) and str(x).replace('.','',1).isdigit() else x
    )

    # [노이즈 3] 논리적 오류 주입 (종료일자가 시작일자보다 과거인 경우, 2% 확률)
    idx_logic = proj_df.dropna(subset=['종료일자']).sample(frac=0.02).index
    proj_df.loc[idx_logic, '종료일자'] = '2010-01-01' # 의도적으로 아주 과거 날짜 세팅

    # ==========================================
    # 5. 오염된 데이터 저장
    # ==========================================
    holding_df.to_csv('dirty_holding_companies.csv', index=False, encoding='utf-8-sig')
    sub_df.to_csv('dirty_subsidiaries.csv', index=False, encoding='utf-8-sig')
    proj_df.to_csv('dirty_projects.csv', index=False, encoding='utf-8-sig')
    
    print("✅ 데이터 오염 완료! (전체 데이터 규모에 맞춰 무작위로 훼손되었습니다.)")

# 실제 파일명이 맞는지 확인 후 실행하세요.
make_dirty_data('D:/cursor/26-1/holding_companies.csv', 'D:/cursor/26-1/subsidiaries.csv', 'D:/cursor/26-1/projects.csv')
