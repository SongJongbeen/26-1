import pandas as pd
import numpy as np

holding_df = pd.read_csv('D:/cursor/26-1/mot716/wk08_0422/holding_companies.csv')
sub_df = pd.read_csv('D:/cursor/26-1/mot716/wk08_0422/subsidiaries.csv')
proj_df = pd.read_csv('D:/cursor/26-1/mot716/wk08_0422/projects.csv')

# 2. 텍스트 불일치 및 범주형 데이터 정규화 (str.strip, str.replace, str.upper)
sub_df['분야별현황'] = sub_df['분야별현황'].str.strip().str.upper()
sub_df['업종별현황'] = sub_df['업종별현황'].str.replace(' ', '')
holding_df['소재지'] = holding_df['소재지'].str.strip()

# 3. 문자열 노이즈 클렌징 (str.replace)
sub_df['매출액'] = sub_df['매출액'].astype(str).str.replace('원', '').str.replace(',', '')
proj_df['지원금액'] = proj_df['지원금액'].astype(str).str.replace(' 백만원', '').str.replace(',', '')
holding_df['자본금_총자본금'] = holding_df['자본금_총자본금'].astype(str).str.replace('억', '').str.replace(',', '')

# 4. 데이터 타입 변환 (pd.to_numeric)
sub_df['매출액'] = pd.to_numeric(sub_df['매출액'], errors='coerce')
proj_df['지원금액'] = pd.to_numeric(proj_df['지원금액'], errors='coerce')
holding_df['자본금_총자본금'] = pd.to_numeric(holding_df['자본금_총자본금'], errors='coerce')

# 5. [핵심] 기준 단위 통일 (스칼라 연산) - '억'을 '백만원'으로 변환
# 1억 = 100백만원이므로 100을 곱하여 단위를 백만원으로 통일
holding_df['자본금_총자본금'] = (holding_df['자본금_총자본금'] * 100).round(0)

# 6. 비즈니스 로직을 반영한 결측치 보간 및 대체 (fillna)
# 매출액과 당기순이익이 없는 초기 기업은 '데스밸리' 구간으로 간주해 보수적으로 0으로 대체
sub_df['매출액'] = sub_df['매출액'].fillna(0)
sub_df['당기순이익'] = sub_df['당기순이익'].fillna(0)

# 7. 극단적 이상치(Outlier) 및 논리적 오류 처리 (clip, np.where)
# [지주회사] 인력 수가 음수(-5 등)인 비정상 데이터를 결측치로 변환 후 중앙값(median)으로 보간
holding_df['전담인력_수'] = np.where(holding_df['전담인력_수'] < 0, np.nan, holding_df['전담인력_수'])
holding_df['전담인력_수'] = holding_df['전담인력_수'].fillna(holding_df['전담인력_수'].median())

# [자회사] 매출액 0원이 대다수인 벤처 특성을 고려해 상위 99% 클리핑 방식으로 악성 극단치만 컷오프
upper_limit = sub_df['매출액'].quantile(0.99) 
sub_df['매출액'] = sub_df['매출액'].clip(upper=upper_limit)

# 8. 날짜 데이터 형식 통일 (pd.to_datetime 및 정규표현식)
# 다양한 문자열('2017년 2월', '18/05')을 정규표현식으로 정리한 뒤 datetime 변환
sub_df['설립연월_정제'] = sub_df['설립연월'].astype(str).str.replace(r'[년월\s/]+', '-', regex=True)
sub_df['설립연월'] = pd.to_datetime(sub_df['설립연월_정제'], errors='coerce')
proj_df['시작일자'] = pd.to_datetime(proj_df['시작일자'].astype(str).str.replace(r'[년월일\.\s]+', '-', regex=True), errors='coerce')
proj_df['종료일자'] = pd.to_datetime(proj_df['종료일자'], errors='coerce')
holding_df['설립인가연월_정제'] = holding_df['설립인가연월'].astype(str).str.replace(r'[년월일\s/\.]+', '-', regex=True)
holding_df['설립인가연월_정제'] = holding_df['설립인가연월_정제'].str.strip('-')
holding_df['설립인가연월'] = pd.to_datetime(holding_df['설립인가연월_정제'], errors='coerce')

# 9. 파생변수 생성 (기간 계산 및 논리적 오류 보정)
# 진행 중인 사업(종료일 결측)은 현재 날짜 기준으로 보정 후 기간 계산
current_date = pd.to_datetime('2023-12-31')
proj_df['종료일자_보정'] = proj_df['종료일자'].fillna(current_date)
proj_df['사업진행일수'] = (proj_df['종료일자_보정'] - proj_df['시작일자']).dt.days

# 시작일보다 종료일이 앞서는 논리적 오류(음수) 제거
proj_df['사업진행일수'] = np.where(proj_df['사업진행일수'] < 0, np.nan, proj_df['사업진행일수'])

# 10. 데이터 그룹핑, 집계 및 병합 (groupby, agg, merge)
# 자회사별로 참여한 프로젝트의 총 지원금액과 평균 진행일수를 집계
proj_summary = proj_df.groupby('자회사ID').agg(
    총참여사업수=('사업ID', 'count'),
    총지원금액=('지원금액', 'sum'),
    평균진행일수=('사업진행일수', 'mean')
).reset_index()

# 자회사 테이블을 기준으로 프로젝트 요약과 지주회사 정보를 모두 병합 (Left Join)
final_df = sub_df.merge(proj_summary, on='자회사ID', how='left')
final_df = final_df.merge(holding_df[['지주회사ID', '전담인력_수', '자본금_총자본금']], left_on='모회사ID', right_on='지주회사ID', how='left')
final_df['총참여사업수'] = final_df['총참여사업수'].fillna(0) # 사업 참여 이력이 없는 경우 0으로 보정

# 저장 / 각 df별로도
sub_df.to_csv('D:/cursor/26-1/mot716/wk08_0422/preprocess_data/subsidiaries.csv', index=False)
proj_df.to_csv('D:/cursor/26-1/mot716/wk08_0422/preprocess_data/projects.csv', index=False)
holding_df.to_csv('D:/cursor/26-1/mot716/wk08_0422/preprocess_data/holding_companies.csv', index=False)
final_df.to_csv('D:/cursor/26-1/mot716/wk08_0422/preprocess_data/final_data.csv', index=False)
