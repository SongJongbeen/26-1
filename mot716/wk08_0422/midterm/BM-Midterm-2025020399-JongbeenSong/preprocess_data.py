import pandas as pd
import numpy as np

holding_df = pd.read_csv('C:/Users/10414/cursor/26-1/mot716/wk08_0422/holding_companies.csv')
sub_df = pd.read_csv('C:/Users/10414/cursor/26-1/mot716/wk08_0422/subsidiaries.csv')
proj_df = pd.read_csv('C:/Users/10414/cursor/26-1/mot716/wk08_0422/projects.csv')

sub_df['분야별현황'] = sub_df['분야별현황'].str.strip().str.upper()
sub_df['업종별현황'] = sub_df['업종별현황'].str.replace(' ', '')
holding_df['소재지'] = holding_df['소재지'].str.strip()

sub_df['매출액'] = sub_df['매출액'].astype(str).str.replace('원', '').str.replace(',', '').str.replace('백만','')
proj_df['지원금액'] = proj_df['지원금액'].astype(str).str.replace(' 백만원', '').str.replace(',', '')
holding_df['자본금_총자본금'] = holding_df['자본금_총자본금'].astype(str).str.replace('억', '').str.replace(',', '')

sub_df['매출액'] = pd.to_numeric(sub_df['매출액'], errors='coerce')
proj_df['지원금액'] = pd.to_numeric(proj_df['지원금액'], errors='coerce')
holding_df['자본금_총자본금'] = pd.to_numeric(holding_df['자본금_총자본금'], errors='coerce')

holding_df['자본금_총자본금'] = (holding_df['자본금_총자본금'] * 100).round(0)

sub_df['매출액'] = sub_df['매출액'].fillna(0)
sub_df['당기순이익'] = sub_df['당기순이익'].fillna(0)

holding_df['전담인력_수'] = np.where(holding_df['전담인력_수'] < 0, np.nan, holding_df['전담인력_수'])
holding_df['전담인력_수'] = holding_df['전담인력_수'].fillna(holding_df['전담인력_수'].median())

non_zero_revenue = sub_df[sub_df['매출액'] > 0]['매출액']

Q1 = non_zero_revenue.quantile(0.25)
Q3 = non_zero_revenue.quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

sub_df['매출액'] = sub_df['매출액'].clip(upper=upper_bound)
sub_df['매출액'] = sub_df['매출액'].round(0)

sub_df['설립연월'] = sub_df['설립연월'].astype(str).str.replace(r'[년월\s/]+', '-', regex=True)
sub_df['설립연월'] = pd.to_datetime(sub_df['설립연월'], errors='coerce')
proj_df['시작일자'] = pd.to_datetime(proj_df['시작일자'].astype(str).str.replace(r'[년월일\.\s]+', '-', regex=True), errors='coerce')
proj_df['종료일자'] = pd.to_datetime(proj_df['종료일자'], errors='coerce')
holding_df['설립인가연월'] = holding_df['설립인가연월'].astype(str).str.replace(r'[년월일\s/\.]+', '-', regex=True)
holding_df['설립인가연월'] = holding_df['설립인가연월'].str.strip('-')
holding_df['설립인가연월'] = pd.to_datetime(holding_df['설립인가연월'], errors='coerce')

current_date = pd.to_datetime('2023-12-31')
proj_df['종료일자'] = proj_df['종료일자'].fillna(current_date)
proj_df['사업진행일수'] = (proj_df['종료일자'] - proj_df['시작일자']).dt.days

proj_df['사업진행일수'] = np.where(proj_df['사업진행일수'] < 0, np.nan, proj_df['사업진행일수'])

proj_summary = proj_df.groupby('자회사ID').agg(
    총참여사업수=('사업ID', 'count'),
    총지원금액=('지원금액', 'sum'),
    평균진행일수=('사업진행일수', 'mean')
).reset_index()

sub_df.to_csv('C:/Users/10414/cursor/26-1/mot716/wk08_0422/preprocess_data/subsidiaries.csv', index=False, encoding='utf8')
proj_df.to_csv('C:/Users/10414/cursor/26-1/mot716/wk08_0422/preprocess_data/projects.csv', index=False, encoding='utf8')
holding_df.to_csv('C:/Users/10414/cursor/26-1/mot716/wk08_0422/preprocess_data/holding_companies.csv', index=False, encoding='utf8')
