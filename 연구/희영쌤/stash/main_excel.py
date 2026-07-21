import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import re
import warnings

# statsmodels 경고 메시지 숨김
warnings.filterwarnings('ignore')

class TechHoldingAnalyzer:
    """산학연협력기술지주회사 5개년 데이터 분석 및 엑셀 리포트 자동화 모듈"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = pd.DataFrame()
        self.results = {} # 분석 결과를 저장할 딕셔너리
        
        self._load_and_merge_data()
        self._preprocess_data()
        
    def _load_and_merge_data(self):
        files = [
            "2020년_이상데이터 변경.csv",
            "2021년_이상데이터 변경.csv",
            "2022년_이상데이터변경.csv",
            "2023년_이상데이터 변경.csv",
            "2024년_이상데이터 변경.csv"
        ]
        
        dfs = []
        for f_name in files:
            file_path = os.path.join(self.data_dir, f_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='cp949')
                
                year = int(f_name[:4])
                if '연도' not in df.columns:
                    df['연도'] = year
                dfs.append(df)
            else:
                print(f"⚠️ 경고: {file_path} 파일을 찾을 수 없습니다.")
                
        if not dfs:
            raise FileNotFoundError("데이터 파일을 찾을 수 없습니다. data 폴더 위치를 확인하세요.")
            
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"✅ 데이터 로드 완료: 총 {len(self.df)}건 병합됨.")

    @staticmethod
    def _extract_num(val, default_map):
        if pd.isna(val): return 0
        val_str = str(val).replace(' ', '')
        match = re.search(r'\((\d+)[^\)]*\)', val_str)
        if match:
            return float(match.group(1))
        for k, v in default_map.items():
            if k in val_str:
                return v
        return 0

    def _preprocess_data(self):
        df = self.df.copy()
        df = df.rename(columns={'기술지주회사(코드)': '회사ID'})
        
        df['has_policy_rev'] = df['국책사업비(정부사업비)'] != '0원'
        df['has_biz_rev'] = (df['자회사배당수익'] != '0원') | \
                            (df['투자조합배당수익'] != '0원') | \
                            (df['자회사지분매각(일부+전부)'] != '0원')
        df['has_biz_rev_int'] = df['has_biz_rev'].astype(int)

        conditions = [
            (df['has_policy_rev'] == True) & (df['has_biz_rev'] == True),
            (df['has_policy_rev'] == True) & (df['has_biz_rev'] == False),
            (df['has_policy_rev'] == False) & (df['has_biz_rev'] == True),
            (df['has_policy_rev'] == False) & (df['has_biz_rev'] == False)
        ]
        df['org_type'] = np.select(conditions, ['이중형', '정책형', '사업형', '무수익형'], default='알수없음')

        df['자본금'] = df['자본금 규모 구분'].apply(lambda x: 
            1 if '10억미만' in str(x) else 2 if '10억이상~20억미만' in str(x) else 
            3 if '20억이상~30억미만' in str(x) else 4 if '30억이상~40억미만' in str(x) else 
            5 if '40억이상~50억미만' in str(x) else 6 if '50억이상~60억미만' in str(x) else 7
        )
        
        df['전담인력'] = df['전담인력'].apply(lambda x: self._extract_num(x, {'0명':0, '1~2명':1.5, '3~5명':4, '6~8명':7}))
        df['신규자회사수'] = df['자회사수(신규)'].apply(lambda x: self._extract_num(x, {'0개사':0, '1~2개사':1.5, '3~5개사':4, '6~8개사':7}))
        df['자회사규모'] = df['자회사 규모'].apply(lambda x: self._extract_num(x, {'10개미만':5, '10개이상~20개미만':15, '20개이상':20}))
        
        pol_map = {
            '0원': 0, '3천만원미만': 1, '3천만원이상~5천만원미만': 2, '5천만원이상~1억원미만': 3,
            '1억원이상~5억원미만': 4, '5억이상~10억미만': 5, '10억이상': 6
        }
        df['국책사업비'] = df['국책사업비(정부사업비)'].map(pol_map).fillna(0)
        
        df['업력'] = df['업력년수']
        df['업력구간'] = pd.cut(df['업력'], bins=[-1, 3, 7, 100], labels=['초기(0~3년)', '중기(4~7년)', '성숙기(8년이상)'])
        df['초기조직여부'] = (df['업력'] <= 5).astype(int)
        
        df['수도권여부'] = df['지역구분'].apply(lambda x: 1 if '서울' in str(x) or '경기' in str(x) or '인천' in str(x) else 0)
        df['지역'] = df['수도권여부'].map({1:'수도권', 0:'비수도권'})
        
        self.df = df
        print("✅ 데이터 전처리 완료.")

    def run_analysis(self):
        """모든 분석을 실행하고 결과를 self.results 딕셔너리에 저장"""
        print("⏳ 통계 모델링 및 분석 진행 중...")
        
        # --- Q1 ---
        self.results['Q1_연도별_수익구조'] = self.df.groupby('연도').agg(
            총회사수=('회사ID', 'count'),
            정책수익비중=('has_policy_rev', 'mean'),
            사업화수익비중=('has_biz_rev', 'mean'),
            이중형비중=('org_type', lambda x: (x == '이중형').mean())
        )
        
        df_sorted = self.df.sort_values(by=['회사ID', '연도'])
        df_sorted['prev_org_type'] = df_sorted.groupby('회사ID')['org_type'].shift(1)
        self.results['Q1_유형전환_Matrix'] = pd.crosstab(df_sorted['prev_org_type'], df_sorted['org_type'], normalize='index')

        # --- Q2 ---
        self.results['Q2_업력별_조직유형'] = pd.crosstab(self.df['업력구간'], self.df['org_type'], normalize='index')
        self.results['Q2_설립유형별_조직유형'] = pd.crosstab(self.df['설립유형'], self.df['org_type'], normalize='index')
        self.results['Q2_지역별_수익의존도'] = self.df.groupby('지역').agg(
            정책수익비중=('has_policy_rev', 'mean'), 사업화수익비중=('has_biz_rev', 'mean')
        )
        
        try:
            m_q2 = smf.logit("has_biz_rev_int ~ C(지역) + 자본금 + 전담인력 + 자회사규모", data=self.df).fit(disp=0)
            self.results['Q2_Logit_지역차이'] = m_q2.summary2().tables[1] # 회귀계수 테이블을 DataFrame으로 추출
        except Exception as e:
            pass

        # --- Q3 (Lag 모델) ---
        df_lag = df_sorted.copy()
        df_lag['전담인력_t1'] = df_lag.groupby('회사ID')['전담인력'].shift(-1)
        df_lag['신규자회사수_t1'] = df_lag.groupby('회사ID')['신규자회사수'].shift(-1)
        df_lag['has_biz_rev_int_t1'] = df_lag.groupby('회사ID')['has_biz_rev_int'].shift(-1)
        df_lag['has_biz_rev_int_t2'] = df_lag.groupby('회사ID')['has_biz_rev_int'].shift(-2)

        df_valid_t1 = df_lag.dropna(subset=['전담인력_t1', '신규자회사수_t1'])
        df_valid_rev_t1 = df_lag.dropna(subset=['has_biz_rev_int_t1'])
        df_valid_rev_t2 = df_lag.dropna(subset=['has_biz_rev_int_t2'])

        try:
            self.results['Q3_OLS_전담인력증가'] = smf.ols("전담인력_t1 ~ 국책사업비 + 전담인력", data=df_valid_t1).fit().summary2().tables[1]
            self.results['Q3_OLS_신규자회사증가'] = smf.ols("신규자회사수_t1 ~ 국책사업비 + 자회사규모", data=df_valid_t1).fit().summary2().tables[1]
            self.results['Q3_Logit_사업화수익_t1'] = smf.logit("has_biz_rev_int_t1 ~ 국책사업비 + 업력 + 자본금", data=df_valid_rev_t1).fit(disp=0).summary2().tables[1]
            self.results['Q3_Logit_사업화수익_t2'] = smf.logit("has_biz_rev_int_t2 ~ 국책사업비 + 업력 + 자본금", data=df_valid_rev_t2).fit(disp=0).summary2().tables[1]
            self.results['Q3_Logit_초기조직상호작용'] = smf.logit("has_biz_rev_int_t1 ~ 국책사업비 * 초기조직여부 + 자본금", data=df_valid_rev_t1).fit(disp=0).summary2().tables[1]
        except Exception as e:
            print(f"Q3 회귀분석 중 오류 발생: {e}")

    def export_to_excel(self, output_path="산학연기술지주_분석결과_2020_2024.xlsx"):
        """저장된 분석 결과를 엑셀 파일의 멀티 시트로 출력"""
        print(f"📊 엑셀 파일 생성 중: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 원본 데이터 스냅샷 저장
            self.df.to_excel(writer, sheet_name='RawData_전처리완료', index=False)
            
            # 1. Q1 관련 결과
            self.results['Q1_연도별_수익구조'].to_excel(writer, sheet_name='Q1_수익구조변화')
            self.results['Q1_유형전환_Matrix'].to_excel(writer, sheet_name='Q1_유형전환(Transition)')
            
            # 2. Q2 관련 결과 (한 시트에 여러 테이블 배치)
            start_row = 0
            sheet_q2 = 'Q2_조직특성별_분화'
            
            self.results['Q2_업력별_조직유형'].to_excel(writer, sheet_name=sheet_q2, startrow=start_row)
            start_row += len(self.results['Q2_업력별_조직유형']) + 3
            
            self.results['Q2_설립유형별_조직유형'].to_excel(writer, sheet_name=sheet_q2, startrow=start_row)
            start_row += len(self.results['Q2_설립유형별_조직유형']) + 3
            
            self.results['Q2_지역별_수익의존도'].to_excel(writer, sheet_name=sheet_q2, startrow=start_row)
            
            # 3. 회귀분석 결과 (Q2_4 및 Q3 모두 한 시트에 깔끔하게 나열)
            start_row = 0
            sheet_reg = 'Q3_회귀분석결과종합'
            
            regression_keys = [
                'Q2_Logit_지역차이', 'Q3_OLS_전담인력증가', 'Q3_OLS_신규자회사증가',
                'Q3_Logit_사업화수익_t1', 'Q3_Logit_사업화수익_t2', 'Q3_Logit_초기조직상호작용'
            ]
            
            for key in regression_keys:
                if key in self.results:
                    # 표 제목 쓰기
                    pd.DataFrame([f"■ {key} 모델 결과"]).to_excel(writer, sheet_name=sheet_reg, startrow=start_row, index=False, header=False)
                    start_row += 1
                    
                    # 실제 회귀계수 표 쓰기
                    self.results[key].to_excel(writer, sheet_name=sheet_reg, startrow=start_row)
                    start_row += len(self.results[key]) + 3

        print(f"✨ 완료! '{output_path}' 파일이 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    TARGET_DIR = "./data" # CSV 파일들이 들어있는 폴더
    OUTPUT_FILE = "산학연기술지주_분석결과_2020_2024.xlsx"
    
    print("🚀 데이터 분석 및 엑셀 리포트 추출 파이프라인 시작...\n")
    try:
        analyzer = TechHoldingAnalyzer(data_dir=TARGET_DIR)
        analyzer.run_analysis()
        analyzer.export_to_excel(output_path=OUTPUT_FILE)
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생: {e}")
