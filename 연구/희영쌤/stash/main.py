import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import re
import warnings

# statsmodels의 일부 경고 메시지 숨김 처리
warnings.filterwarnings('ignore')

class TechHoldingAnalyzer:
    """산학연협력기술지주회사 5개년(2020~2024) 패널 데이터 분석 모듈"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = pd.DataFrame()
        
        # 초기화 시 데이터 로드 및 전처리 자동 수행
        self._load_and_merge_data()
        self._preprocess_data()
        
    def _load_and_merge_data(self):
        """지정된 디렉토리에서 5개년 CSV 파일을 읽어 하나의 데이터프레임으로 병합"""
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
                
                # 연도 컬럼이 없으면 파일명에서 추출하여 추가
                year = int(f_name[:4])
                if '연도' not in df.columns:
                    df['연도'] = year
                dfs.append(df)
            else:
                print(f"⚠️ 경고: {file_path} 파일을 찾을 수 없습니다.")
                
        if not dfs:
            raise FileNotFoundError("데이터 파일을 하나도 로드하지 못했습니다. data 폴더 위치를 확인하세요.")
            
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"✅ 데이터 로드 완료: 총 {len(self.df)}건의 데이터가 병합되었습니다.\n")

    @staticmethod
    def _extract_num(val, default_map):
        """문자열에서 숫자를 추출하거나, 정해진 매핑 규칙에 따라 변환하는 헬퍼 함수"""
        if pd.isna(val): return 0
        val_str = str(val).replace(' ', '')
        # 괄호 안의 구체적 숫자(예: '20개이상(33개)') 추출
        match = re.search(r'\((\d+)[^\)]*\)', val_str)
        if match:
            return float(match.group(1))
        # 괄호가 없으면 매핑 사전 활용
        for k, v in default_map.items():
            if k in val_str:
                return v
        return 0

    def _preprocess_data(self):
        """통계 분석을 위한 변수 조작적 정의 및 수치형 변환"""
        df = self.df.copy()
        
        # 1. 식별자 및 수익 구조 정의
        df = df.rename(columns={'기술지주회사(코드)': '회사ID'})
        
        df['has_policy_rev'] = df['국책사업비(정부사업비)'] != '0원'
        df['has_biz_rev'] = (df['자회사배당수익'] != '0원') | \
                            (df['투자조합배당수익'] != '0원') | \
                            (df['자회사지분매각(일부+전부)'] != '0원')
        # Logit 분석을 위한 int 형 변환
        df['has_biz_rev_int'] = df['has_biz_rev'].astype(int)

        # 2. 4가지 조직 유형(Type) 분류
        conditions = [
            (df['has_policy_rev'] == True) & (df['has_biz_rev'] == True),
            (df['has_policy_rev'] == True) & (df['has_biz_rev'] == False),
            (df['has_policy_rev'] == False) & (df['has_biz_rev'] == True),
            (df['has_policy_rev'] == False) & (df['has_biz_rev'] == False)
        ]
        choices = ['이중형', '정책형', '사업형', '무수익형']
        df['org_type'] = np.select(conditions, choices, default='알수없음')

        # 3. 회귀분석을 위한 순서형/연속형 변수 변환
        df['자본금'] = df['자본금 규모 구분'].apply(lambda x: 
            1 if '10억미만' in str(x) else 
            2 if '10억이상~20억미만' in str(x) else 
            3 if '20억이상~30억미만' in str(x) else 
            4 if '30억이상~40억미만' in str(x) else 
            5 if '40억이상~50억미만' in str(x) else 
            6 if '50억이상~60억미만' in str(x) else 7
        )
        
        df['전담인력'] = df['전담인력'].apply(lambda x: self._extract_num(x, {'0명':0, '1~2명':1.5, '3~5명':4, '6~8명':7}))
        df['신규자회사수'] = df['자회사수(신규)'].apply(lambda x: self._extract_num(x, {'0개사':0, '1~2개사':1.5, '3~5개사':4, '6~8개사':7}))
        df['자회사규모'] = df['자회사 규모'].apply(lambda x: self._extract_num(x, {'10개미만':5, '10개이상~20개미만':15, '20개이상':20}))
        
        # 국책사업비 서열척도화
        pol_map = {
            '0원': 0, '3천만원미만': 1, '3천만원이상~5천만원미만': 2, '5천만원이상~1억원미만': 3,
            '1억원이상~5억원미만': 4, '5억이상~10억미만': 5, '10억이상': 6
        }
        df['국책사업비'] = df['국책사업비(정부사업비)'].map(pol_map).fillna(0)
        
        # 4. 그룹화 및 통제변수 생성
        df['업력'] = df['업력년수']
        df['업력구간'] = pd.cut(df['업력'], bins=[-1, 3, 7, 100], labels=['초기(0~3년)', '중기(4~7년)', '성숙기(8년이상)'])
        df['초기조직여부'] = (df['업력'] <= 5).astype(int)
        
        df['수도권여부'] = df['지역구분'].apply(lambda x: 1 if '서울' in str(x) or '경기' in str(x) or '인천' in str(x) else 0)
        df['지역'] = df['수도권여부'].map({1:'수도권', 0:'비수도권'})
        
        self.df = df
        print("✅ 데이터 전처리 및 분석 변수 생성 완료.\n")

    def analyze_q1(self):
        print("="*60)
        print("[질문 1] 수익 결합구조의 시계열적 변화 분석")
        print("="*60)
        
        # 1-1 ~ 1-3. 연도별 변화 추이
        trend = self.df.groupby('연도').agg(
            총회사수=('회사ID', 'count'),
            정책수익비중=('has_policy_rev', 'mean'),
            사업화수익비중=('has_biz_rev', 'mean'),
            이중형비중=('org_type', lambda x: (x == '이중형').mean())
        ).round(3)
        print("\n<연도별 수익구조 비중 변화>")
        print(trend)
        
        # 1-4. 전이 행렬 (Transition Matrix)
        df_sorted = self.df.sort_values(by=['회사ID', '연도'])
        df_sorted['prev_org_type'] = df_sorted.groupby('회사ID')['org_type'].shift(1)
        transition = pd.crosstab(df_sorted['prev_org_type'], df_sorted['org_type'], normalize='index').round(3)
        print("\n<조직 유형 간 전환 행렬 (t-1 -> t)>")
        print(transition)

    def analyze_q2(self):
        print("\n" + "="*60)
        print("[질문 2] 조직 특성에 따른 수익구조 분화 분석")
        print("="*60)
        
        print("\n<Q2-1. 업력 구간별 조직 유형 비중>")
        print(pd.crosstab(self.df['업력구간'], self.df['org_type'], normalize='index').round(3))
        
        print("\n<Q2-2. 설립유형(공동형/단독형)별 조직 유형 비중>")
        print(pd.crosstab(self.df['설립유형'], self.df['org_type'], normalize='index').round(3))
        
        print("\n<Q2-3. 지역별 정책/사업화 수익 의존도>")
        print(self.df.groupby('지역').agg(정책수익비중=('has_policy_rev', 'mean'), 
                                       사업화수익비중=('has_biz_rev', 'mean')).round(3))
        
        print("\n<Q2-4. 지역 차이 및 자원 역량이 사업화 수익에 미치는 영향 (Logit)>")
        try:
            model = smf.logit("has_biz_rev_int ~ C(지역) + 자본금 + 전담인력 + 자회사규모", data=self.df).fit(disp=0)
            print(model.summary().tables[1])
        except Exception as e:
            print(f"회귀분석 에러: {e}")

    def analyze_q3(self):
        print("\n" + "="*60)
        print("[질문 3] 정책사업 수익의 후속 성과 관련성 분석 (Lag 시계열 모델)")
        print("="*60)
        
        # 시차(Lag) 변수 생성
        df_lag = self.df.sort_values(by=['회사ID', '연도']).copy()
        df_lag['전담인력_t1'] = df_lag.groupby('회사ID')['전담인력'].shift(-1)
        df_lag['신규자회사수_t1'] = df_lag.groupby('회사ID')['신규자회사수'].shift(-1)
        df_lag['has_biz_rev_int_t1'] = df_lag.groupby('회사ID')['has_biz_rev_int'].shift(-1)
        df_lag['has_biz_rev_int_t2'] = df_lag.groupby('회사ID')['has_biz_rev_int'].shift(-2)

        df_valid_t1 = df_lag.dropna(subset=['전담인력_t1', '신규자회사수_t1'])
        df_valid_rev_t1 = df_lag.dropna(subset=['has_biz_rev_int_t1'])
        df_valid_rev_t2 = df_lag.dropna(subset=['has_biz_rev_int_t2'])

        print("\n<Q3-1. 정책수익(t) -> 전담인력 증가(t+1) (OLS)>")
        m1 = smf.ols("전담인력_t1 ~ 국책사업비 + 전담인력", data=df_valid_t1).fit()
        print(m1.summary().tables[1])

        print("\n<Q3-2. 정책수익(t) -> 신규 자회사 수 증가(t+1) (OLS)>")
        m2 = smf.ols("신규자회사수_t1 ~ 국책사업비 + 자회사규모", data=df_valid_t1).fit()
        print(m2.summary().tables[1])

        print("\n<Q3-3. 정책수익(t) -> 다음 해(t+1) 사업화수익 발생 (Logit)>")
        m3_1 = smf.logit("has_biz_rev_int_t1 ~ 국책사업비 + 업력 + 자본금", data=df_valid_rev_t1).fit(disp=0)
        print(m3_1.summary().tables[1])

        print("\n<Q3-3. 정책수익(t) -> 2년 후(t+2) 사업화수익 발생 (Logit)>")
        m3_2 = smf.logit("has_biz_rev_int_t2 ~ 국책사업비 + 업력 + 자본금", data=df_valid_rev_t2).fit(disp=0)
        print(m3_2.summary().tables[1])

        print("\n<Q3-4. 초기 조직 여부와 정책수익의 상호작용 효과 (t+1) (Logit)>")
        m4 = smf.logit("has_biz_rev_int_t1 ~ 국책사업비 * 초기조직여부 + 자본금", data=df_valid_rev_t1).fit(disp=0)
        print(m4.summary().tables[1])

if __name__ == "__main__":
    # 데이터가 저장된 폴더 경로 지정 (기본값: 현재 디렉토리의 data 폴더)
    TARGET_DIR = "./data"
    
    print("분석 파이프라인 시작...\n")
    try:
        analyzer = TechHoldingAnalyzer(data_dir=TARGET_DIR)
        analyzer.analyze_q1()
        analyzer.analyze_q2()
        analyzer.analyze_q3()
        print("\n✅ 모든 가설 검증 프로세스가 성공적으로 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생: {e}")
