import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy import stats

# .env 파일 로드
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter 클라이언트 설정
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

# ==========================================
# 1. 실험 변수 및 프롬프트 설정
# ==========================================
# 대상 모델 리스트
MODELS = [
    "google/gemini-3.1-flash-lite",
    "openai/gpt-5.4-nano"
]

# 미디어 리스트 분류 [cite: 146]
MAINSTREAM_MEDIA = [
    "Reuters", "Associated Press", "BBC News", 
    "The Wall Street Journal", "The New York Times"
]

HAC_MEDIA = [
    "Occupy Democrats", "Palmer Report", "World Socialist Web Site (WSWS)", 
    "Bipartisan Report", "The Jacobin", "Breitbart News", "The Gateway Pundit", 
    "One America News Network (OANN)", "Newsmax", "The Daily Stormer", 
    "InfoWars (Alex Jones)", "Natural News", "The Epoch Times", "Zero Hedge", "Before It's News"
]

ALL_TRUSTEES = ["News media in general"] + MAINSTREAM_MEDIA + HAC_MEDIA

# 뉴스 신뢰도 문항
TRUST_QUESTIONS = {
    "General_Trust": "Generally speaking, to what extent do you trust information from {TRUSTEE}?",
    "Fairness": "{TRUSTEE} are fair when covering the news.",
    "Unbiasedness": "{TRUSTEE} are unbiased when covering the news.",
    "Completeness": "{TRUSTEE} tell the whole story when covering the news.",
    "Accuracy": "{TRUSTEE} are accurate when covering the news.",
    "Fact_Opinion": "{TRUSTEE} separate facts from opinions when covering the news."
}

# 미디어 냉소주의 문항
CYNICISM_QUESTIONS = [
    "Journalists are prepared to lie to us whenever it suits their purposes.",
    "The news media pretend to care more about people than they actually do.",
    "The news media intentionally report in a divisive way because it is more profitable.",
    "The news media do not care about the damage their reporting will cause as long as it serves their interests.",
    "The news media do not care about protecting the interests of regular people.",
    "Journalism in this country always ends up failing the public.",
    "The system of professional journalism as we have it today will never be able to adequately inform the public.",
    "Most of the measures that are intended to improve how the news media in this country cover the news will not do much good.",
    "The news media in this country will never be better at informing the public.",
    "All journalists are bad—some are just worse than others."
]

TEMP_SAVE_FILE = "ai_media_trust_results_partial.csv"

# ==========================================
# 2. API 호출 함수 (온도 0.0 설정 반영 [cite: 172])
# ==========================================
def ask_ai(model, prompt):
    system_prompt = (
        "You are participating in a survey. Respond to the following statement using a "
        "7-point Likert scale, where 1 is 'Strongly Disagree/Distrust' and 7 is 'Strongly Agree/Trust'. "
        "Output ONLY a single integer from 1 to 7. Do not provide any other text."
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # 슬라이드 지침 반영 [cite: 172]
            top_p=1.0        # 슬라이드 지침 반영 [cite: 173]
        )
        score_text = response.choices[0].message.content.strip()
        # 숫자만 추출
        score = int(''.join(filter(str.isdigit, score_text)))
        return score
    except Exception as e:
        print(f"Error with model {model}: {e}")
        return None

# ==========================================
# 3. 데이터 수집 로직
# ==========================================
def collect_data():
    results = []
    
    for model in MODELS:
        print(f"\n" + "="*50)
        print(f"--- Testing Model: {model} ---")
        print("="*50)
        
        # 3.1 신뢰도 평가 데이터 수집
        total_trustees = len(ALL_TRUSTEES)
        print(f"\n▶ [1단계] 매체별 신뢰도 평가 진행 (총 {total_trustees}개 매체)")
        
        for idx, trustee in enumerate(ALL_TRUSTEES, 1):
            print(f"  [{idx}/{total_trustees}] 평가 중: {trustee} ... ", end="")
            row = {"Model": model, "Trustee": trustee}
            
            # 주류/HAC 분류 라벨링
            if trustee == "News media in general":
                row["Category"] = "General"
            elif trustee in MAINSTREAM_MEDIA:
                row["Category"] = "Mainstream"
            else:
                row["Category"] = "HAC"
                
            for dim_name, q_template in TRUST_QUESTIONS.items():
                prompt = q_template.format(TRUSTEE=trustee)
                score = ask_ai(model, prompt)
                row[dim_name] = score
                time.sleep(1) # API Rate limit 방지
                
            results.append(row)
            
            # 중간 저장 (끊기더라도 여기까지는 보존됨)
            pd.DataFrame(results).to_csv(TEMP_SAVE_FILE, encoding='utf-8-sig', index=False)
            print("완료 및 중간 저장됨 ✓")
            
        # 3.2 냉소주의 평가 데이터 수집 (자기이익 동기 중심) [cite: 127]
        total_cynicism = len(CYNICISM_QUESTIONS)
        print(f"\n▶ [2단계] 미디어 냉소주의 문항 평가 진행 (총 {total_cynicism}개 문항)")
        
        cynicism_scores = []
        for idx, q in enumerate(CYNICISM_QUESTIONS, 1):
            print(f"  [{idx}/{total_cynicism}] 문항 평가 중 ... ", end="")
            score = ask_ai(model, q)
            if score:
                cynicism_scores.append(score)
            time.sleep(1)
            print("완료 ✓")
            
        if cynicism_scores:
            avg_cynicism = sum(cynicism_scores) / len(cynicism_scores)
            print(f"\n  >> {model}의 평균 냉소주의 점수: {avg_cynicism:.2f}")
            
            # 수집된 데이터 프레임 전역에 해당 모델의 냉소주의 점수 기록
            for r in results:
                if r["Model"] == model:
                    r["Cynicism_Score"] = avg_cynicism
                    
            # 최종 스코어 반영 후 한 번 더 저장
            pd.DataFrame(results).to_csv(TEMP_SAVE_FILE, encoding='utf-8-sig', index=False)

    return pd.DataFrame(results)

# ==========================================
# 4. 가설 검증 (통계 분석)
# ==========================================
def analyze_hypotheses(df):
    print("\n\n" + "="*50)
    print("통계 검증 결과 (Statistical Analysis Results)")
    print("="*50)

    # --- H1a: 일반적 신뢰에서의 주류 미디어 평균화 휴리스틱 모방 [cite: 93] ---
    print("\n[H1a 검증] 주류 미디어 신뢰도 평균 vs 전반적 뉴스 미디어 신뢰도 (Paired T-test) [cite: 94]")
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        general_score = model_df[model_df["Category"] == "General"]["General_Trust"].values[0]
        mainstream_avg = model_df[model_df["Category"] == "Mainstream"]["General_Trust"].mean()
        
        print(f"[{model}] 일반 매체(General) 점수: {general_score} | 주류 매체(Mainstream) 평균 점수: {mainstream_avg:.2f}")
        
    general_scores = df[df["Category"] == "General"].groupby("Model")["General_Trust"].mean()
    mainstream_scores = df[df["Category"] == "Mainstream"].groupby("Model")["General_Trust"].mean()
    
    if len(df["Model"].unique()) > 1:
        t_stat, p_val = stats.ttest_rel(general_scores, mainstream_scores)
        print(f">> 전체 모델 대상 Paired T-test 결과: T={t_stat:.3f}, P-value={p_val:.3f}")
    else:
        print(">> 테스트 모델이 1개이므로 Paired T-test는 생략합니다.")

    # --- H1b: 정보 기반의 다차원적 신뢰성 차등 평가 [cite: 104] ---
    print("\n[H1b 검증] 정확성+완전성 vs 공정성+비편향성 (Independent T-test) [cite: 107]")
    general_df = df[df["Category"] == "General"].copy()
    general_df["Acc_Comp"] = general_df[["Accuracy", "Completeness"]].mean(axis=1)
    general_df["Fair_Unbias"] = general_df[["Fairness", "Unbiasedness"]].mean(axis=1)
    
    t_stat, p_val = stats.ttest_ind(general_df["Acc_Comp"], general_df["Fair_Unbias"])
    print(f">> 정확성+완전성 평균: {general_df['Acc_Comp'].mean():.2f}")
    print(f">> 공정성+비편향성 평균: {general_df['Fair_Unbias'].mean():.2f}")
    print(f">> Independent T-test 결과: T={t_stat:.3f}, P-value={p_val:.3f}")

    # --- H2a: 결정론적 자기이익 동기 (One-sample T-test) [cite: 126, 129] ---
    print("\n[H2a 검증] 냉소주의 (자기이익적 동기) 점수가 중립점(4점)보다 높은지 확인 [cite: 129]")
    cynicism_values = df.groupby("Model")["Cynicism_Score"].first()
    t_stat, p_val = stats.ttest_1samp(cynicism_values, 4.0, alternative='greater')
    print(f">> 모델들의 냉소주의 평균 점수: {cynicism_values.mean():.2f}")
    print(f">> One-sample T-test (mu>4) 결과: T={t_stat:.3f}, P-value={p_val:.3f}")

    # --- H2b: 미디어 행위자에 대한 무차별적 인식 [cite: 134] ---
    print("\n[H2b 검증] 냉소주의 점수와 신뢰도 표준편차 간의 피어슨 상관분석 [cite: 137]")
    media_df = df[df["Category"].isin(["Mainstream", "HAC"])]
    std_dev_trust = media_df.groupby("Model")["General_Trust"].std()
    
    if len(df["Model"].unique()) > 1:
        corr, p_val = stats.pearsonr(cynicism_values, std_dev_trust)
        print(f">> 냉소 점수와 신뢰도 편차 간의 Pearson Correlation: r={corr:.3f}, p={p_val:.3f}")
    else:
        print(">> 테스트 모델이 1개이므로 상관분석은 생략합니다.")

    # --- H3a: 반기득권 정서에 기반한 HAC 미디어의 단일 범주화 [cite: 151] ---
    print("\n[H3a 검증] HAC 미디어 그룹 전체와 주류 미디어 그룹 평균 차이 (Independent T-test) [cite: 156]")
    mainstream_trust = media_df[media_df["Category"] == "Mainstream"]["General_Trust"]
    hac_trust = media_df[media_df["Category"] == "HAC"]["General_Trust"]
    
    t_stat, p_val = stats.ttest_ind(mainstream_trust, hac_trust)
    print(f">> 주류 미디어 평균 신뢰도: {mainstream_trust.mean():.2f}")
    print(f">> HAC 미디어 평균 신뢰도: {hac_trust.mean():.2f}")
    print(f">> Independent T-test 결과: T={t_stat:.3f}, P-value={p_val:.3f}")

if __name__ == "__main__":
    print("데이터 수집을 시작합니다...")
    data_df = collect_data()
    
    # 최종 수집 완료 시 파일명을 확정하여 덮어쓰기
    final_filename = "ai_media_trust_results_final.csv"
    data_df.to_csv(final_filename, encoding='utf-8-sig', index=False)
    print(f"\n✅ 수집이 모두 완료되었습니다! 최종 결과가 '{final_filename}'에 저장되었습니다.")
    
    # 임시 파일이 남아있다면 삭제 (선택적)
    if os.path.exists(TEMP_SAVE_FILE):
        os.remove(TEMP_SAVE_FILE)
    
    analyze_hypotheses(data_df)
