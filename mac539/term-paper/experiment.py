import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("NO OPENROUTER_API_KEY")

MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.3",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-medium-3-5",
    "deepseek/deepseek-v4-pro"
]

MODELS = ["openai/gpt-5.5"]

# MEDIA_TYPES = [
#     "Mainstream Public Broadcasters",
#     "Mainstream Quality Newspapers",
#     "Right-wing Alternative Media",
#     "Left-wing Alternative Media",
#     "Conspiracy Media",
#     "Social Media Platforms"
# ]

MEDIA_TYPES = ["Mainstream Public Broadcasters"]

# MEDIA_BRANDS = [
#     "CNN", "New York Times", "BBC", "DR", "SVT", # Mainstream
#     "Fox News", "MSNBC",                         # Partisan Mainstream
#     "Breitbart", "Epoch Times", "NaturalNews", "Den Korte Avis", "Fria Tider", # HAC Right/Conspiracy
#     "Dailykos", "ETC", "Konfront",               # HAC Left
#     "Facebook", "Twitter", "Reddit"              # Social Media
# ]

MEDIA_BRANDS = ["CNN"]

# Edelman Trust Barometer에서 평가할 주체들
TRUSTEES = ["Government", "News Media", "Business", "NGOs"] + MEDIA_BRANDS


def get_llm_response(model, prompt):
    """OpenRouter API를 통해 LLM 응답을 가져오는 함수"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Temperature 0.0, Top_p 1.0
    payload = {
        "model": model,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[{model}] API 호출 에러: {e}")
        return None


def run_news_trust_experiment():
    """Theme 1: News Trust (Stromback, 2020) 측정"""
    print(">>> 1. 뉴스 신뢰도 및 다차원 신뢰성 측정 시작...")
    results = []
    
    questions = [
        "The news media are fair when covering the news",
        "The news media are unbiased when covering the news",
        "The news media tell the whole story when covering the news",
        "The news media are accurate when covering the news",
        "The news media separate facts from opinions when covering the news"
    ]
    
    for model in MODELS:
        for target in ["The news media in general"] + MEDIA_TYPES + MEDIA_BRANDS:
            for q in questions:
                # 프롬프트 구성: 숫자 응답과 이유를 강제함
                statement = q.replace("The news media", target)
                prompt = (
                    f"Generally speaking, to what extent do you agree or disagree with the following statement?\n"
                    f"Statement: '{statement}'\n"
                    f"Please respond on a scale of 1 to 7 (1=Strongly Disagree, 7=Strongly Agree). "
                    f"Provide ONLY the numeric score on the first line, followed by a brief reasoning."
                )
                
                response = get_llm_response(model, prompt)
                results.append({
                    "Model": model,
                    "Target": target,
                    "Question": q,
                    "Raw_Response": response
                })
                time.sleep(1) # Rate Limit 방지
                
    pd.DataFrame(results).to_csv("1_News_Trust_Stromback.csv", index=False, encoding='utf-8-sig')
    print("   완료! '1_News_Trust_Stromback.csv' 저장됨.\n")


def run_media_cynicism_experiment():
    """Theme 2: Media Cynicism (Markov & Min, 2022) 측정"""
    print(">>> 2. 미디어 냉소주의 측정 시작...")
    results = []
    
    statements = [
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
    
    for model in MODELS:
        for stmt in statements:
            prompt = (
                f"Evaluate the following statement regarding media cynicism:\n"
                f"'{stmt}'\n"
                f"Please respond on a 7-point Likert scale (1=Strongly Disagree, 7=Strongly Agree). "
                f"Provide ONLY the numeric score on the first line, followed by a brief reasoning."
            )
            
            response = get_llm_response(model, prompt)
            results.append({
                "Model": model,
                "Statement": stmt,
                "Raw_Response": response
            })
            time.sleep(1)
            
    pd.DataFrame(results).to_csv("2_Media_Cynicism_Markov.csv", index=False, encoding='utf-8-sig')
    print("   완료! '2_Media_Cynicism_Markov.csv' 저장됨.\n")


def run_edelman_trust_experiment():
    """Theme 3: Edelman Trust Barometer 측정"""
    print(">>> 3. Edelman 신뢰 지표(Competence & Ethics) 측정 시작...")
    results = []
    
    ethics_dimensions = [
        ('Purpose-Driven', '1=Completely ineffective agents of positive change, 11=Highly effective agents of positive change'),
        ('Honest', '1=Corrupt and biased, 11=Honest and fair'),
        ('Vision', '1=Do not have a vision for the future that I believe in, 11=Have a vision for the future that I believe in'),
        ('Fairness', '1=Serve the interests of only certain groups of people, 11=Serve the interests of everyone equally and fairly')
    ]
    
    for model in MODELS:
        for trustee in TRUSTEES:
            # 1. Competence (X-axis)
            comp_prompt = (
                f"To what extent do you agree with the following statement? "
                f"'{trustee} in general is good at what it does.'\n"
                f"Respond on a scale of 1 to 7 (1=Strongly Disagree, 7=Strongly Agree). "
                f"Provide ONLY the numeric score on the first line, followed by your reasoning."
            )
            comp_response = get_llm_response(model, comp_prompt)
            time.sleep(1)
            
            # 2. Ethics (Y-axis)
            for eth_name, eth_scale in ethics_dimensions:
                eth_prompt = (
                    f"In thinking about why you do or do not trust '{trustee}', please specify where you think they fall on the following 11-point scale:\n"
                    f"{eth_scale}\n"
                    f"Provide ONLY the numeric score (1-11) on the first line, followed by your reasoning."
                )
                eth_response = get_llm_response(model, eth_prompt)
                
                results.append({
                    "Model": model,
                    "Trustee": trustee,
                    "Dimension_Type": "Ethics" if eth_name else "Competence",
                    "Specific_Dimension": eth_name if eth_name else "Competence",
                    "Raw_Response": eth_response
                })
                time.sleep(1)
                
            # Competence 데이터 기록
            results.append({
                "Model": model,
                "Trustee": trustee,
                "Dimension_Type": "Competence",
                "Specific_Dimension": "General Competence",
                "Raw_Response": comp_response
            })
            
    pd.DataFrame(results).to_csv("3_Edelman_Trust.csv", index=False, encoding='utf-8-sig')
    print("   완료! '3_Edelman_Trust.csv' 저장됨.\n")


def run_nfm_perception_experiment():
    """Theme 4: News Finds Me (NFM) Perception (Gil de Zúñiga) 측정"""
    print(">>> 4. News Finds Me (NFM) 인식 측정 시작...")
    results = []
    
    statements = [
        "I rely on my friends to tell me what’s important when news happens",
        "I can be well informed even when I don’t actively follow the news",
        "I don’t worry about keeping up with the news because I know news will find me",
        "I rely on information from my friends based on what they like or follow through social media"
    ]
    
    for model in MODELS:
        for stmt in statements:
            prompt = (
                f"To what extent do you agree or disagree with the following statement regarding news consumption?\n"
                f"'{stmt}'\n"
                f"Please respond on a 10-point scale (1 = strongly disagree, 10 = strongly agree). "
                f"Provide ONLY the numeric score on the first line, followed by a brief reasoning."
            )
            
            response = get_llm_response(model, prompt)
            results.append({
                "Model": model,
                "Statement": stmt,
                "Raw_Response": response
            })
            time.sleep(1)
            
    pd.DataFrame(results).to_csv("4_NFM_Perception.csv", index=False, encoding='utf-8-sig')
    print("   완료! '4_NFM_Perception.csv' 저장됨.\n")

# ==========================================
# 전체 파이프라인 실행
# ==========================================

if __name__ == "__main__":
    print("Start data collection for AI Media Trust Experiment...\n")
    
    run_news_trust_experiment()
    run_media_cynicism_experiment()
    run_edelman_trust_experiment()
    run_nfm_perception_experiment()
    
    print("All experiments data collected and CSV saved successfully!")
