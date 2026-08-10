import os
import json
import time
import requests
import pandas as pd
import re
from dotenv import load_dotenv

# .env 파일에서 OpenRouter API 키 로드
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# 평가할 모델 목록
MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.3",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-medium-3-5",
    "deepseek/deepseek-v4-pro"
]

# 13가지 위험 주제 (Marris et al. 1996)
RISK_ISSUES = [
    "Sunbathing", "Food colourings", "Genetic engineering", "Nuclear power", 
    "Mugging", "Home accidents", "Ozone depletion", "Car driving", 
    "Microwave ovens", "AIDS", "War", "Terrorism", "Alcoholic drinks"
]

# 9가지 질적 아웃레이지 요인 (독립 변수)
OUTRAGE_FACTORS = {
    "Involuntariness": "Do you think people are exposed to {risk_issue} voluntarily or involuntarily?\nScale: 1 (Risk assumed voluntarily) to 5 (Risk assumed involuntarily)",
    "Delayed_effects": "Do you think the harmful effects of {risk_issue} are likely to occur at some later time?\nScale: 1 (No effect delayed) to 5 (Many effects delayed)",
    "Severity": "Do you think people will die from {risk_issue} when things go wrong?\nScale: 1 (Certain not to be fatal) to 5 (Certain to be fatal)",
    "Dread": "Are the risks of {risk_issue} ones that you can think about reasonably calmly, or are they ones that you have a great dread for on the level of gut reaction?\nScale: 1 (Think about calmly) to 5 (Think about with dread)",
    "Catastrophic_potential": "Do you think {risk_issue} can harm a large number of people at once or only one person at a time?\nScale: 1 (One person at a time) to 5 (Many people at once)",
    "Harm_to_future_generations": "Do you think {risk_issue} poses risks to future generations?\nScale: 1 (Very little threat) to 5 (Very great threat)",
    "Lack_of_knowledge_exposed": "Do you think the risks associated with {risk_issue} are well known by the persons who are exposed to the risk?\nScale: 1 (Risk well known) to 5 (Risk not known)",
    "Lack_of_knowledge_scientists": "Do you think the risks associated with {risk_issue} are well known by scientists?\nScale: 1 (Risk well known) to 5 (Risk not known)",
    "Unfairness": "Do you think that the people who are exposed to the risks of {risk_issue} are the same as those who receive the benefits?\nScale: 1 (Same people / Fair) to 5 (Not same people / Unfair)"
}

# 2가지 전반적 위험 인식 (종속 변수)
OVERALL_PERCEPTIONS = {
    "Riskiness": "On a scale of 1 to 5, how much risk do you think is associated with {risk_issue}?\nScale: 1 (Very low risk) to 5 (Very high risk)",
    "Unacceptability": "On a scale of 1 to 5, how acceptable do you feel the current risk is for {risk_issue}?\nScale: 1 (Completely acceptable) to 5 (Completely unacceptable)"
}

def query_model(model_name, system_prompt, user_prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000", 
        "X-Title": "Risk Perception Research (Study 2)" 
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1000 
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            
            if response.status_code != 200:
                print(f"\n[HTTP {response.status_code} Error] 상세 원인: {response.text}")
                response.raise_for_status()
                
            result = response.json()
            message_data = result['choices'][0]['message']
            message_content = message_data.get('content')
            
            if message_content is None:
                finish_reason = result['choices'][0].get('finish_reason', 'unknown')
                raise ValueError(f"모델이 빈 응답을 반환함. (이유: {finish_reason})")
                
            answer_text = message_content.strip()
            
            # 모델이 숫자 외의 부연설명을 붙이는 경우를 대비해 첫 번째 연속된 숫자(1~5)만 추출
            match = re.search(r'[1-5]', answer_text)
            if not match:
                raise ValueError(f"응답에서 1~5 사이의 숫자를 찾을 수 없음. (원본: {answer_text})")
                
            return int(match.group())
            
        except Exception as e:
            print(f"  ↪ [재시도 {attempt+1}/{max_retries}] 통신 오류: {e}")
            if "400" in str(e):
                print("존재하지 않는 모델 ID")
                break 
            time.sleep(2)
            
    print(f"!!!!! [{model_name}] 해당 문항 응답 실패 -> 결측치(None) 처리 !!!!!")
    return None

def run_experiment():
    results = []
    
    # 모델이 오직 1부터 5 사이의 숫자 하나만 반환하도록 강제하는 시스템 프롬프트
    system_prompt = (
        "You are participating in a risk perception survey. "
        "Read the question and the scale carefully. "
        "You MUST return ONLY a single integer between 1 and 5. "
        "Do not include any words, punctuation, explanations, or periods."
    )

    print("🚀 [Study 2] AI 위험 인식 및 아웃레이지 측정 실험을 시작합니다...")

    for model in MODELS:
        print(f"\n=========================================")
        print(f"[{model}] 평가 시작...")
        print(f"=========================================")
        
        for issue in RISK_ISSUES:
            print(f"\n !!!!! 현재 위험 주제: {issue} !!!!!")
            
            # 행(Row) 데이터 초기화
            row_data = {
                "Model": model,
                "Risk_Issue": issue
            }
            
            # 1. 9가지 질적 아웃레이지 요인 질의
            for factor_name, prompt_template in OUTRAGE_FACTORS.items():
                user_prompt = prompt_template.format(risk_issue=issue)
                print(f"    - 측정 중 [{factor_name}]...", end=" ")
                
                answer = query_model(model, system_prompt, user_prompt)
                row_data[factor_name] = answer
                print(f"응답: {answer}")
                
            # 2. 2가지 종속 변수(위험성, 불수용성) 질의
            for perception_name, prompt_template in OVERALL_PERCEPTIONS.items():
                user_prompt = prompt_template.format(risk_issue=issue)
                print(f"    - 측정 중 [{perception_name}]...", end=" ")
                
                answer = query_model(model, system_prompt, user_prompt)
                row_data[perception_name] = answer
                print(f"응답: {answer}")
                
            results.append(row_data)
            
            # 중간 저장: 각 주제가 끝날 때마다 CSV 덮어쓰기 (네트워크 단절 대비)
            df_temp = pd.DataFrame(results)
            df_temp.to_csv("study2_risk_perception_results.csv", index=False, encoding='utf-8-sig')

    print("\n실험 종료...")

if __name__ == "__main__":
    run_experiment()
