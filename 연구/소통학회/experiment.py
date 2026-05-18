import os
import json
import time
import requests
import pandas as pd
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

def load_questions(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']

# 파일 경로 (실행 환경에 맞게 유지)
grid_questions = load_questions('연구/소통학회/grid_questions.json')
group_questions = load_questions('연구/소통학회/group_questions.json')

def query_model(model_name, system_prompt, user_prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000", 
        "X-Title": "Cultural Prototype Research" 
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0, 
        "top_p": 1.0,
        "max_tokens": 1000 # 추론형 모델을 위한 넉넉한 토큰 할당
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
            
            # 빈 응답(None) 방어 로직
            if message_content is None:
                finish_reason = result['choices'][0].get('finish_reason', 'unknown')
                raise ValueError(f"모델이 빈 응답을 반환했습니다. (이유: {finish_reason})")
                
            answer_text = message_content.strip()
            
            # 텍스트에서 숫자만 추출
            digits = ''.join(filter(str.isdigit, answer_text))
            if not digits:
                raise ValueError(f"응답에 숫자가 없습니다. (원본: {answer_text})")
                
            return int(digits)
            
        except Exception as e:
            print(f"  ↪ [재시도 {attempt+1}/{max_retries}] 통신 지연 또는 오류: {e}")
            if "400" in str(e):
                print("  💡 400 에러 발생: 지원되지 않는 Model ID일 수 있습니다.")
                break 
            time.sleep(2)
            
    print(f"  ❌ [{model_name}] 해당 문항의 응답을 받아오지 못해 결측치(None) 처리합니다.")
    return None

def run_experiment():
    results = []
    
    # 논문에 기반한 강력한 시스템 프롬프트
    system_prompt = (
        "You must answer the following question. "
        "Return ONLY a single integer corresponding to your choice. "
        "Do not include any explanations, introductory text, or punctuation."
    )

    print("🚀 인공지능 문화원형 진단(Grid-Group) 실험을 시작합니다...")

    for model in MODELS:
        print(f"\n[{model}] 모델 평가 진행 중...")
        model_result = {"Model": model}
        
        # Grid 11문항 테스트
        for i, q_data in enumerate(grid_questions):
            print(f" - Grid Q{i+1} 질의 중...", end=" ")
            answer = query_model(model, system_prompt, q_data['question'])
            print(f"응답: {answer}")
            model_result[f"Grid_Q{i+1}"] = answer
            
        # Group 11문항 테스트
        for i, q_data in enumerate(group_questions):
            print(f" - Group Q{i+1} 질의 중...", end=" ")
            answer = query_model(model, system_prompt, q_data['question'])
            print(f"응답: {answer}")
            model_result[f"Group_Q{i+1}"] = answer
            
        results.append(model_result)
        
        # 중간 저장 (만약 중간에 스크립트가 끊기더라도 지금까지의 데이터를 보존하기 위함)
        df_temp = pd.DataFrame(results)
        df_temp.to_csv("ai_cultural_prototypes_results.csv", index=False, encoding='utf-8-sig')

    print("\n✅ 모든 실험이 성공적으로 완료되었습니다!")
    print("📊 결과가 'ai_cultural_prototypes_results.csv' 파일에 저장되었습니다.")

if __name__ == "__main__":
    run_experiment()
