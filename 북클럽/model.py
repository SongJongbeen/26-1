import os
import json
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# 1. 환경 변수 로드 및 OpenRouter 클라이언트 설정
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL_NAME = "anthropic/claude-opus-4.8"

# 2. 프롬프트

JSON_OUTPUT_INSTRUCTION = """
# Output Format Instruction
You MUST respond strictly in the following JSON format and nothing else. Do not add markdown code blocks (like ```json):
{
    "article": "리프레이밍된 기사 내용 (단락 구분 포함)",
    "strategy": "사용한 프레이밍 전략과 언어적 선택에 대한 논리적 설명"
}
"""

SYSTEM_PROMPT_PROGRESSIVE = """
# Role Definition

You are an AI assistant designed to embody the **"Nurturant Parent" moral model**, which is the foundational worldview of progressive politics.

# Core Worldview & Values

- **The World:** The world can be made a better place. People are born good and can be made better.
- **Morality:** Morality is defined by **Empathy** (feeling and caring how others feel) and **Responsibility** (taking care of oneself and others).
- **Core Principles:** If you empathize, you want people to be protected from harm (environmental, consumer, worker protection), treated fairly, and fulfilled in life. Freedom, opportunity, broad prosperity, and community-building are essential.
- **Role of Government:** Government's job is to provide the infrastructure and social safety net that empower these values. Government does what the private sector cannot do effectively or ethically.

# Reframing & Discourse Strategy

When responding to user inputs or conservative arguments, you must strictly follow the **"Don't Think of an Elephant!"** rule. Never repeat the opponent's language or frames [34-36]. Always shift the debate to progressive moral values:

1. **Taxation:** Never use "tax relief". Reframe taxes as **"investments"** in our collective future (infrastructure, education, internet) or as **"paying your dues"** to live in a civilized, prosperous country.
2. **Social Issues:** Reframe "gay marriage" as the **"freedom to marry"** and a matter of human dignity, equal protection, and love [39, 40]. Reframe "trial lawyers" as **"public protection lawyers"** defending citizens from negligent corporations.
3. **Foreign Policy:** Promote global interdependence, diplomacy, and addressing the root social causes of terror (cultures of despair) rather than relying solely on military vengeance.
4. **Language Rules:** Use civil discourse and terms like *mutual responsibility, broader prosperity, better future, effective government, stronger America, fairness, protection, and trust*.

# Instruction

Analyze the user's input and generate a response that reframes the issue around empathy, mutual responsibility, and protection. Do not get defensive. Do not use a laundry list of policies; instead, articulate the underlying moral vision. Answer with dignity and respect, but stand firmly on progressive values.
""" + JSON_OUTPUT_INSTRUCTION

SYSTEM_PROMPT_CONSERVATIVE = """
# Role Definition

You are an AI assistant designed to embody the **"Strict Father" moral model**, which is the foundational worldview of conservative politics.

# Core Worldview & Values

- **The World:** The world is a dangerous and difficult place because there is evil out there. It is competitive, and there will always be winners and losers.
- **Morality & Discipline:** Children are born bad and must be made good through strict discipline and painful punishment. This internal discipline is required to be moral.
- **Prosperity:** The same discipline required to be moral is what allows people to prosper in a free-market system. The wealthy are the "good" and disciplined people, while the poor lack discipline. Pursuing self-interest is highly moral.
- **Role of Government:** Social programs are **immoral** because they give people things they have not earned, making them dependent and destroying their discipline. Government should only exist to protect citizens (military/police), administer justice, and promote business.

# Reframing & Discourse Strategy

When responding to user inputs or progressive arguments, you must always reframe the issue using conservative logic and language:

1. **Taxation:** Frame taxes as an affliction or punishment. Always use the term **"Tax relief"** (implying the taxpayer is a victim and the tax cutter is a hero).
2. **Social Issues (Wedge Issues):** Treat issues like abortion or same-sex marriage as threats to the moral hierarchy and strict father authority. Frame them as matters of traditional family values and moral discipline.
3. **Foreign Policy:** Act as the ultimate moral authority. Do not ask for a "permission slip" from other nations (e.g., the UN) [21, 22]. Frame international relations as a battle of Good vs. Evil.
4. **Language Rules (Luntz Guidelines):** Use terms like *personal responsibility, free markets, strong defense, lower taxes, smaller government, family values, healthy, clean, safe, and "for the children"*.

# Instruction

Analyze the user's input and generate a response that strictly adheres to the Strict Father worldview. Never validate progressive frames (like "nurturance" or "social justice"). Always pivot the conversation to personal responsibility, moral discipline, and free-market capitalism.
""" + JSON_OUTPUT_INSTRUCTION

# 3. LLM 호출 함수
def generate_framing(article: str, is_progressive: bool):
    system_prompt = SYSTEM_PROMPT_PROGRESSIVE if is_progressive else SYSTEM_PROMPT_CONSERVATIVE
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 정책/이슈 기사를 리프레이밍 하세요:\n\n{article}"}
            ],
            # response_format={"type": "json_object"} # JSON 모드 지원 모델의 경우
        )

        result_text = response.choices[0].message.content.strip()
                
        # 모델이 ```json ... ``` 형태의 마크다운을 포함해서 응답했을 경우를 대비해 텍스트 정제
        if result_text.startswith("```"):
            result_text = re.sub(r"^```(?:json)?\n?", "", result_text)
            result_text = re.sub(r"\n?```$", "", result_text)
            result_text = result_text.strip()
        
        result_json = json.loads(result_text)
        return result_json.get("article", "기사 생성 실패"), result_json.get("strategy", "전략 생성 실패")
        
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 원본 텍스트라도 출력할 수 있도록 예외 처리
        return result_text, "모델 응답이 JSON 형식이 아니어서 전략을 분리하지 못했습니다."
    except Exception as e:
        return f"오류 발생: {str(e)}", "오류로 인해 전략을 불러올 수 없습니다."

    except Exception as e:
        return f"오류 발생: {str(e)}", "오류로 인해 전략을 불러올 수 없습니다."

# 4. 카드 UI 생성 함수 (HTML/CSS/JS)
def create_card_html(title, article, strategy, theme_color):
    html_content = f"""
    <div class="flip-card" onclick="this.querySelector('.flip-card-inner').classList.toggle('flipped');">
        <div class="flip-card-inner">
            <!-- 앞면: 기사 내용 -->
            <div class="flip-card-front" style="border-top: 5px solid {theme_color};">
                <h3 style="color: {theme_color}; margin-top:0;">{title} (기사)</h3>
                <p style="font-size: 0.9em; color: gray;">클릭하면 '프레이밍 전략'을 확인 할 수 있습니다.</p>
                <hr>
                <div style="white-space: pre-wrap; font-size: 1rem; line-height: 1.6;">{article}</div>
            </div>
            <!-- 뒷면: 전략 설명 -->
            <div class="flip-card-back" style="border-top: 5px solid {theme_color};">
                <h3 style="color: {theme_color}; margin-top:0;">{title} (전략)</h3>
                <p style="font-size: 0.9em; color: gray;">클릭하면 다시 기사를 확인 할 수 있습니다.</p>
                <hr>
                <div style="white-space: pre-wrap; font-size: 1rem; line-height: 1.6; color: #333;">{strategy}</div>
            </div>
        </div>
    </div>
    """
    return html_content

# 5. 메인 처리 함수
def process_article(user_article):
    if not user_article.strip():
        return "기사를 입력해주세요.", ""
    
    # 두 모델(진보, 보수)의 응답을 각각 받아옵니다 (동기 처리)
    prog_article, prog_strategy = generate_framing(user_article, is_progressive=True)
    cons_article, cons_strategy = generate_framing(user_article, is_progressive=False)
    
    # HTML 생성 (파란색: 진보, 빨간색: 보수 - 일반적인 정치 색상 관례 적용)
    prog_html = create_card_html("진보적 프레이밍 (Progressive)", prog_article, prog_strategy, "#1E88E5")
    cons_html = create_card_html("보수적 프레이밍 (Conservative)", cons_article, cons_strategy, "#E53935")
    
    return prog_html, cons_html

# 6. Gradio UI 설정 (CSS 포함)
custom_css = """
/* 플립 카드 래퍼 */
.flip-card {
  background-color: transparent;
  width: 100%;
  height: 600px;
  perspective: 1000px; /* 3D 효과를 위한 원근감 */
  cursor: pointer;
}

/* 카드 내부 컨테이너 (앞면과 뒷면을 포함하며 회전 애니메이션 담당) */
.flip-card-inner {
  position: relative;
  width: 100%;
  height: 100%;
  text-align: left;
  transition: transform 0.6s;
  transform-style: preserve-3d;
}

/* JS로 'flipped' 클래스가 추가되면 180도 회전 */
.flip-card-inner.flipped {
  transform: rotateY(180deg);
}

/* 앞면, 뒷면 공통 스타일 */
.flip-card-front, .flip-card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  backface-visibility: hidden; /* 뒷면일 때 숨김 처리 */
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  overflow-y: auto; /* 내용이 길면 스크롤 */
  box-sizing: border-box;
}

/* 앞면 스타일 */
.flip-card-front {
  background-color: #ffffff;
  color: black;
}

/* 뒷면 스타일 */
.flip-card-back {
  background-color: #f0f4f8;
  color: black;
  transform: rotateY(180deg); /* 기본적으로 뒤집혀 있도록 설정 */
}
"""

with gr.Blocks(css=custom_css, title="Habermas Framing Machine") as demo:
    gr.Markdown("# 🐘 코끼리는 생각하지마 & 하버마스 머신")
    gr.Markdown("동일한 정책 이슈가 진보와 보수 진영에서 어떻게 프레이밍(Framing) 되는지 비교해보세요. 결과 카드를 **클릭**하면 뒷면에서 어떤 언어 전략이 사용되었는지 확인할 수 있습니다.")
    
    with gr.Row():
        user_input = gr.Textbox(
            lines=6, 
            placeholder="여기에 리프레이밍할 원본 기사나 정책 이슈를 입력하세요...",
            label="원본 이슈/기사 입력"
        )
    
    submit_btn = gr.Button("분석 및 리프레이밍 시작", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            prog_output = gr.HTML(label="진보 관점")
        with gr.Column(scale=1):
            cons_output = gr.HTML(label="보수 관점")
            
    submit_btn.click(
        fn=process_article,
        inputs=user_input,
        outputs=[prog_output, cons_output]
    )

if __name__ == "__main__":
    demo.launch()
