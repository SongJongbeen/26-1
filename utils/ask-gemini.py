import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL = "google/gemini-3.1-pro-preview"

async def fetch_response(prompt: str) -> tuple[str, str]:
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ 오류 발생: {str(e)}"

async def main():
    user_input = input("Q: ")
    result = await fetch_response(user_input)
    print(result)
    with open(f"utils/{datetime.now().strftime('%Y%m%d%H%M%S')}.md", "w", encoding="utf-8") as f:
        f.write(result)

if __name__ == "__main__":
    asyncio.run(main())
