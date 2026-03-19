import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODELS = {
    "OpenAI (GPT 5.2)": "openai/gpt-5.2",
    "Anthropic (Claude 4.6 Sonnet)": "anthropic/claude-sonnet-4.6",
    "Google (Gemini 3.1 Pro)": "google/gemini-3.1-pro-preview",
    "DeepSeek (DeepSeek 3.2)": "deepseek/deepseek-v3.2",
}


async def fetch_response(model_name: str, model_id: str, prompt: str) -> tuple[str, str]:
    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return model_name, response.choices[0].message.content
    except Exception as e:
        return model_name, f"⚠️ 오류 발생: {str(e)}"


async def generate_all(prompt: str) -> list[tuple[str, str]]:
    tasks = [
        fetch_response(model_name, model_id, prompt)
        for model_name, model_id in MODELS.items()  # .items() to keep name + id together
    ]
    return await asyncio.gather(*tasks)  # all models called concurrently


async def main():
    user_input = input("Q: ")
    results = await generate_all(user_input)

    for model_name, result in results:
        print(f"\n{'='*40}\n[{model_name}]\n{result}")
        model_name = model_name.split(" ")[0]
        with open(f"{safe_name}.md", "w", encoding="utf-8") as f:
            f.write(result)


if __name__ == "__main__":
    asyncio.run(main())  # properly launches the async event loop
