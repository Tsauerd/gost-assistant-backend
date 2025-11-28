# app/llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Подтягиваем .env (на всякий случай, как и в db.py)
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_llm(prompt: str, model: str = "gpt-4.1-mini"):
    """
    Универсальная обёртка вокруг Chat Completions.
    Возвращает (answer_text, usage).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Ты ассистент по ГОСТам и строительным нормативам."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,  # более детерминированные ответы
    )

    answer = response.choices[0].message.content
    usage = response.usage  # здесь есть prompt_tokens и completion_tokens

    return answer, usage
