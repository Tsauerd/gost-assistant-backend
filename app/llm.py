# app/llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from app.guard.post_guard import post_guard  # Импортируем нашу охрану

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def call_llm(prompt: str, used_sources: list = None, model: str = "gpt-4o-mini"):
    """
    Умная обертка:
    1. Генерирует ответ.
    2. Прогоняет через post_guard.
    3. Если есть ошибки -> просит LLM исправить (1 попытка ретрая).
    """
    if used_sources is None:
        used_sources = []

    messages = [
        {
            "role": "system",
            "content": "Ты ассистент по ГОСТам и строительным нормативам. Будь точен в цифрах."
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # --- Шаг 1: Первая генерация ---
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    answer = response.choices[0].message.content
    usage = response.usage
    
    # --- Шаг 2: Проверка Guard ---
    warnings = post_guard(prompt, answer, used_sources)

    # --- Шаг 3: Если есть варнинги, делаем Retry (самокоррекцию) ---
    if warnings:
        print(f"DEBUG: Guard сработал! Ошибки: {warnings}")
        
        # Добавляем контекст ошибки в диалог
        correction_prompt = (
            f"В твоем ответе найдены фактические неточности:\n"
            f"{'; '.join(warnings)}\n\n"
            "Пожалуйста, перепиши ответ, исправив эти ошибки и ссылаясь на верные пункты ГОСТ."
        )
        
        # Обновляем историю сообщений
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": correction_prompt})

        # Повторный вызов
        response_retry = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        
        # Обновляем ответ и usage (суммируем токены, если нужно, или берем новые)
        answer = response_retry.choices[0].message.content
        usage = response_retry.usage # Тут будут токены за второй запрос

    return answer, usage