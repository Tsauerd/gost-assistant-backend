# app/llm.py
import os
from typing import List, Optional, Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from .guard.post_guard import post_guard  # ВАЖНО: относительный импорт

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_llm(prompt: str, used_sources: Optional[List[Dict[str, Any]]] = None, model: str = "gpt-4o-mini"):
    """
    1) Генерация
    2) post_guard
    3) Если есть warnings -> 1 retry с исправлением (без смены модели)
    """
    used_sources = used_sources or []

    messages = [
        {"role": "system", "content": "Ты ассистент по ГОСТ/СП. Будь точен в числах и формулировках."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    answer = response.choices[0].message.content
    usage = response.usage

    warnings = post_guard(prompt, answer, used_sources)

    if warnings:
        correction_prompt = (
            "В твоем ответе найдены потенциальные фактические неточности:\n"
            f"- " + "\n- ".join(warnings) + "\n\n"
            "Перепиши ответ так, чтобы он строго соответствовал контексту и источникам. "
            "Если точных данных нет — так и скажи. Добавь ссылки на пункты/разделы в формате "
            "[СТАНДАРТ | Раздел X | Пункт Y | chunk Z]."
        )

        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": correction_prompt})

        response_retry = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        answer = response_retry.choices[0].message.content
        usage = response_retry.usage

    return answer, usage
