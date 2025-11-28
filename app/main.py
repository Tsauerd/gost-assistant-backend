# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel          # <--- ВАЖНО: этот импорт вернули
from sqlalchemy import text as sql_text
from typing import Optional

from .db import SessionLocal
from .rag import search_chunks
from .llm import call_llm

app = FastAPI(title="GOST Assistant Backend")

# На время разработки открываем CORS для всех источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # потом сузим до домена Netlify/Render
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    task_type: Optional[str] = "question"   # question, summary, extract и т.п.
    model_version: Optional[str] = "mini"   # "mini" или "pro"


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def resolve_model_name(model_version: str) -> str:
    mv = (model_version or "mini").lower()
    if mv == "pro":
        return "gpt-4.1"
    return "gpt-4.1-mini"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.query

    # 1. RAG: ищем релевантные куски
    chunks = search_chunks(user_query, top_k=3)

    context_text = "\n\n".join([f"Отрывок: {row.text}" for row in chunks])
    if not context_text.strip():
        context_text = "Нет конкретной информации в базе знаний."

    # 2. Формируем промпт
    full_prompt = f"""
Ты — экспертный ассистент по ГОСТам и нормативным документам.

Правила:
1. Отвечай строго на основе приведённого ниже контекста (выдержек из ГОСТов и других документов).
2. Если информации в контексте недостаточно, честно напиши, что точной нормы нет, и можешь дать аккуратный комментарий, но без выдуманных пунктов ГОСТ.
3. Если в контексте есть таблицы в формате markdown (строки, начинающиеся с '|', и блоки с 'Таблица ... (markdown):'),
   обязательно сохраняй их структуру и отображай как markdown-таблицы в ответе.
4. Если таблиц нет, просто отвечай текстом.
5. Всегда указывай, на какие пункты/разделы ГОСТ ты опираешься (номер ГОСТ, пункт, страница, если есть).

Контекст:
{context_text}

Вопрос пользователя:
{user_query}
"""

    # 3. Выбираем модель (по переключателю mini/pro)
    model_name = resolve_model_name(request.model_version)

    answer, usage = call_llm(full_prompt, model=model_name)

    # 4. Токены и стоимость
    tokens_in = 0
    tokens_out = 0

    if usage:
        if hasattr(usage, "prompt_tokens"):
            tokens_in = usage.prompt_tokens
        if hasattr(usage, "completion_tokens"):
            tokens_out = usage.completion_tokens
        if hasattr(usage, "input_tokens"):
            tokens_in = usage.input_tokens
        if hasattr(usage, "output_tokens"):
            tokens_out = usage.output_tokens

    # Тарифы
    if model_name == "gpt-4.1-mini":
        in_price = 0.40
        out_price = 1.60
    else:  # gpt-4.1
        in_price = 2.00
        out_price = 10.00

    cost = (tokens_in * in_price + tokens_out * out_price) / 1_000_000

    # 5. Логируем запрос
    log_query = sql_text("""
        INSERT INTO requests (query_text, task_type, model_used, tokens_in, tokens_out, cost_usd, status)
        VALUES (:q, :tt, :m, :ti, :to, :c, 'success')
    """)

    try:
        with SessionLocal() as db:
            db.execute(log_query, {
                "q": user_query,
                "tt": request.task_type,
                "m": model_name,
                "ti": tokens_in,
                "to": tokens_out,
                "c": cost
            })
            db.commit()
    except Exception as e:
        print(f"Ошибка логирования: {e}")

    return {
        "answer": answer,
        "model_used": model_name,
        "context_used": [row.text for row in chunks]
    }
