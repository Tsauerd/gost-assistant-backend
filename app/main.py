from __future__ import annotations

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text as sql_text
from typing import Optional

# Твои существующие импорты
from .db import SessionLocal
from .rag import search_chunks
from .llm import call_llm

# 1. ДОБАВЛЯЕМ ИМПОРТ БОТА
from .telegram_bot import setup_telegram

app = FastAPI(title="GOST Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # потом лучше ограничить доменом фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    task_type: Optional[str] = "norm"
    client_id: Optional[str] = None

class RateRequest(BaseModel):
    request_id: int
    rating: int = Field(ge=1, le=5)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "gost-assistant-backend"}

def resolve_model_name(task_type: Optional[str]) -> str:
    tt = (task_type or "").lower()
    if tt in ("complaint", "claim", "letter", "legal", "complaint_letter"):
        return "gpt-4o"
    return "gpt-4o-mini"

@app.post("/chat")
async def chat_endpoint(body: ChatRequest, request: Request):
    user_query = (body.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query")

    task_type = (body.task_type or "norm").lower()
    client_id = body.client_id
    user_agent = request.headers.get("user-agent", "")

    request_id = Nonea
    db_error = None
    try:
        with SessionLocal() as db:
            ins = sql_text("""
                INSERT INTO requests (query_text, task_type, client_id, user_agent, status)
                VALUES (:q, :tt, :cid, :ua, 'pending')
                RETURNING id
            """)
            request_id = db.execute(ins, {
                "q": user_query,
                "tt": task_type,
                "cid": client_id,
                "ua": user_agent,
            }).scalar()
            db.commit()
    except Exception as e:
        db_error = str(e)
        print(f"[db] insert pending failed: {db_error}")

    chunks = search_chunks(user_query, top_k=6)
    context_used = [c.text for c in chunks]
    context_text = "\n\n".join(context_used) if context_used else "Нет конкретной информации в базе знаний."

    full_prompt = f"""
Ты — профессиональный технический эксперт по стандартам ГОСТ/СНиП.

Правила:
1) Отвечай строго на основе контекста ниже. Если данных нет — так и скажи.
2) Если в контексте есть таблицы/формулы — учитывай их.
3) Пиши структурировано: Нормативное обоснование / Анализ / Вывод.

Контекст:
{context_text}

Вопрос:
{user_query}
""".strip()

    model_name = resolve_model_name(task_type)
    answer, usage = call_llm(full_prompt, model=model_name)

    tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
    tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

    if "mini" in model_name:
        in_price = 0.15
        out_price = 0.60
    else:
        in_price = 2.50
        out_price = 10.00
    cost = (tokens_in * in_price + tokens_out * out_price) / 1_000_000

    if request_id is not None:
        try:
            with SessionLocal() as db:
                upd = sql_text("""
                    UPDATE requests
                    SET
                        status = 'success',
                        model_used = :m,
                        tokens_in = :ti,
                        tokens_out = :to,
                        cost_usd = :c,
                        answer_text = :ans
                    WHERE id = :id
                """)
                db.execute(upd, {
                    "m": model_name,
                    "ti": tokens_in,
                    "to": tokens_out,
                    "c": cost,
                    "ans": answer,
                    "id": request_id,
                })
                db.commit()
        except Exception as e:
            print(f"[db] update success failed: {e}")

    return {
        "answer": answer,
        "model_used": model_name,
        "context_used": context_used,
        "request_id": request_id,
        "db_error": db_error,
    }

@app.post("/rate")
async def rate_endpoint(body: RateRequest):
    with SessionLocal() as db:
        exists = db.execute(
            sql_text("SELECT 1 FROM requests WHERE id = :id"),
            {"id": body.request_id},
        ).scalar()

        if not exists:
            raise HTTPException(status_code=404, detail="request_id not found")

        try:
            db.execute(
                sql_text("""
                    UPDATE requests
                    SET rating = :r, rating_created_at = now()
                    WHERE id = :id
                """),
                {"r": body.rating, "id": body.request_id},
            )
            db.commit()
        except Exception:
            db.rollback()
            db.execute(
                sql_text("""
                    UPDATE requests
                    SET rating = :r
                    WHERE id = :id
                """),
                {"r": body.rating, "id": body.request_id},
            )
            db.commit()

    return {"ok": True}

# 2. ИНИЦИАЛИЗИРУЕМ ТЕЛЕГРАМ БОТА В САМОМ КОНЦЕ
setup_telegram(app)