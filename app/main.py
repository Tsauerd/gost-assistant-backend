from __future__ import annotations

from typing import Optional, Any, Dict, List
import traceback
from decimal import Decimal
from uuid import UUID

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from sqlalchemy import text as sql_text

from .db import SessionLocal, ensure_schema
from .rag import search_chunks
from .llm import call_llm

app = FastAPI(title="GOST Assistant Backend")

# Важно:
# allow_credentials=True + allow_origins=["*"] часто ломает CORS в браузере.
# Т.к. куки/авторизация тебе тут не нужны — ставим allow_credentials=False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    task_type: Optional[str] = "norm"
    client_id: Optional[str] = None

class RateRequest(BaseModel):
    # Делаем str, чтобы поддержать и int, и UUID, и т.п.
    request_id: str
    rating: int = Field(ge=1, le=5)

@app.on_event("startup")
def _startup() -> None:
    # создаст таблицу requests если её нет
    ensure_schema()

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "gost-assistant-backend"}

def resolve_model_name(task_type: Optional[str]) -> str:
    tt = (task_type or "").lower()
    if tt in ("complaint", "claim", "letter", "legal", "complaint_letter"):
        return "gpt-4o"
    return "gpt-4o-mini"

def _coerce_request_id(value: Any) -> Any:
    """
    Возвращаем request_id в сериализуемом виде.
    - int -> int
    - uuid.UUID -> str
    - Decimal -> float
    - None -> None
    """
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    return value

@app.post("/chat")
async def chat_endpoint(request_body: ChatRequest, request: Request):
    user_query = (request_body.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query")

    task_type = (request_body.task_type or "norm").lower()
    client_id = request_body.client_id
    user_agent = request.headers.get("user-agent", "")
    model_name = resolve_model_name(task_type)

    # 1) СНАЧАЛА создаём запись в БД -> получаем request_id
    request_id = None
    db_error = None

    try:
        with SessionLocal() as db:
            insert_q = sql_text("""
                INSERT INTO requests (query_text, task_type, model_used, client_id, user_agent, status)
                VALUES (:q, :tt, :m, :cid, :ua, 'pending')
                RETURNING id
            """)
            res = db.execute(insert_q, {
                "q": user_query,
                "tt": task_type,
                "m": model_name,
                "cid": client_id,
                "ua": user_agent,
            })
            request_id = res.scalar()
            db.commit()
    except Exception as e:
        db_error = f"{type(e).__name__}: {e}"
        print("[DB] insert failed:", db_error)

    # 2) RAG
    try:
        chunks = search_chunks(user_query, top_k=6)
    except Exception as e:
        chunks = []
        print("[RAG] search_chunks failed:", e)

    context_text = "\n\n".join([f"Отрывок:\n{row.text}" for row in chunks]) or "Нет конкретной информации в базе знаний."

    # 3) Prompt
    full_prompt = f"""
Ты — профессиональный технический эксперт по стандартам ГОСТ/СП/СНиП.

Правила:
1) Отвечай строго по контексту. Если данных нет — так и скажи.
2) Если есть таблицы/формулы — сохраняй структуру.
3) Числа и условия перепроверяй.

Контекст:
{context_text}

Запрос пользователя:
{user_query}
""".strip()

    # 4) LLM
    answer = "Нет ответа."
    usage = None
    llm_error = None

    try:
        answer, usage = call_llm(full_prompt, model=model_name)
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        print("[LLM] call failed:", llm_error)
        answer = "Ошибка генерации ответа. Попробуйте позже."

    tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    tokens_out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

    # Условный расчёт стоимости
    if "mini" in model_name:
        in_price = 0.15
        out_price = 0.60
    else:
        in_price = 2.50
        out_price = 10.00
    cost = (tokens_in * in_price + tokens_out * out_price) / 1_000_000

    # 5) Обновляем запись в БД
    if request_id is not None:
        try:
            with SessionLocal() as db:
                upd_q = sql_text("""
                    UPDATE requests
                    SET status = :st,
                        answer_text = :ans,
                        tokens_in = :ti,
                        tokens_out = :to,
                        cost_usd = :c,
                        model_used = :m,
                        error_text = :err
                    WHERE id = :id
                """)
                db.execute(upd_q, {
                    "st": "success" if llm_error is None else "error",
                    "ans": answer,
                    "ti": tokens_in,
                    "to": tokens_out,
                    "c": cost,
                    "m": model_name,
                    "err": llm_error,
                    "id": request_id,
                })
                db.commit()
        except Exception as e:
            print("[DB] update failed:", e)

    payload = {
        "answer": answer,
        "model_used": model_name,
        "context_used": [row.text for row in chunks],
        "request_id": _coerce_request_id(request_id),  # <-- ключевой фикс
        "db_error": db_error,                          # <-- чтобы видеть в F12 почему request_id null
    }

    # Гарантируем корректную сериализацию (UUID/Decimal и т.п.)
    return JSONResponse(content=jsonable_encoder(payload))

@app.post("/rate")
async def rate_endpoint(body: RateRequest):
    # Пробуем привести к int (если у тебя serial id)
    rid: Any = body.request_id
    if isinstance(rid, str) and rid.isdigit():
        rid = int(rid)

    try:
        with SessionLocal() as db:
            res = db.execute(
                sql_text("""
                    UPDATE requests
                    SET rating = :r, rating_created_at = now()
                    WHERE id = :id
                """),
                {"r": body.rating, "id": rid}
            )
            db.commit()

            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="request_id not found")

        return {"ok": True}

    except HTTPException:
        raise
    except Exception as e:
        print("[DB] rate failed:", e)
        raise HTTPException(status_code=500, detail="Internal DB error")
