from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text as sql_text
from typing import Optional
import logging

from .db import SessionLocal
from .rag import search_chunks
from .llm import call_llm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gost-backend")

app = FastAPI(title="GOST Assistant Backend")

# ВАЖНО: если origins = ["*"], лучше allow_credentials=False
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

    request_id: Optional[int] = None
    db_err_msg: Optional[str] = None

    # 1) Сначала создаём запись pending → получаем request_id
    try:
        with SessionLocal() as db:
            insert_q = sql_text("""
                INSERT INTO requests (query_text, task_type, client_id, user_agent, status)
                VALUES (:q, :tt, :cid, :ua, 'pending')
                RETURNING id
            """)
            request_id = db.execute(insert_q, {
                "q": user_query,
                "tt": task_type,
                "cid": client_id,
                "ua": user_agent,
            }).scalar()
            db.commit()
    except Exception as e:
        db_err_msg = str(e)
        log.exception("DATABASE INSERT ERROR: %s", db_err_msg)

    # 2) RAG
    chunks = search_chunks(user_query, top_k=6)
    context_text = "\n\n".join([row.text for row in chunks]) or "Нет конкретной информации в базе знаний."

    # 3) Prompt
    full_prompt = f"""
Ты — профессиональный технический эксперт, специализирующийся на стандартах ГОСТ/СП/СНиП.

Правила:
1) Отвечай строго на основе контекста. Если данных нет — скажи прямо.
2) Если в контексте есть таблицы Markdown — учитывай их в приоритете.
3) Числа/условия проверяй внимательно.

Контекст:
{context_text}

Запрос:
{user_query}
""".strip()

    # 4) LLM
    model_name = resolve_model_name(task_type)
    answer, usage = call_llm(full_prompt, model=model_name)

    # 5) Пытаемся обновить запись в БД (не критично для ответа)
    if request_id is not None:
        try:
            tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
            tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

            # цены (твой расчёт)
            if "mini" in model_name:
                in_price = 0.15
                out_price = 0.60
            else:
                in_price = 2.50
                out_price = 10.00
            cost = (tokens_in * in_price + tokens_out * out_price) / 1_000_000

            with SessionLocal() as db:
                # пробуем "богатое" обновление
                try:
                    upd = sql_text("""
                        UPDATE requests
                        SET status='success',
                            model_used=:m,
                            tokens_in=:ti,
                            tokens_out=:to,
                            cost_usd=:c,
                            answer_text=:ans
                        WHERE id=:id
                    """)
                    db.execute(upd, {
                        "m": model_name,
                        "ti": tokens_in,
                        "to": tokens_out,
                        "c": cost,
                        "ans": answer,
                        "id": request_id
                    })
                except Exception:
                    # если каких-то колонок нет — хотя бы статус
                    upd_min = sql_text("""
                        UPDATE requests
                        SET status='success'
                        WHERE id=:id
                    """)
                    db.execute(upd_min, {"id": request_id})

                db.commit()
        except Exception as e:
            log.exception("DATABASE UPDATE ERROR: %s", str(e))

    return {
        "answer": answer,
        "model_used": model_name,
        "context_used": [row.text for row in chunks],
        "request_id": request_id,
        "db_error": db_err_msg,
    }

@app.post("/rate")
async def rate_endpoint(body: RateRequest):
    try:
        with SessionLocal() as db:
            upd = sql_text("""
                UPDATE requests
                SET rating = :r,
                    rating_created_at = now()
                WHERE id = :id
            """)
            res = db.execute(upd, {"r": body.rating, "id": body.request_id})
            db.commit()

            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="request_id not found")

        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        # Это как раз покажет "column rating does not exist" и т.п.
        log.exception("RATE DB ERROR: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
