from __future__ import annotations

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text as sql_text
from typing import Optional

from fastapi.concurrency import run_in_threadpool

from .db import SessionLocal
from .chat_service import run_chat_sync
from .telegram_webhook import setup_telegram_webhook


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


@app.post("/chat")
async def chat_endpoint(body: ChatRequest, request: Request):
    user_query = (body.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query")

    user_agent = request.headers.get("user-agent", "")
    client_id = body.client_id
    task_type = (body.task_type or "norm").lower()

    # запускаем синхронный пайплайн в threadpool, чтобы не блокировать event loop
    result = await run_in_threadpool(
        run_chat_sync,
        user_query,
        task_type,
        client_id,
        user_agent,
    )
    return result


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
                sql_text("UPDATE requests SET rating = :r WHERE id = :id"),
                {"r": body.rating, "id": body.request_id},
            )
            db.commit()

    return {"ok": True}


# ✅ подключаем Telegram webhook
setup_telegram_webhook(app)
