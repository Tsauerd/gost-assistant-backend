from __future__ import annotations

from typing import Optional, Dict, Any, List
from sqlalchemy import text as sql_text

from .db import SessionLocal
from .rag import search_chunks
from .llm import call_llm


def resolve_model_name(task_type: Optional[str]) -> str:
    tt = (task_type or "").lower()
    if tt in ("complaint", "claim", "letter", "legal", "complaint_letter"):
        return "gpt-4o"
    return "gpt-4o-mini"


def run_chat_sync(
    user_query: str,
    task_type: str = "norm",
    client_id: Optional[str] = None,
    user_agent: str = "",
) -> Dict[str, Any]:
    """
    Синхронный пайплайн: лог в БД -> RAG -> LLM -> обновление лога.
    Его можно вызывать из /chat и из Telegram webhook.
    """
    user_query = (user_query or "").strip()
    if not user_query:
        return {
            "answer": "Пустой запрос.",
            "model_used": None,
            "context_used": [],
            "request_id": None,
            "db_error": None,
        }

    task_type = (task_type or "norm").lower()

    # 1) Создаём request сразу (чтобы request_id не был null)
    request_id = None
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

    # 2) RAG
    chunks = search_chunks(user_query, top_k=6)
    context_used: List[str] = [c.text for c in chunks]
    context_text = "\n\n".join(context_used) if context_used else "Нет конкретной информации в базе знаний."

    # 3) Prompt + LLM
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

    # Примерная стоимость (можешь убрать)
    if "mini" in model_name:
        in_price = 0.15
        out_price = 0.60
    else:
        in_price = 2.50
        out_price = 10.00
    cost = (tokens_in * in_price + tokens_out * out_price) / 1_000_000

    # 4) Обновляем запись (если request_id есть)
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
        except Exception:
            pass

    return {
        "answer": answer,
        "model_used": model_name,
        "context_used": context_used,
        "request_id": request_id,
        "db_error": db_error,
    }
