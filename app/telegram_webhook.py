from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.concurrency import run_in_threadpool

from .chat_service import run_chat_sync
from .telegram_bot import HELP_TEXT, latex_to_telegram, split_telegram

TELEGRAM_API_BASE = "https://api.telegram.org"

# user_id -> mode
_user_mode: Dict[int, str] = {}


async def _tg_call(token: str, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TELEGRAM_API_BASE}/bot{token}/{method}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


async def _tg_send_message(token: str, chat_id: int, text: str) -> None:
    # Telegram лимит 4096; мы уже режем до 3500, но на всякий случай:
    await _tg_call(token, "sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    })


async def _tg_set_webhook(token: str, webhook_url: str) -> None:
    # Можно использовать secret_token, но у нас секрет в URL — достаточно.
    await _tg_call(token, "setWebhook", {
        "url": webhook_url,
        "drop_pending_updates": True,
    })


def setup_telegram_webhook(app: FastAPI) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    secret_env = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    public_base = os.getenv("PUBLIC_BASE_URL", "").strip()  # например: https://gost-assistant-backend.onrender.com

    # endpoint: /telegram/webhook/{secret}
    @app.get("/telegram/webhook/{secret}")
    async def telegram_webhook_probe(secret: str):
        # Удобно для дебага: видишь 200/403/503 вместо непонятного 404
        if not secret_env:
            raise HTTPException(status_code=503, detail="TELEGRAM_WEBHOOK_SECRET is not set")
        if secret != secret_env:
            raise HTTPException(status_code=403, detail="Forbidden")
        return {"ok": True, "hint": "Use POST for Telegram updates"}

    @app.post("/telegram/webhook/{secret}")
    async def telegram_webhook(secret: str, request: Request):
        if not secret_env:
            raise HTTPException(status_code=503, detail="TELEGRAM_WEBHOOK_SECRET is not set")
        if secret != secret_env:
            raise HTTPException(status_code=403, detail="Forbidden")

        update = await request.json()

        # Telegram хочет быстрый 200 OK — обработаем в фоне
        asyncio.create_task(_handle_update(token, update))
        return {"ok": True}

    @app.on_event("startup")
    async def _startup():
        # Авто-регистрация webhook (если всё задано)
        if not token or not secret_env or not public_base:
            print("[telegram] webhook auto-setup skipped (need TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, PUBLIC_BASE_URL)")
            return

        webhook_url = public_base.rstrip("/") + f"/telegram/webhook/{secret_env}"
        try:
            await _tg_set_webhook(token, webhook_url)
            print(f"[telegram] webhook set: {webhook_url}")
        except Exception as e:
            # Не роняем весь сервис из-за телеги
            print(f"[telegram] setWebhook failed: {e}")


async def _handle_update(token: str, update: Dict[str, Any]) -> None:
    if not token:
        # токен не задан — ответить не сможем
        return

    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return

    chat = msg.get("chat") or {}
    sender = msg.get("from") or {}

    chat_id = chat.get("id")
    user_id = sender.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id or not user_id:
        return

    # --- команды ---
    if text.startswith("/start"):
        _user_mode[user_id] = "norm"
        await _tg_send_message(token, chat_id, "GOST_AI в Telegram.\n\n" + HELP_TEXT)
        return

    if text.startswith("/norm"):
        _user_mode[user_id] = "norm"
        await _tg_send_message(token, chat_id, "Режим переключен: norm")
        return

    if text.startswith("/proc"):
        _user_mode[user_id] = "procedure"
        await _tg_send_message(token, chat_id, "Режим переключен: procedure")
        return

    if text.startswith("/calc"):
        _user_mode[user_id] = "calculation"
        await _tg_send_message(token, chat_id, "Режим переключен: calculation")
        return

    if text.startswith("/claim"):
        _user_mode[user_id] = "complaint"
        await _tg_send_message(token, chat_id, "Режим переключен: complaint")
        return

    if not text:
        return

    task_type = _user_mode.get(user_id, "norm")
    client_id = f"tg:{user_id}"

    # маленький статус
    await _tg_send_message(token, chat_id, "Ищу в ГОСТах…")

    try:
        result = await run_in_threadpool(
            run_chat_sync,
            text,
            task_type,
            client_id,
            "telegram-webhook",
        )

        raw_answer = (result or {}).get("answer") or "Нет ответа."
        answer = latex_to_telegram(raw_answer)

        for part in split_telegram(answer, max_len=3500):
            await _tg_send_message(token, chat_id, part)

    except Exception as e:
        await _tg_send_message(token, chat_id, f"Ошибка: {e}")
