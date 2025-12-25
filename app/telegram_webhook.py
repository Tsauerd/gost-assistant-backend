from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from .telegram_bot import handle_telegram_update


TELEGRAM_API_BASE = "https://api.telegram.org"


async def _tg_set_webhook(token: str, webhook_url: str, secret: str) -> Dict[str, Any]:
    url = f"{TELEGRAM_API_BASE}/bot{token}/setWebhook"
    payload = {
        "url": webhook_url,
        "secret_token": secret,
        "drop_pending_updates": True,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def setup_telegram_webhook(app: FastAPI) -> None:
    """
    1) Регистрирует endpoint: POST /telegram/webhook
    2) На startup вызывает Telegram setWebhook(url=PUBLIC_BASE_URL + /telegram/webhook)
    3) Проверяет секрет через заголовок X-Telegram-Bot-Api-Secret-Token
    """

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET")
    public_base = os.getenv("PUBLIC_BASE_URL")  # например https://gost-assistant-backend.onrender.com

    # Если переменных нет — не включаем телеграм, но сервис НЕ падает
    if not token or not secret or not public_base:
        print(
            "[telegram] webhook disabled. Need env vars: "
            "TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, PUBLIC_BASE_URL"
        )
        return

    webhook_url = public_base.rstrip("/") + "/telegram/webhook"

    @app.on_event("startup")
    async def _startup_register_webhook():
        try:
            res = await _tg_set_webhook(token, webhook_url, secret)
            print(f"[telegram] setWebhook: {res}")
        except Exception as e:
            # важно: не роняем весь FastAPI из-за Telegram
            print(f"[telegram] setWebhook failed: {e}")

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request):
        got = request.headers.get("x-telegram-bot-api-secret-token")
        if got != secret:
            raise HTTPException(status_code=401, detail="Bad telegram secret token")

        update = await request.json()

        # Telegram надо отвечать быстро -> обработку в фоне
        asyncio.create_task(handle_telegram_update(token, update))

        return PlainTextResponse("OK")
