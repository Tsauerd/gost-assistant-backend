import os
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")

TELEGRAM_API_BASE = "https://api.telegram.org"


async def _tg_send_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    url = f"{TELEGRAM_API_BASE}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()


def setup_telegram_webhook(app: FastAPI) -> None:
    """
    Регистрирует endpoint:
    POST /telegram/webhook/{secret}
    """

    if not TELEGRAM_WEBHOOK_SECRET:
        # Можно не падать, просто не включать телеграм.
        print("[telegram] TELEGRAM_WEBHOOK_SECRET not set -> webhook disabled")
        return

    @app.post("/telegram/webhook/{secret}")
    async def telegram_webhook(secret: str, request: Request):
        # 1) защита секретом в URL
        if secret != TELEGRAM_WEBHOOK_SECRET:
            raise HTTPException(status_code=403, detail="Forbidden")

        update = await request.json()

        # 2) извлекаем message.text
        message = update.get("message") or update.get("edited_message")
        if not message:
            return {"ok": True}

        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if not chat_id:
            return {"ok": True}

        text = (message.get("text") or "").strip()
        if not text:
            return {"ok": True}

        # 3) режимы можно добавить позже; пока default
        task_type = "norm"
        client_id = f"tg:{chat_id}"

        # 4) дергаем твой же backend /chat ВНУТРИ (локально)
        #    ВАЖНО: мы вызываем не через интернет, а напрямую через ASGI не будем.
        #    Поэтому проще — сходить HTTP в этот же сервис по публичному URL.
        #    Render выдержит, но если хочешь "без self-http" — скажи, сделаем прямой вызов функции.
        backend_url = os.getenv("BACKEND_URL", "https://gost-assistant-backend.onrender.com")

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f"{backend_url}/chat",
                    json={"query": text, "task_type": task_type, "client_id": client_id},
                )
                r.raise_for_status()
                data = r.json()
                answer = data.get("answer") or "Нет ответа."
        except Exception as e:
            answer = f"Ошибка при обращении к /chat: {e}"

        # 5) ответ в Telegram
        # Telegram лимит ~4096 на сообщение. Режем безопасно.
        max_len = 3500
        parts = [answer[i:i+max_len] for i in range(0, len(answer), max_len)] or ["(пустой ответ)"]

        for part in parts:
            await _tg_send_message(chat_id=chat_id, text=part)

        return {"ok": True}
