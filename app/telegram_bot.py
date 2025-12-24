from __future__ import annotations

import os
import re
import logging
from typing import Dict, Optional, List

import httpx
from fastapi import FastAPI, Request, HTTPException

# Можно оставить, если используешь .env локально
from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("telegram")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").strip()  # например: https://gost-assistant-backend.onrender.com
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()

# Куда дергать /chat (можно оставить свой сервис)
BACKEND_URL = os.environ.get("BACKEND_URL", "").strip() or PUBLIC_URL

# Простое состояние режима в памяти
USER_MODE: Dict[int, str] = {}  # chat_id -> task_type

# ---- LaTeX -> Telegram text (как у тебя) ----
SUB = str.maketrans("0123456789+-=()n", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙ")
SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")

LATEX_MAP = {
    r"\times": "·",
    r"\cdot": "·",
    r"\%": "%",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\le": "≤",
    r"\ge": "≥",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\pm": "±",
}

def _strip_wrapped_commands(text: str) -> str:
    for _ in range(3):
        text = re.sub(r"\\(text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\2", text)
    return text

def latex_to_telegram(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"\$\$([\s\S]*?)\$\$", r"\1", text)
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text, flags=re.S)
    text = re.sub(r"\$([^\n$]+)\$", r"\1", text)

    text = _strip_wrapped_commands(text)

    for k, v in LATEX_MAP.items():
        text = text.replace(k, v)

    def frac(m):
        a = m.group(1).strip()
        b = m.group(2).strip()
        return f"({a})/({b})"
    for _ in range(2):
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", frac, text)

    def sub1(m):
        return m.group(1) + m.group(2).translate(SUB)
    text = re.sub(r"([A-Za-zА-Яа-я])_([0-9]+)", sub1, text)

    def sub2(m):
        return m.group(1) + m.group(2).translate(SUB)
    text = re.sub(r"([A-Za-zА-Яа-я])_\{([0-9+\-()n=]+)\}", sub2, text)

    def sup1(m):
        return m.group(1) + m.group(2).translate(SUP)
    text = re.sub(r"([A-Za-zА-Яа-я0-9])\^([0-9]+)", sup1, text)

    def sup2(m):
        return m.group(1) + m.group(2).translate(SUP)
    text = re.sub(r"([A-Za-zА-Яа-я0-9])\^\{([0-9+\-()n=]+)\}", sup2, text)

    text = re.sub(r"\\(left|right)\b", "", text)
    text = text.replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ")
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def split_telegram(text: str, max_len: int = 3500) -> List[str]:
    if len(text) <= max_len:
        return [text]
    parts = []
    buf = text
    while len(buf) > max_len:
        cut = buf.rfind("\n", 0, max_len)
        if cut < 800:
            cut = max_len
        parts.append(buf[:cut].rstrip())
        buf = buf[cut:].lstrip()
    if buf:
        parts.append(buf)
    return parts

HELP_TEXT = (
    "Напиши вопрос по ГОСТам.\n"
    "Команды:\n"
    "/norm — режим нормативки\n"
    "/proc — процедура испытаний\n"
    "/calc — расчёт\n"
    "/claim — претензия\n"
)

def _webhook_url() -> str:
    base = (PUBLIC_URL or "").rstrip("/")
    return f"{base}/telegram/webhook"

async def _tg_api(method: str, payload: dict) -> dict:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

async def _send_message(chat_id: int, text: str) -> None:
    for part in split_telegram(text, max_len=3500):
        await _tg_api("sendMessage", {
            "chat_id": chat_id,
            "text": part
        })

async def _set_webhook() -> None:
    if not BOT_TOKEN or not PUBLIC_URL:
        log.warning("Telegram webhook not configured: TELEGRAM_BOT_TOKEN or PUBLIC_URL missing")
        return

    payload = {
        "url": _webhook_url(),
        "drop_pending_updates": True,
    }
    if WEBHOOK_SECRET:
        payload["secret_token"] = WEBHOOK_SECRET

    res = await _tg_api("setWebhook", payload)
    log.info("setWebhook: %s", res)

async def _call_backend_chat(query: str, task_type: str, client_id: str) -> str:
    # Дергаем твой /chat
    url = (BACKEND_URL or "").rstrip("/") + "/chat"
    timeout = httpx.Timeout(120.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json={
            "query": query,
            "task_type": task_type,
            "client_id": client_id,
        })
        r.raise_for_status()
        data = r.json()
        return data.get("answer") or "Нет ответа."

def setup_telegram(app: FastAPI) -> None:
    # 1) Startup: поставить webhook
    @app.on_event("startup")
    async def _startup():
        await _set_webhook()

    # 2) Webhook endpoint
    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request):
        # Если используешь секрет — проверяем
        if WEBHOOK_SECRET:
            got = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if got != WEBHOOK_SECRET:
                raise HTTPException(status_code=403, detail="Bad secret token")

        update = await request.json()

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

        # Команды режимов
        if text in ("/start", "/help"):
            USER_MODE[chat_id] = "norm"
            await _send_message(chat_id, "GOST_AI в Telegram.\n\n" + HELP_TEXT)
            return {"ok": True}

        if text == "/norm":
            USER_MODE[chat_id] = "norm"
            await _send_message(chat_id, "Режим переключен: norm")
            return {"ok": True}

        if text == "/proc":
            USER_MODE[chat_id] = "procedure"
            await _send_message(chat_id, "Режим переключен: procedure")
            return {"ok": True}

        if text == "/calc":
            USER_MODE[chat_id] = "calculation"
            await _send_message(chat_id, "Режим переключен: calculation")
            return {"ok": True}

        if text == "/claim":
            USER_MODE[chat_id] = "complaint"
            await _send_message(chat_id, "Режим переключен: complaint")
            return {"ok": True}

        task_type = USER_MODE.get(chat_id, "norm")
        client_id = f"tg:{chat_id}"

        try:
            raw_answer = await _call_backend_chat(text, task_type, client_id)
            answer = latex_to_telegram(raw_answer)
            await _send_message(chat_id, answer)
        except Exception as e:
            await _send_message(chat_id, f"Ошибка: {e}")

        return {"ok": True}
