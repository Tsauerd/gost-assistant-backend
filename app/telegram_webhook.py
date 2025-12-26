from __future__ import annotations

import os
import re
import asyncio
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.concurrency import run_in_threadpool

from .chat_service import run_chat_sync

TELEGRAM_API = "https://api.telegram.org"

# --- LaTeX -> Telegram plain unicode ---
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

HELP_TEXT = (
    "Напиши вопрос по ГОСТам.\n\n"
    "Режимы (можно командами или кнопками):\n"
    "/norm — нормативка\n"
    "/proc — процедура испытаний\n"
    "/calc — расчёт\n"
    "/claim — претензия\n"
)

INSTRUCTION_TEXT = (
    "📘 Инструкция:\n"
    "1) Нажми «🧭 Режимы» и выбери режим (или введи /norm /proc /calc /claim)\n"
    "2) Напиши вопрос текстом\n"
    "3) Бот вернёт ответ на основе базы\n"
)

# --- Reply keyboard (3 главные кнопки) ---
MAIN_MENU_KB: Dict[str, Any] = {
    "keyboard": [
        [{"text": "🏠 Главная"}, {"text": "📘 Инструкция"}, {"text": "🧭 Режимы"}],
    ],
    "resize_keyboard": True,
    "is_persistent": True,
    "one_time_keyboard": False,
}

# --- Подменю режимов ---
MODES_KB: Dict[str, Any] = {
    "keyboard": [
        [{"text": "📝 Нормативка"}, {"text": "🧪 Процедура"}],
        [{"text": "🧮 Расчёт"}, {"text": "⚠️ Претензия"}],
        [{"text": "⬅️ Назад"}],
    ],
    "resize_keyboard": True,
    "is_persistent": True,
    "one_time_keyboard": False,
}

_user_mode: Dict[int, str] = {}  # user_id -> task_type (RAM)


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

    text = re.sub(
        r"([A-Za-zА-Яа-я])_\{([0-9+\-()n=]+)\}",
        lambda m: m.group(1) + m.group(2).translate(SUB),
        text,
    )
    text = re.sub(
        r"([A-Za-zА-Яа-я])_([0-9]+)",
        lambda m: m.group(1) + m.group(2).translate(SUB),
        text,
    )

    text = re.sub(
        r"([A-Za-zА-Яа-я0-9])\^\{([0-9+\-()n=]+)\}",
        lambda m: m.group(1) + m.group(2).translate(SUP),
        text,
    )
    text = re.sub(
        r"([A-Za-zА-Яа-я0-9])\^([0-9]+)",
        lambda m: m.group(1) + m.group(2).translate(SUP),
        text,
    )

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


async def _tg_call(token: str, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TELEGRAM_API}/bot{token}/{method}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


async def _tg_send_message(
    token: str,
    chat_id: int,
    text: str,
    reply_markup: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    await _tg_call(token, "sendMessage", payload)


async def _set_webhook(token: str, public_base_url: str, secret: str) -> None:
    webhook_url = public_base_url.rstrip("/") + "/telegram/webhook"
    await _tg_call(
        token,
        "setWebhook",
        {
            "url": webhook_url,
            "secret_token": secret,
            "drop_pending_updates": True,
        },
    )


def setup_telegram_webhook(app: FastAPI) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    public_base = os.getenv("PUBLIC_BASE_URL", "").strip() or os.getenv("BACKEND_URL", "").strip()

    @app.get("/telegram/status")
    async def telegram_status():
        return {
            "token_set": bool(token),
            "secret_set": bool(secret),
            "public_base_set": bool(public_base),
            "webhook_path": "/telegram/webhook",
        }

    if not token or not secret or not public_base:
        print("[telegram] disabled: set TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, PUBLIC_BASE_URL(or BACKEND_URL)")
        return

    @app.on_event("startup")
    async def _startup_set_webhook():
        try:
            await _set_webhook(token, public_base, secret)
            print("[telegram] webhook set OK")
        except Exception as e:
            print(f"[telegram] setWebhook failed: {e}")

    @app.post("/telegram/webhook")
    async def telegram_webhook(request: Request):
        got = request.headers.get("x-telegram-bot-api-secret-token", "")
        if got != secret:
            raise HTTPException(status_code=401, detail="Unauthorized")

        update = await request.json()
        asyncio.create_task(_handle_update(token, update))
        return {"ok": True}


async def _handle_update(token: str, update: Dict[str, Any]) -> None:
    try:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return

        chat_id = (msg.get("chat") or {}).get("id")
        user_id = (msg.get("from") or {}).get("id")
        text = (msg.get("text") or "").strip()

        if not chat_id or not user_id:
            return

        # --------- КНОПКИ (Reply keyboard) ----------
        if text == "🏠 Главная":
            _user_mode[user_id] = "norm"
            await _tg_send_message(token, chat_id, "Главная.\n\n" + HELP_TEXT, reply_markup=MAIN_MENU_KB)
            return

        if text == "📘 Инструкция":
            await _tg_send_message(token, chat_id, INSTRUCTION_TEXT, reply_markup=MAIN_MENU_KB)
            return

        if text == "🧭 Режимы":
            await _tg_send_message(token, chat_id, "Выбери режим:", reply_markup=MODES_KB)
            return

        if text == "⬅️ Назад":
            await _tg_send_message(token, chat_id, "Главное меню:", reply_markup=MAIN_MENU_KB)
            return

        # Режимы кнопками
        if text == "📝 Нормативка":
            _user_mode[user_id] = "norm"
            await _tg_send_message(token, chat_id, "Режим: norm", reply_markup=MAIN_MENU_KB)
            return

        if text == "🧪 Процедура":
            _user_mode[user_id] = "procedure"
            await _tg_send_message(token, chat_id, "Режим: procedure", reply_markup=MAIN_MENU_KB)
            return

        if text == "🧮 Расчёт":
            _user_mode[user_id] = "calculation"
            await _tg_send_message(token, chat_id, "Режим: calculation", reply_markup=MAIN_MENU_KB)
            return

        if text == "⚠️ Претензия":
            _user_mode[user_id] = "complaint"
            await _tg_send_message(token, chat_id, "Режим: complaint", reply_markup=MAIN_MENU_KB)
            return

        # --------- КОМАНДЫ ----------
        if text.startswith("/start"):
            _user_mode[user_id] = "norm"
            await _tg_send_message(token, chat_id, "GOST_AI в Telegram.\n\n" + HELP_TEXT, reply_markup=MAIN_MENU_KB)
            return

        if text.startswith("/norm"):
            _user_mode[user_id] = "norm"
            await _tg_send_message(token, chat_id, "Режим: norm", reply_markup=MAIN_MENU_KB)
            return

        if text.startswith("/proc"):
            _user_mode[user_id] = "procedure"
            await _tg_send_message(token, chat_id, "Режим: procedure", reply_markup=MAIN_MENU_KB)
            return

        if text.startswith("/calc"):
            _user_mode[user_id] = "calculation"
            await _tg_send_message(token, chat_id, "Режим: calculation", reply_markup=MAIN_MENU_KB)
            return

        if text.startswith("/claim"):
            _user_mode[user_id] = "complaint"
            await _tg_send_message(token, chat_id, "Режим: complaint", reply_markup=MAIN_MENU_KB)
            return

        if not text:
            return

        task_type = _user_mode.get(user_id, "norm")
        client_id = f"tg:{user_id}"

        await _tg_send_message(token, chat_id, "Ищу в ГОСТах…", reply_markup=MAIN_MENU_KB)

        result = await run_in_threadpool(
            run_chat_sync,
            text,
            task_type,
            client_id,
            "telegram-webhook",
        )

        raw_answer = result.get("answer") or "Нет ответа."
        answer = latex_to_telegram(raw_answer)

        for part in split_telegram(answer, max_len=3500):
            await _tg_send_message(token, chat_id, part, reply_markup=MAIN_MENU_KB)

    except Exception as e:
        try:
            chat_id = ((update.get("message") or {}).get("chat") or {}).get("id")
            if chat_id:
                await _tg_send_message(token, chat_id, f"Ошибка: {e}", reply_markup=MAIN_MENU_KB)
        except Exception:
            pass
