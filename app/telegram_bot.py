from __future__ import annotations

import re
from typing import Dict, Any, List

import httpx
from fastapi.concurrency import run_in_threadpool

from .chat_service import run_chat_sync


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
    "Команды:\n"
    "/norm — режим нормативки\n"
    "/proc — процедура испытаний\n"
    "/calc — расчёт\n"
    "/claim — претензия\n"
)

_user_mode: Dict[int, str] = {}  # user_id -> task_type (память в RAM)


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


async def _tg_send_message(token: str, chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()


async def handle_telegram_update(token: str, update: Dict[str, Any]) -> None:
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

    await _tg_send_message(token, chat_id, "Ищу в ГОСТах…")

    try:
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
            await _tg_send_message(token, chat_id, part)

    except Exception as e:
        await _tg_send_message(token, chat_id, f"Ошибка: {e}")
