import os
import re
import requests
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.error import BadRequest

load_dotenv()

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
BACKEND_URL = os.environ.get("BACKEND_URL", "https://gost-assistant-backend.onrender.com")

# подстрочные индексы и степени
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
    """
    Убираем оболочки типа \text{...}, \mathrm{...}, \mathbf{...}
    """
    # несколько проходов, т.к. команды могут быть вложенными
    for _ in range(3):
        text = re.sub(r"\\(text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\2", text)
    return text

def latex_to_telegram(text: str) -> str:
    """
    Конвертирует LaTeX/MathJax разметку в читабельный Unicode-текст для Telegram.
    Telegram не умеет рендерить LaTeX, поэтому делаем "плоский" вид.
    """
    if not text:
        return text

    # 1) Снимаем math-обёртки (аккуратно)
    # $$ ... $$  (блок)
    text = re.sub(r"\$\$([\s\S]*?)\$\$", r"\1", text)
    # \[ ... \]  (блок)
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text, flags=re.S)
    # \( ... \)  (inline)
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text, flags=re.S)
    # $ ... $    (inline) — осторожно, без переносов строк
    text = re.sub(r"\$([^\n$]+)\$", r"\1", text)

    # 2) Убираем оболочки \text{...}, \mathrm{...}
    text = _strip_wrapped_commands(text)

    # 3) Простые замены команд
    for k, v in LATEX_MAP.items():
        text = text.replace(k, v)

    # 4) Дроби \frac{a}{b} -> (a)/(b)
    def frac(m):
        a = m.group(1).strip()
        b = m.group(2).strip()
        return f"({a})/({b})"
    # несколько проходов, т.к. дроби могут быть вложенные
    for _ in range(2):
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", frac, text)

    # 5) Индексы: m_1 -> m₁ и m_{12} -> m₁₂
    def sub1(m):
        base = m.group(1)
        idx = m.group(2)
        return base + idx.translate(SUB)
    text = re.sub(r"([A-Za-zА-Яа-я])_([0-9]+)", sub1, text)

    def sub2(m):
        base = m.group(1)
        idx = m.group(2)
        return base + idx.translate(SUB)
    text = re.sub(r"([A-Za-zА-Яа-я])_\{([0-9+\-()n=]+)\}", sub2, text)

    # 6) Степени: x^2 -> x² и 10^6 -> 10⁶, x^{12} -> x¹²
    def sup1(m):
        base = m.group(1)
        p = m.group(2)
        return base + p.translate(SUP)
    text = re.sub(r"([A-Za-zА-Яа-я0-9])\^([0-9]+)", sup1, text)

    def sup2(m):
        base = m.group(1)
        p = m.group(2)
        return base + p.translate(SUP)
    text = re.sub(r"([A-Za-zА-Яа-я0-9])\^\{([0-9+\-()n=]+)\}", sup2, text)

    # 7) Убираем left/right и прочие команды-разделители
    text = re.sub(r"\\(left|right)\b", "", text)
    text = text.replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ")

    # 8) Убираем остаточные команды \something
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # 9) Немного чистки пробелов
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text

def split_telegram(text: str, max_len: int = 3500):
    """
    Делим длинный текст на куски, чтобы не резать посередине.
    """
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["task_type"] = "norm"
    await update.message.reply_text("GOST_AI в Telegram.\n\n" + HELP_TEXT)

async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE, mode: str):
    context.user_data["task_type"] = mode
    await update.message.reply_text(f"Режим переключен: {mode}")

async def norm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_mode(update, context, "norm")

async def proc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_mode(update, context, "procedure")

async def calc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_mode(update, context, "calculation")

async def claim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_mode(update, context, "complaint")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    task_type = context.user_data.get("task_type", "norm")
    client_id = f"tg:{update.effective_user.id}"

    status_msg = await update.message.reply_text("Ищу в ГОСТах…")

    try:
        resp = requests.post(
            f"{BACKEND_URL}/chat",
            json={"query": user_text, "task_type": task_type, "client_id": client_id},
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()

        raw_answer = data.get("answer") or "Нет ответа."
        answer = latex_to_telegram(raw_answer)

        # удаляем "Ищу..." безопасно
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=status_msg.message_id
            )
        except BadRequest:
            pass

        # отправляем частями
        for part in split_telegram(answer, max_len=3500):
            await update.message.reply_text(part)

    except Exception as e:
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=status_msg.message_id
            )
        except BadRequest:
            pass
        await update.message.reply_text(f"Ошибка при обращении к серверу: {e}")

def main():
    if not BOT_TOKEN:
        print("Ошибка: Не задан TELEGRAM_BOT_TOKEN в переменных окружения.")
        return

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("norm", norm))
    app.add_handler(CommandHandler("proc", proc))
    app.add_handler(CommandHandler("calc", calc))
    app.add_handler(CommandHandler("claim", claim))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
