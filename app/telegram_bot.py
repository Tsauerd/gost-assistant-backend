from __future__ import annotations

import re
from typing import List

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

HELP_TEXT = (
    "Напиши вопрос по ГОСТам.\n\n"
    "Команды:\n"
    "/norm — режим нормативки\n"
    "/proc — процедура испытаний\n"
    "/calc — расчёт\n"
    "/claim — претензия\n"
)


def _strip_wrapped_commands(text: str) -> str:
    for _ in range(3):
        text = re.sub(r"\\(text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\2", text)
    return text


def latex_to_telegram(text: str) -> str:
    if not text:
        return text

    # снимаем math-обёртки
    text = re.sub(r"\$\$([\s\S]*?)\$\$", r"\1", text)
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text, flags=re.S)
    text = re.sub(r"\$([^\n$]+)\$", r"\1", text)

    text = _strip_wrapped_commands(text)

    for k, v in LATEX_MAP.items():
        text = text.replace(k, v)

    # дроби
    def frac(m):
        a = m.group(1).strip()
        b = m.group(2).strip()
        return f"({a})/({b})"

    for _ in range(2):
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", frac, text)

    # индексы
    text = re.sub(r"([A-Za-zА-Яа-я])_\{([0-9+\-()n=]+)\}", lambda m: m.group(1) + m.group(2).translate(SUB), text)
    text = re.sub(r"([A-Za-zА-Яа-я])_([0-9]+)", lambda m: m.group(1) + m.group(2).translate(SUB), text)

    # степени
    text = re.sub(r"([A-Za-zА-Яа-я0-9])\^\{([0-9+\-()n=]+)\}", lambda m: m.group(1) + m.group(2).translate(SUP), text)
    text = re.sub(r"([A-Za-zА-Яа-я0-9])\^([0-9]+)", lambda m: m.group(1) + m.group(2).translate(SUP), text)

    # чистка
    text = re.sub(r"\\(left|right)\b", "", text)
    text = text.replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ")
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def split_telegram(text: str, max_len: int = 3500) -> List[str]:
    if len(text) <= max_len:
        return [text]
    parts: List[str] = []
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
