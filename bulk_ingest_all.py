# bulk_ingest_all.py
from pathlib import Path
import re

from app.ingest import ingest_pdf  # берем функцию из app/ingest.py

# Папка с PDF ГОСТами
FOLDER = Path(r"C:\Users\artem\Documents\gost-assistant-backend\data\new")


def parse_std_and_year(base: str):
    """
    base: имя файла без .pdf, например:
      'GOST 10060 — 2012'
      'GOST 26633-2015 – Heavy-weight and sand concretes'
      'GOST 18353-79'
      'SP 48.13330.2019'
      'GOST-31937_2011.-Mezhgosudarstvenny-standart...'

    Возвращает (standard_number, year) или (None, None), если не получилось.
    """
    # Нормализуем тире, подчёркивания и т.п.
    norm = (
        base.replace("—", "-")
            .replace("–", "-")
            .replace("_", ".")
    )

    # Отделяем префикс (GOST / GOST R / SP) и остальную часть
    m = re.match(r"^(GOST\s*R|GOSTR|GOST|SP)\s*[- ]*(.+)$", norm, re.IGNORECASE)
    if not m:
        return None, None

    prefix = m.group(1).upper()
    rest = m.group(2).strip()

    if prefix == "GOSTR":
        prefix = "GOST R"

    # --- Случай СП: SP 48.13330.2019 ---
    # Шаблон: 2 цифры . 5 цифр . (2–4 цифры года)
    m_sp = None
    if prefix.startswith("SP"):
        m_sp = re.search(r"(\d{2}\.\d{5})\.(\d{2,4})", rest)
    if m_sp:
        code_main = m_sp.group(1)              # "48.13330"
        y_str = m_sp.group(2)                  # "2019"
        if len(y_str) == 4:
            year = int(y_str)
        else:
            yy = int(y_str)
            year = 2000 + yy if yy <= 30 else 1900 + yy
        code = f"{code_main}.{y_str}"
        standard_number = f"{prefix} {code}"
        return standard_number, year

    # --- Случай ГОСТ: число - год (с пробелами или без) ---
    # Например:
    #   "10060-2012"
    #   "10060 - 2012"
    #   "18353-79"
    m_gy = re.search(r"\b(\d{4,6})\s*-\s*(\d{2,4})\b", rest)
    if m_gy:
        num = m_gy.group(1)  # 10060
        y_str = m_gy.group(2)  # "2012" или "79"

        if len(y_str) == 4:
            year = int(y_str)
        else:
            yy = int(y_str)
            year = 2000 + yy if yy <= 30 else 1900 + yy

        code = f"{num}-{y_str}"
        standard_number = f"{prefix} {code}"
        return standard_number, year

    # --- Фоллбек: первая "большая" цифра + последний 4-значный год ---
    y4_all = re.findall(r"\b(\d{4})\b", rest)
    year = int(y4_all[-1]) if y4_all else None

    m_num = re.search(r"\b(\d{4,6})\b", rest)
    main_num = m_num.group(1) if m_num else None

    if main_num and year:
        code = f"{main_num}-{year}"
        standard_number = f"{prefix} {code}"
        return standard_number, year

    return None, None


def main():
    print(f"Сканирую папку: {FOLDER}")
    pdf_files = sorted(FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("PDF-файлов не найдено.")
        return

    for pdf in pdf_files:
        base = pdf.stem
        std, year = parse_std_and_year(base)

        if not std or not year:
            print(f"⚠ Пропускаю '{pdf.name}': не смог разобрать номер/год стандарта")
            continue

        print(f"➡ Ingest: {pdf.name} -> {std} ({year})")

        ingest_pdf(
            pdf_path=str(pdf),
            standard_number=std,
            year=year,
            doc_name=base,
        )

    print("✅ Готово: все подходящие файлы обработаны.")


if __name__ == "__main__":
    main()
