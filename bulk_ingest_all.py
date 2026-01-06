# bulk_ingest_all.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

from app.ingest import ingest_pdf


def _normalize_filename(base: str) -> str:
    s = base.strip()
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _two_digit_year_to_four(y: int) -> int:
    # эвристика: 00–30 -> 2000+, иначе 1900+
    return 2000 + y if y <= 30 else 1900 + y


def parse_std_and_year(base: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Возвращает (standard_number, year_4digits) или (None, None).
    """
    norm = _normalize_filename(base)

    # prefix: GOST / GOST R / ГОСТ / ГОСТ Р / SP / СП
    m = re.match(r"^(GOST\s*R|GOSTR|GOST|ГОСТ\s*Р|ГОСТР|ГОСТ|SP|СП)\s*[- ]*(.+)$", norm, re.IGNORECASE)
    if not m:
        return None, None

    raw_prefix = m.group(1).upper().replace("  ", " ").strip()
    rest = m.group(2).strip()

    # нормализуем префикс
    if raw_prefix in ("GOSTR",):
        prefix = "GOST R"
    elif raw_prefix in ("ГОСТР",):
        prefix = "ГОСТ Р"
    else:
        prefix = raw_prefix.replace("  ", " ")

    # --- СП: SP 48.13330.2019 ---
    if prefix in ("SP", "СП"):
        m_sp = re.search(r"(\d{1,3}\.\d{5})\.(\d{2,4})", rest)
        if m_sp:
            code_main = m_sp.group(1)   # 48.13330
            y_str = m_sp.group(2)       # 2019 или 19
            if len(y_str) == 4:
                year = int(y_str)
            else:
                year = _two_digit_year_to_four(int(y_str))
            standard_number = f"{prefix} {code_main}.{y_str}"
            return standard_number, year

    # --- ГОСТ: 26633-2015 или 18353-79 ---
    m_gy = re.search(r"\b(\d{4,6})\s*-\s*(\d{2,4})\b", rest)
    if m_gy:
        num = m_gy.group(1)
        y_str = m_gy.group(2)
        if len(y_str) == 4:
            year = int(y_str)
        else:
            year = _two_digit_year_to_four(int(y_str))

        standard_number = f"{prefix} {num}-{y_str}"
        return standard_number, year

    # fallback: последний 4-значный год + первая крупная цифра
    y4_all = re.findall(r"\b(\d{4})\b", rest)
    year = int(y4_all[-1]) if y4_all else None

    m_num = re.search(r"\b(\d{4,6})\b", rest)
    main_num = m_num.group(1) if m_num else None

    if main_num and year:
        standard_number = f"{prefix} {main_num}-{year}"
        return standard_number, year

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest all PDFs from folder into DB")
    parser.add_argument("--folder", default=None, help="Папка с PDF (по умолчанию ./data/new)")
    parser.add_argument("--replace", action="store_true", help="Удалять старые версии стандарта перед загрузкой")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    folder = Path(args.folder) if args.folder else (base_dir / "data" / "new")
    folder = folder.resolve()

    print(f"Сканирую папку: {folder}")
    pdf_files = sorted(folder.glob("*.pdf"))

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
            replace_existing=args.replace,
        )

    print("✅ Готово: все подходящие файлы обработаны.")


if __name__ == "__main__":
    main()
