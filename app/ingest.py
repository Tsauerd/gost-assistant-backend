# app/ingest.py
from __future__ import annotations

import os
import re
import argparse
from typing import List, Optional, Tuple

import pdfplumber
from dotenv import load_dotenv
from sqlalchemy import text as sql_text

from .db import SessionLocal
from .rag import embed_text  # ВАЖНО: embed_text должен существовать в rag.py

load_dotenv()


def clean_cell_text(text: Optional[str]) -> str:
    """Убираем лишние переносы и пробелы для 'плоского' текста."""
    if not text:
        return ""
    return " ".join(str(text).strip().split())


def page_to_markdown(page) -> str:
    """
    Текст + таблицы со страницы.
    Таблицы: Markdown + вербализация (для улучшения поиска).
    """
    parts: List[str] = []

    # 1) Текст
    text = page.extract_text() or ""
    text = text.strip()
    if text:
        parts.append(text)

    # 2) Таблицы
    tables = page.extract_tables()
    for t_idx, table in enumerate(tables or []):
        if not table or not table[0]:
            continue

        header_raw = table[0]
        rows = table[1:] if len(table) > 1 else []

        clean_headers = [clean_cell_text(h) for h in header_raw]

        md_lines: List[str] = []
        md_header = "| " + " | ".join(clean_cell_text(cell) for cell in header_raw) + " |"
        md_sep = "| " + " | ".join("---" for _ in header_raw) + " |"
        md_lines.append(md_header)
        md_lines.append(md_sep)

        verbalized_rows: List[str] = []

        for row in rows:
            row = row or []
            clean_row = [clean_cell_text(cell) for cell in row]
            # markdown строка (по количеству колонок как header)
            padded = (clean_row + [""] * len(clean_headers))[:len(clean_headers)]
            md_lines.append("| " + " | ".join(padded) + " |")

            row_pairs = []
            for h, cell in zip(clean_headers, padded):
                if h and cell:
                    row_pairs.append(f"{h}: {cell}")
            if row_pairs:
                verbalized_rows.append("; ".join(row_pairs) + ".")

        md_table_block = "\n".join(md_lines)
        verbalized_text_block = "\n".join(verbalized_rows).strip() or "(нет данных)"

        table_output = (
            f"\nТаблица {t_idx + 1}:\n"
            f"{md_table_block}\n\n"
            f"Описание строк таблицы {t_idx + 1}:\n"
            f"{verbalized_text_block}\n"
        )
        parts.append(table_output)

    return "\n\n".join(parts).strip()


def read_pdf_with_tables(pdf_path: str) -> str:
    """Считываем PDF: текст + таблицы."""
    all_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_md = page_to_markdown(page)
            if page_md:
                all_parts.append(f"=== Страница {page_idx + 1} ===\n{page_md}")
    return "\n\n".join(all_parts)


def split_text_to_chunks(text: str, max_chars: int = 2000) -> List[str]:
    """
    Чанкинг по пустым строкам (абзацам).
    Длинные блоки режем по символам.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    chunks: List[str] = []
    current = ""

    for block in blocks:
        candidate = (current + "\n\n" + block).strip() if current else block

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)

            if len(block) <= max_chars:
                current = block
            else:
                start = 0
                while start < len(block):
                    end = start + max_chars
                    chunks.append(block[start:end].strip())
                    start = end
                current = ""

    if current:
        chunks.append(current)

    return chunks


def extract_section_paragraph(chunk_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    section: "5"
    paragraph: "5.2.1"
    """
    section = None
    paragraph = None

    lines = [l.strip() for l in chunk_text.splitlines() if l.strip()]
    if not lines:
        return None, None

    full_text = "\n".join(lines)

    para_match = re.search(r"^(\d+(?:\.\d+){1,3})\s", full_text, re.MULTILINE)
    if para_match:
        paragraph = para_match.group(1)

    sec_match = re.search(r"^(\d{1,2})\s+[А-ЯЁA-Z]", lines[0])
    if sec_match:
        section = sec_match.group(1)

    return section, paragraph


def purge_document(db, standard_number: str, year: Optional[int]) -> int:
    """
    Удаляет из БД документы и чанки по standard_number (+ year если задан).
    Возвращает число удалённых документов.
    """
    where = "standard_number = :std"
    params = {"std": standard_number}
    if year is not None:
        where += " AND year = :year"
        params["year"] = year

    ids = db.execute(
        sql_text(f"SELECT id FROM documents WHERE {where}"),
        params,
    ).scalars().all()

    if not ids:
        return 0

    db.execute(
        sql_text("DELETE FROM document_chunks WHERE document_id = ANY(:ids)"),
        {"ids": ids},
    )
    db.execute(
        sql_text("DELETE FROM documents WHERE id = ANY(:ids)"),
        {"ids": ids},
    )
    return len(ids)


def ingest_pdf(
    pdf_path: str,
    standard_number: str,
    year: int,
    doc_name: Optional[str] = None,
    purge: bool = False,
    source: str = "manual_ingest",
):
    if doc_name is None:
        doc_name = os.path.basename(pdf_path)

    print(f"Читаем PDF (текст + таблицы): {pdf_path}")
    full_text = read_pdf_with_tables(pdf_path)
    print(f"Длина текста: {len(full_text)} символов")

    chunks = split_text_to_chunks(full_text, max_chars=2000)
    print(f"Чанков: {len(chunks)}")

    insert_doc_sql = sql_text("""
        INSERT INTO documents (name, standard_number, year, source)
        VALUES (:name, :std, :year, :src)
        RETURNING id;
    """)

    insert_chunk_sql = sql_text("""
        INSERT INTO document_chunks (document_id, chunk_index, text, section, paragraph, embedding)
        VALUES (:doc_id, :idx, :text, :section, :paragraph, :embedding)
    """)

    # ВАЖНО: одна сессия на документ = быстрее
    with SessionLocal() as db:
        if purge:
            deleted = purge_document(db, standard_number=standard_number, year=year)
            db.commit()
            if deleted:
                print(f"🧹 purge: удалено документов: {deleted}")

        doc_id = db.execute(insert_doc_sql, {
            "name": doc_name,
            "std": standard_number,
            "year": year,
            "src": source,
        }).scalar()
        db.commit()
        print(f"Создан documents.id = {doc_id}")

        for i, raw_chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"Обработка чанка {i+1}/{len(chunks)}...")

            section, paragraph = extract_section_paragraph(raw_chunk)

            header_parts = [f"ГОСТ {standard_number}", str(year)]
            if section:
                header_parts.append(f"Раздел {section}")
            if paragraph:
                header_parts.append(f"Пункт {paragraph}")

            header = " | ".join(header_parts)
            chunk_text = f"[{header}]\n{raw_chunk}"

            emb = embed_text(chunk_text)  # модель должна совпадать с rag.py

            # pgvector literal
            emb_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

            db.execute(insert_chunk_sql, {
                "doc_id": doc_id,
                "idx": i,
                "text": chunk_text,
                "section": section,
                "paragraph": paragraph,
                "embedding": emb_literal,
            })

            # коммитим порциями, чтобы не держать гигантскую транзакцию
            if (i + 1) % 50 == 0:
                db.commit()

        db.commit()

    print("✅ Готово: документ и все чанки загружены.")


def main():
    parser = argparse.ArgumentParser(description="Ingest GOST PDF into DB")
    parser.add_argument("--file", required=True, help="Путь к PDF")
    parser.add_argument("--std", required=True, help="Номер стандарта (например, 'ГОСТ 12345-2020')")
    parser.add_argument("--year", required=True, type=int, help="Год стандарта")
    parser.add_argument("--name", help="Название документа (по умолчанию — имя файла)")
    parser.add_argument("--purge", action="store_true", help="Удалить старую версию этого стандарта перед загрузкой")
    args = parser.parse_args()

    ingest_pdf(
        pdf_path=args.file,
        standard_number=args.std,
        year=args.year,
        doc_name=args.name,
        purge=args.purge,
    )


if __name__ == "__main__":
    main()
