import os
import argparse
from typing import List

import pdfplumber
from dotenv import load_dotenv
from sqlalchemy import text as sql_text

from .db import SessionLocal
from .rag import embed_text  # используем нашу обновлённую функцию

load_dotenv()


def page_to_markdown(page) -> str:
    """
    Достаём текст + таблицы с одной страницы и превращаем в markdown.
    """
    parts = []

    # Обычный текст
    text = page.extract_text() or ""
    if text.strip():
        parts.append(text.strip())

    # Таблицы
    tables = page.extract_tables()
    for t_idx, table in enumerate(tables or []):
        if not table:
            continue
        # Первая строка — заголовок (часто так и есть)
        header = table[0]
        rows = table[1:] if len(table) > 1 else []

        md_lines = []
        # Заголовок
        md_header = "| " + " | ".join(cell or "" for cell in header) + " |"
        md_sep = "| " + " | ".join("---" for _ in header) + " |"
        md_lines.append(md_header)
        md_lines.append(md_sep)

        # Строки
        for row in rows:
            line = "| " + " | ".join(cell or "" for cell in row) + " |"
            md_lines.append(line)

        md_table = "\n".join(md_lines)
        parts.append(f"Таблица {t_idx + 1} (markdown):\n{md_table}")

    return "\n\n".join(parts)


def read_pdf_with_tables(pdf_path: str) -> str:
    """Считываем PDF, вытаскиваем текст + таблицы в markdown."""
    all_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_md = page_to_markdown(page)
            if page_md.strip():
                all_parts.append(f"=== Страница {page_idx + 1} ===\n{page_md}")
    return "\n\n".join(all_parts)


def split_text_to_chunks(
    text: str,
    max_chars: int = 1200,
    overlap: int = 200
) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]


def embedding_to_pgvector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(str(round(x, 6)) for x in embedding) + "]"


def insert_document(
    name: str,
    standard_number: str,
    year: int,
    source: str = "manual_ingest"
) -> int:
    insert_sql = sql_text("""
        INSERT INTO documents (name, standard_number, year, source)
        VALUES (:name, :std, :year, :src)
        RETURNING id;
    """)
    with SessionLocal() as db:
        doc_id = db.execute(insert_sql, {
            "name": name,
            "std": standard_number,
            "year": year,
            "src": source
        }).scalar()
        db.commit()
        return doc_id


def insert_chunk(
    document_id: int,
    chunk_index: int,
    chunk_text: str,
    embedding: List[float],
    section: str = None,
    paragraph: str = None
):
    emb_literal = embedding_to_pgvector_literal(embedding)

    insert_sql = sql_text("""
        INSERT INTO document_chunks
            (document_id, chunk_index, text, section, paragraph, embedding)
        VALUES
            (:doc_id, :idx, :text, :section, :paragraph, :embedding)
    """)
    with SessionLocal() as db:
        db.execute(insert_sql, {
            "doc_id": document_id,
            "idx": chunk_index,
            "text": chunk_text,
            "section": section,
            "paragraph": paragraph,
            "embedding": emb_literal
        })
        db.commit()


def ingest_pdf(
    pdf_path: str,
    standard_number: str,
    year: int,
    doc_name: str = None
):
    if doc_name is None:
        doc_name = os.path.basename(pdf_path)

    print(f"Читаем PDF (текст + таблицы): {pdf_path}")
    full_text = read_pdf_with_tables(pdf_path)

    print(f"Длина текста (с таблицами): {len(full_text)} символов")
    chunks = split_text_to_chunks(full_text, max_chars=1200, overlap=200)
    print(f"Получилось чанков: {len(chunks)}")

    # 1. Запись в documents
    doc_id = insert_document(
        name=doc_name,
        standard_number=standard_number,
        year=year
    )
    print(f"Создан documents.id = {doc_id}")

    # 2. Чанки + embeddings
    for i, chunk in enumerate(chunks):
        print(f"Чанк {i+1}/{len(chunks)} — считаем embedding...")
        emb = embed_text(chunk)  # text-embedding-3-large
        insert_chunk(
            document_id=doc_id,
            chunk_index=i,
            chunk_text=chunk,
            embedding=emb
        )

    print("Готово: документ и все чанки загружены.")


def main():
    parser = argparse.ArgumentParser(description="Ingest GOST PDF into DB")
    parser.add_argument("--file", required=True, help="Путь к PDF-файлу ГОСТа")
    parser.add_argument("--std", required=True, help="Номер стандарта (например, 'ГОСТ 12345-2020')")
    parser.add_argument("--year", required=True, type=int, help="Год стандарта")
    parser.add_argument("--name", help="Название документа (по умолчанию — имя файла)")
    args = parser.parse_args()

    ingest_pdf(
        pdf_path=args.file,
        standard_number=args.std,
        year=args.year,
        doc_name=args.name
    )


if __name__ == "__main__":
    main()
