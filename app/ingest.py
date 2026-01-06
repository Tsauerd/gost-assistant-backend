# ingest.py
import os
import re
import argparse
from typing import List, Optional

import pdfplumber
from dotenv import load_dotenv
from sqlalchemy import text as sql_text

from .db import SessionLocal
from .rag import embed_text

load_dotenv()


def clean_cell_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


def page_to_markdown(page) -> str:
    parts = []

    text = page.extract_text() or ""
    if text.strip():
        parts.append(text.strip())

    tables = page.extract_tables()
    for t_idx, table in enumerate(tables or []):
        if not table:
            continue

        header_raw = table[0]
        rows = table[1:] if len(table) > 1 else []

        clean_headers = [clean_cell_text(h) for h in header_raw]

        md_lines = []
        md_header = "| " + " | ".join(cell or "" for cell in header_raw) + " |"
        md_sep = "| " + " | ".join("---" for _ in header_raw) + " |"
        md_lines.append(md_header)
        md_lines.append(md_sep)

        verbalized_rows = []

        for row in rows:
            clean_row_md = [cell if cell else "" for cell in row]
            line_md = "| " + " | ".join(clean_row_md) + " |"
            md_lines.append(line_md)

            row_pairs = []
            for h, cell in zip(clean_headers, row):
                c_text = clean_cell_text(cell)
                if h and c_text:
                    row_pairs.append(f"{h}: {c_text}")

            if row_pairs:
                verbalized_rows.append("; ".join(row_pairs) + ".")

        md_table_block = "\n".join(md_lines)
        verbalized_text_block = "\n".join(verbalized_rows)

        table_output = (
            f"\nТаблица {t_idx + 1} (Markdown):\n"
            f"{md_table_block}\n\n"
            f"Описание строк Таблицы {t_idx + 1}:\n"
            f"{verbalized_text_block}\n"
        )
        parts.append(table_output)

    return "\n\n".join(parts)


def read_pdf_with_tables(pdf_path: str) -> str:
    all_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_md = page_to_markdown(page)
            if page_md.strip():
                all_parts.append(f"=== Страница {page_idx + 1} ===\n{page_md}")
    return "\n\n".join(all_parts)


def split_text_to_chunks(text: str, max_chars: int = 2000) -> List[str]:
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


def embedding_to_pgvector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


def extract_section_paragraph(chunk_text: str):
    section = None
    paragraph = None

    lines = [l.strip() for l in chunk_text.splitlines() if l.strip()]
    if not lines:
        return section, paragraph

    full_text = "\n".join(lines)

    para_match = re.search(r"^(\d+(?:\.\d+){1,3})\s", full_text, re.MULTILINE)
    if para_match:
        paragraph = para_match.group(1)

    sec_match = re.search(r"^(\d{1,2})\s+[А-ЯЁA-Z]", lines[0])
    if sec_match:
        section = sec_match.group(1)

    return section, paragraph


def delete_existing_document(standard_number: str, year: int) -> int:
    """
    Удаляет документ(ы) с тем же standard_number+year и все их чанки.
    Возвращает сколько документов удалили.
    """
    with SessionLocal() as db:
        doc_ids = db.execute(
            sql_text("""
                SELECT id FROM documents
                WHERE standard_number = :std AND year = :year
            """),
            {"std": standard_number, "year": year},
        ).scalars().all()

        if not doc_ids:
            return 0

        db.execute(
            sql_text("DELETE FROM document_chunks WHERE document_id = ANY(:ids)"),
            {"ids": doc_ids},
        )
        res = db.execute(
            sql_text("DELETE FROM documents WHERE id = ANY(:ids)"),
            {"ids": doc_ids},
        )
        db.commit()
        return res.rowcount or 0


def insert_document(name: str, standard_number: str, year: int, source: str = "manual_ingest") -> int:
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


def insert_chunk(document_id: int, chunk_index: int, chunk_text: str, embedding: List[float],
                 section: Optional[str] = None, paragraph: Optional[str] = None):
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


def ingest_pdf(pdf_path: str, standard_number: str, year: int, doc_name: str = None, replace_existing: bool = False):
    if doc_name is None:
        doc_name = os.path.basename(pdf_path)

    if replace_existing:
        deleted = delete_existing_document(standard_number, year)
        if deleted:
            print(f"🧹 Удалил старую версию: {standard_number} ({year}) -> документов удалено: {deleted}")

    print(f"Читаем PDF (текст + таблицы): {pdf_path}")
    full_text = read_pdf_with_tables(pdf_path)
    print(f"Длина текста: {len(full_text)} символов")

    chunks = split_text_to_chunks(full_text, max_chars=2000)
    print(f"Получилось чанков: {len(chunks)}")

    doc_id = insert_document(name=doc_name, standard_number=standard_number, year=year)
    print(f"Создан documents.id = {doc_id}")

    for i, raw_chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"Обработка чанка {i+1}/{len(chunks)}...")

        section, paragraph = extract_section_paragraph(raw_chunk)

        header_parts = [f"{standard_number}"]
        if year:
            header_parts.append(str(year))
        if section:
            header_parts.append(f"Раздел {section}")
        if paragraph:
            header_parts.append(f"Пункт {paragraph}")

        header = " | ".join(header_parts)
        chunk_text = f"[{header}]\n{raw_chunk}"

        emb = embed_text(chunk_text)
        insert_chunk(
            document_id=doc_id,
            chunk_index=i,
            chunk_text=chunk_text,
            embedding=emb,
            section=section,
            paragraph=paragraph
        )

    print("✅ Готово: документ и все чанки загружены.")


def main():
    parser = argparse.ArgumentParser(description="Ingest GOST PDF into DB")
    parser.add_argument("--file", required=True, help="Путь к PDF")
    parser.add_argument("--std", required=True, help="Номер стандарта")
    parser.add_argument("--year", required=True, type=int, help="Год")
    parser.add_argument("--name", help="Название документа")
    parser.add_argument("--replace-existing", action="store_true", help="Удалить старую версию этого стандарта перед ingest")
    args = parser.parse_args()

    ingest_pdf(
        pdf_path=args.file,
        standard_number=args.std,
        year=args.year,
        doc_name=args.name,
        replace_existing=args.replace_existing,
    )


if __name__ == "__main__":
    main()
