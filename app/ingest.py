# app/ingest.py
from __future__ import annotations

import os
import re
import argparse
from typing import List, Tuple, Optional

import pdfplumber
from dotenv import load_dotenv
from sqlalchemy import text as sql_text
from openai import OpenAI

from .db import SessionLocal

load_dotenv()

# ===== Embeddings config =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# поддержим оба имени переменной, чтобы не путаться между файлами
EMBED_MODEL = (
    os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
    or os.getenv("EMBED_MODEL", "").strip()
    or "text-embedding-3-small"
)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in env")

client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------- helpers: tables/text --------------------

def clean_cell_text(text: Optional[str]) -> str:
    """Убираем лишние переносы и пробелы для 'плоского' текста."""
    if not text:
        return ""
    return " ".join(str(text).strip().split())


def page_to_markdown(page) -> str:
    """
    Текст + таблицы со страницы.
    Таблицы:
      - Markdown блок (чтобы LLM мог читать структуру)
      - Вербализация строк (чтобы embedding лучше находил значения)
    """
    parts: List[str] = []

    # 1) обычный текст
    text = page.extract_text() or ""
    if text.strip():
        parts.append(text.strip())

    # 2) таблицы
    tables = page.extract_tables()  # pdfplumber это поддерживает
    for t_idx, table in enumerate(tables or []):
        if not table or not any(any(cell for cell in row) for row in table):
            continue

        header_raw = table[0] if table else []
        rows = table[1:] if len(table) > 1 else []

        header_clean = [clean_cell_text(h) for h in header_raw]

        # Markdown таблица
        md_lines: List[str] = []
        md_header = "| " + " | ".join(clean_cell_text(cell) for cell in header_raw) + " |"
        md_sep = "| " + " | ".join("---" for _ in header_raw) + " |"
        md_lines.append(md_header)
        md_lines.append(md_sep)

        verbalized_rows: List[str] = []

        for r_idx, row in enumerate(rows, start=1):
            row_cells = [clean_cell_text(cell) for cell in row]
            if not any(row_cells):
                continue

            md_lines.append("| " + " | ".join(row_cells) + " |")

            pairs: List[str] = []
            for h, c in zip(header_clean, row_cells):
                if h and c:
                    pairs.append(f"{h}: {c}")

            # добавим номер строки — удобно для ссылок
            if pairs:
                verbalized_rows.append(f"Строка {r_idx}: " + "; ".join(pairs) + ".")

        md_table_block = "\n".join(md_lines).strip()
        verbalized_block = "\n".join(verbalized_rows).strip()

        table_output = (
            f"\nТаблица {t_idx + 1}:\n"
            f"{md_table_block}\n"
        )
        if verbalized_block:
            table_output += f"\nОписание таблицы {t_idx + 1} (построчно):\n{verbalized_block}\n"

        parts.append(table_output)

    return "\n\n".join(p for p in parts if p.strip()).strip()


def iter_pdf_pages_markdown(pdf_path: str) -> List[Tuple[int, str]]:
    """Возвращаем список (page_no, text_with_tables) по страницам."""
    out: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_md = page_to_markdown(page)
            if page_md.strip():
                out.append((i, page_md))
    return out


def split_text_to_chunks(text: str, max_chars: int = 2000, overlap_chars: int = 200) -> List[str]:
    """
    Чанкинг по абзацам (пустая строка).
    overlap_chars — небольшой нахлёст, чтобы не терять связь на границе.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks: List[str] = []
    current = ""

    for block in blocks:
        candidate = (current + "\n\n" + block).strip() if current else block

        if len(candidate) <= max_chars:
            current = candidate
            continue

        # фиксируем текущий чанк
        if current:
            chunks.append(current)

            # overlap: хвост предыдущего чанка
            tail = current[-overlap_chars:] if overlap_chars and len(current) > overlap_chars else ""
        else:
            tail = ""

        # если block сам слишком большой — режем грубо
        if len(block) > max_chars:
            start = 0
            while start < len(block):
                end = start + max_chars
                piece = block[start:end].strip()
                if piece:
                    chunks.append(piece)
                start = end
            current = ""
        else:
            current = (tail + "\n\n" + block).strip() if tail else block

    if current:
        chunks.append(current)

    return chunks


# -------------------- helpers: embeddings/db --------------------

def embed_text(text: str) -> List[float]:
    """Единая функция эмбеддинга для ingest."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def detect_existing_vector_dims(db) -> Optional[int]:
    """
    Узнаём размерность уже существующих embedding в БД.
    Важно: если в таблице намешаны разные dims — будет боль при поиске.
    """
    q = sql_text("""
        SELECT vector_dims(embedding) AS dims
        FROM document_chunks
        WHERE embedding IS NOT NULL
        LIMIT 1
    """)
    row = db.execute(q).mappings().first()
    if row and row.get("dims"):
        return int(row["dims"])
    return None


def delete_document_if_exists(db, standard_number: str, year: int) -> None:
    """
    Удаляем документ и его чанки по (standard_number, year).
    Это чтобы повторный ingest не делал дубли.
    """
    # найдём id
    q = sql_text("""
        SELECT id FROM documents
        WHERE standard_number = :std AND year = :year
        LIMIT 1
    """)
    row = db.execute(q, {"std": standard_number, "year": year}).mappings().first()
    if not row:
        return

    doc_id = int(row["id"])

    db.execute(sql_text("DELETE FROM document_chunks WHERE document_id = :id"), {"id": doc_id})
    db.execute(sql_text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})


def extract_section_paragraph(chunk_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Эвристика:
      - paragraph: 5.2 / 5.2.1 ...
      - section: "5 Методы контроля"
    """
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


def ingest_pdf(
    pdf_path: str,
    standard_number: str,
    year: int,
    doc_name: Optional[str] = None,
    replace_existing: bool = True,
    max_chars: int = 2000,
    overlap_chars: int = 200,
    batch_size: int = 50,
) -> None:
    if doc_name is None:
        doc_name = os.path.basename(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    print(f"[ingest] PDF: {pdf_path}")
    print(f"[ingest] EMBED_MODEL={EMBED_MODEL}")

    pages = iter_pdf_pages_markdown(pdf_path)
    if not pages:
        print("[ingest] No text extracted from PDF (empty result).")
        return

    with SessionLocal() as db:
        # 1) Проверка размерности (если в БД уже есть эмбеддинги)
        existing_dims = detect_existing_vector_dims(db)
        # Сгенерируем 1 embedding на коротком тексте, чтобы узнать dims текущей модели
        probe_vec = embed_text("probe")
        model_dims = len(probe_vec)

        if existing_dims and existing_dims != model_dims:
            raise RuntimeError(
                f"DB already contains embeddings with dims={existing_dims}, "
                f"but current model '{EMBED_MODEL}' returns dims={model_dims}. "
                f"Нужно очистить document_chunks/documents (или держать одну модель)."
            )

        # 2) Удалим старую версию документа, если нужно
        if replace_existing:
            delete_document_if_exists(db, standard_number=standard_number, year=year)

        # 3) Создаём документ
        ins_doc = sql_text("""
            INSERT INTO documents (name, standard_number, year, source)
            VALUES (:name, :std, :year, :src)
            RETURNING id;
        """)
        doc_id = db.execute(ins_doc, {
            "name": doc_name,
            "std": standard_number,
            "year": year,
            "src": f"ingest_pdfplumber:{EMBED_MODEL}",
        }).scalar()
        db.commit()

        print(f"[ingest] documents.id={doc_id}")

        # 4) Готовим insert для чанков
        ins_chunk = sql_text("""
            INSERT INTO document_chunks
                (document_id, chunk_index, text, section, paragraph, embedding)
            VALUES
                (:doc_id, :idx, :text, :section, :paragraph, :embedding)
        """)

        chunk_index = 0
        buffer_params: List[dict] = []

        for page_no, page_text in pages:
            # режем на чанки внутри страницы
            chunks = split_text_to_chunks(page_text, max_chars=max_chars, overlap_chars=overlap_chars)

            for raw_chunk in chunks:
                section, paragraph = extract_section_paragraph(raw_chunk)

                header_parts = [f"ГОСТ {standard_number}", str(year), f"Стр. {page_no}"]
                if section:
                    header_parts.append(f"Раздел {section}")
                if paragraph:
                    header_parts.append(f"Пункт {paragraph}")

                header = " | ".join(header_parts)
                chunk_text = f"[{header}]\n{raw_chunk}".strip()

                vec = embed_text(chunk_text)
                if len(vec) != model_dims:
                    raise RuntimeError(f"Embedding dims mismatch: got {len(vec)} expected {model_dims}")

                buffer_params.append({
                    "doc_id": doc_id,
                    "idx": chunk_index,
                    "text": chunk_text,
                    "section": section,
                    "paragraph": paragraph,
                    "embedding": to_pgvector_literal(vec),
                })

                chunk_index += 1

                if len(buffer_params) >= batch_size:
                    db.execute(ins_chunk, buffer_params)
                    db.commit()
                    print(f"[ingest] inserted {chunk_index} chunks...")
                    buffer_params.clear()

        # вставим хвост
        if buffer_params:
            db.execute(ins_chunk, buffer_params)
            db.commit()

        print(f"[ingest] DONE. Total chunks inserted: {chunk_index}")


def main():
    parser = argparse.ArgumentParser(description="Ingest GOST PDF into DB")
    parser.add_argument("--file", required=True, help="Путь к PDF-файлу")
    parser.add_argument("--std", required=True, help="Номер стандарта (пример: '12345-2020' или 'ГОСТ 12345-2020')")
    parser.add_argument("--year", required=True, type=int, help="Год стандарта")
    parser.add_argument("--name", help="Название документа (по умолчанию имя файла)")

    parser.add_argument("--no-replace", action="store_true", help="НЕ удалять старую версию (по умолчанию удаляем)")
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)

    args = parser.parse_args()

    ingest_pdf(
        pdf_path=args.file,
        standard_number=args.std,
        year=args.year,
        doc_name=args.name,
        replace_existing=not args.no_replace,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
