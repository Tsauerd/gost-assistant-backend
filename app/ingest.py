import os
import argparse
from typing import List

import pdfplumber
from dotenv import load_dotenv
from sqlalchemy import text as sql_text

# Предполагается, что эти модули существуют в вашем проекте
from .db import SessionLocal
from .rag import embed_text  # используем нашу обновлённую функцию

load_dotenv()


def clean_cell_text(text: str) -> str:
    """Убираем лишние переносы и пробелы для 'плоского' текста."""
    if not text:
        return ""
    return " ".join(text.strip().split())


def page_to_markdown(page) -> str:
    """
    Достаём текст + таблицы с одной страницы.
    Таблицы преобразуем в Markdown И добавляем их вербализацию (текстовое описание)
    для улучшения семантического поиска.
    """
    parts = []

    # 1. Обычный текст страницы
    text = page.extract_text() or ""
    if text.strip():
        parts.append(text.strip())

    # 2. Обработка таблиц
    tables = page.extract_tables()
    for t_idx, table in enumerate(tables or []):
        if not table:
            continue
        
        # Первая строка — заголовок
        header_raw = table[0]
        rows = table[1:] if len(table) > 1 else []

        # Очищаем заголовки для использования в вербализации
        # (Header: Value) работает лучше, чем сырой markdown
        clean_headers = [clean_cell_text(h) for h in header_raw]

        # --- Блок 1: Markdown представление (для визуального чтения LLM) ---
        md_lines = []
        # Заголовок
        md_header = "| " + " | ".join(cell or "" for cell in header_raw) + " |"
        md_sep = "| " + " | ".join("---" for _ in header_raw) + " |"
        md_lines.append(md_header)
        md_lines.append(md_sep)

        # --- Блок 2: Вербализация (для embedding поиска) ---
        # Формируем список предложений, описывающих каждую строку.
        # Пример: "Температура, °C: ниже минус 30; Время оттаивания, ч: 5."
        verbalized_rows = []

        for row in rows:
            # Добавляем строку в Markdown
            # Заменяем None на пустую строку
            clean_row_md = [cell if cell else "" for cell in row] 
            line_md = "| " + " | ".join(clean_row_md) + " |"
            md_lines.append(line_md)

            # Формируем вербализацию строки
            # Собираем пары "Заголовок: Значение"
            # Пропускаем пустые ячейки, чтобы не создавать мусор
            row_pairs = []
            for h, cell in zip(clean_headers, row):
                c_text = clean_cell_text(cell)
                if h and c_text:
                    row_pairs.append(f"{h}: {c_text}")
            
            if row_pairs:
                # Объединяем пары в одно предложение
                verbalized_sentence = "; ".join(row_pairs) + "."
                verbalized_rows.append(verbalized_sentence)

        # Собираем всё вместе
        md_table_block = "\n".join(md_lines)
        verbalized_text_block = "\n".join(verbalized_rows)

        # Добавляем в итоговый текст:
        # 1. Заголовок секции
        # 2. Markdown таблицу
        # 3. Явный разделитель и вербализованный текст
        table_output = (
            f"\nТаблица {t_idx + 1} (форматирование Markdown):\n"
            f"{md_table_block}\n\n"
            f"Детальное описание данных Таблицы {t_idx + 1} (построчно):\n"
            f"{verbalized_text_block}\n"
        )
        parts.append(table_output)

    return "\n\n".join(parts)


def read_pdf_with_tables(pdf_path: str) -> str:
    """Считываем PDF, вытаскиваем текст + таблицы в markdown с вербализацией."""
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
        # Сдвигаем окно, но если дошли до конца - выходим
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

    print(f"Читаем PDF (текст + таблицы с вербализацией): {pdf_path}")
    full_text = read_pdf_with_tables(pdf_path)

    print(f"Длина текста (с таблицами): {len(full_text)} символов")
    # Примечание: так как мы добавили вербализацию, текст стал больше.
    # Возможно, стоит немного увеличить max_chars, если строки таблиц очень длинные,
    # чтобы таблица и её описание не разрывались слишком часто.
    chunks = split_text_to_chunks(full_text, max_chars=1500, overlap=300)
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
        # Простой прогресс-бар в консоль
        if i % 10 == 0:
            print(f"Обработка чанка {i+1}/{len(chunks)}...")
            
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