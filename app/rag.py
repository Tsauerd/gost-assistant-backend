# rag.py
import os
import re
from types import SimpleNamespace
from typing import List, Optional

from sqlalchemy import text as sql_text
from openai import OpenAI

from .db import SessionLocal

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str) -> List[float]:
    """
    Считаем эмбеддинг через text-embedding-3-large.
    """
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding


def embedding_to_pgvector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


def extract_std_pattern(user_query: str) -> Optional[str]:
    """
    Пытаемся вытащить номер ГОСТа/СП из запроса.
    Примеры:
      'ГОСТ 26633-2015' -> '%26633-2015%'
      'ГОСТ 7473' -> '%7473%'
    """
    q_up = user_query.upper()
    m = re.search(r"(ГОСТ|СП)\s*([\d.]+-?\d*)", q_up)
    if not m:
        return None
    num = m.group(2)
    if not num:
        return None
    return f"%{num}%"


def _normalize(value: Optional[float], values: List[float]) -> float:
    """
    Простейшая min-max нормализация.
    """
    if value is None or not values:
        return 0.0
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return 1.0
    return (value - vmin) / (vmax - vmin)


def search_chunks(user_query: str, top_k: int = 6):
    """
    Гибридный поиск:
      - dense по pgvector
      - BM25 по tsvector (ts_rank_cd)
      - фильтрация по ГОСТу, если он явный в запросе
    Возвращает объекты с атрибутами:
      text, section, paragraph, standard_number, year, document_name, final_score, ...
    """
    q_emb = embed_text(user_query)
    q_emb_literal = embedding_to_pgvector_literal(q_emb)
    std_pattern = extract_std_pattern(user_query)

    dense_limit = max(top_k * 3, top_k)

    with SessionLocal() as db:
        # --- Dense-поиск ---
        dense_sql = sql_text("""
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.text,
                c.section,
                c.paragraph,
                d.standard_number,
                d.year,
                d.name AS document_name,
                (1 - (c.embedding <=> :q_emb::vector)) AS dense_score,
                NULL::float AS bm25_score
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE (:std_pattern IS NULL OR d.standard_number ILIKE :std_pattern)
            ORDER BY c.embedding <=> :q_emb::vector
            LIMIT :limit
        """)

        dense_rows = db.execute(
            dense_sql,
            {
                "q_emb": q_emb_literal,
                "std_pattern": std_pattern,
                "limit": dense_limit,
            }
        ).mappings().all()

        # --- Full-text поиск (BM25) ---
        ft_rows = []
        ft_sql = sql_text("""
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.text,
                c.section,
                c.paragraph,
                d.standard_number,
                d.year,
                d.name AS document_name,
                NULL::float AS dense_score,
                ts_rank_cd(
                    c.tsv,
                    plainto_tsquery('russian', :q_text)
                ) AS bm25_score
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.tsv @@ plainto_tsquery('russian', :q_text)
              AND (:std_pattern IS NULL OR d.standard_number ILIKE :std_pattern)
            ORDER BY bm25_score DESC
            LIMIT :limit
        """)

        try:
            ft_rows = db.execute(
                ft_sql,
                {
                    "q_text": user_query,
                    "std_pattern": std_pattern,
                    "limit": dense_limit,
                }
            ).mappings().all()
        except Exception as e:
            # Если колонка tsv ещё не создана — просто пишем в лог и живём на dense
            print(f"[search_chunks] Full-text search disabled (no tsv?): {e}")

    # --- Объединяем кандидатов и считаем финальный score ---
    candidates = {}

    for row in dense_rows:
        cid = row["id"]
        candidates[cid] = dict(row)

    for row in ft_rows:
        cid = row["id"]
        if cid in candidates:
            candidates[cid]["bm25_score"] = row["bm25_score"]
        else:
            candidates[cid] = dict(row)

    if not candidates:
        return []

    dense_vals = [
        r.get("dense_score")
        for r in candidates.values()
        if r.get("dense_score") is not None
    ]
    bm25_vals = [
        r.get("bm25_score")
        for r in candidates.values()
        if r.get("bm25_score") is not None
    ]

    for r in candidates.values():
        nd = _normalize(r.get("dense_score"), dense_vals)
        nb = _normalize(r.get("bm25_score"), bm25_vals)
        # Веса можно будет подстроить по логам
        r["final_score"] = 0.7 * nd + 0.3 * nb

    sorted_rows = sorted(
        candidates.values(),
        key=lambda x: x.get("final_score", 0.0),
        reverse=True
    )

    top = sorted_rows[:top_k]

    # Превращаем dict'ы в объекты с атрибутами .text и т.п.
    return [SimpleNamespace(**row) for row in top]
