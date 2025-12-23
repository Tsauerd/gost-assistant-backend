from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import text as sql_text

from .db import SessionLocal

# Если эмбеддинги у тебя делаются в другом месте — подставь свою функцию.
from openai import OpenAI

_OPENAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


@dataclass
class Chunk:
    id: int
    document_id: int
    chunk_index: int
    text: str
    section: Optional[str] = None
    paragraph: Optional[str] = None
    standard_number: Optional[str] = None
    year: Optional[int] = None
    document_name: Optional[str] = None
    dense_score: Optional[float] = None


def _embed(text: str) -> List[float]:
    resp = _OPENAI.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _to_pgvector_literal(vec: List[float]) -> str:
    # pgvector принимает строку вида: [0.1,0.2,...]
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def search_chunks(query: str, top_k: int = 6, std_pattern: Optional[str] = None) -> List[Chunk]:
    q = (query or "").strip()
    if not q:
        return []

    # 1) Эмбеддинг
    try:
        q_emb = _embed(q)
    except Exception as e:
        print(f"[rag] embedding failed: {e}")
        return []

    q_emb_vec = _to_pgvector_literal(q_emb)

    # 2) DENSE поиск через pgvector
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
            (1 - (c.embedding <=> (:q_emb)::vector)) AS dense_score
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE (:std_pattern IS NULL OR d.standard_number ILIKE :std_pattern)
        ORDER BY c.embedding <=> (:q_emb)::vector
        LIMIT :limit
    """)

    # Немного запас по кандидатам
    limit = max(top_k * 3, top_k)

    params = {
        "q_emb": q_emb_vec,                 # строка "[..]" -> кастуется в vector
        "std_pattern": std_pattern,
        "limit": limit,
    }

    with SessionLocal() as db:
        rows = db.execute(dense_sql, params).mappings().all()

    chunks: List[Chunk] = []
    for r in rows[:top_k]:
        chunks.append(
            Chunk(
                id=r["id"],
                document_id=r["document_id"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                section=r.get("section"),
                paragraph=r.get("paragraph"),
                standard_number=r.get("standard_number"),
                year=r.get("year"),
                document_name=r.get("document_name"),
                dense_score=float(r["dense_score"]) if r.get("dense_score") is not None else None,
            )
        )

    return chunks
