from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import text as sql_text

from .db import SessionLocal
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Кэш, чтобы не ходить в БД каждый запрос
_DB_VECTOR_DIMS: Optional[int] = None
_EMBED_MODEL_RESOLVED: Optional[str] = None

# Маппинг размерности -> модель
DIMS_TO_MODEL = {
    1536: "text-embedding-3-small",
    3072: "text-embedding-3-large",
}


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


def _detect_db_vector_dims() -> Optional[int]:
    """
    Возвращает размерность embedding в БД (например 1536).
    Нужна функция pgvector vector_dims(vector).
    """
    global _DB_VECTOR_DIMS

    if _DB_VECTOR_DIMS is not None:
        return _DB_VECTOR_DIMS

    q = sql_text("""
        SELECT vector_dims(embedding) AS dims
        FROM document_chunks
        WHERE embedding IS NOT NULL
        LIMIT 1
    """)

    try:
        with SessionLocal() as db:
            row = db.execute(q).mappings().first()
            _DB_VECTOR_DIMS = int(row["dims"]) if row and row.get("dims") else None
            return _DB_VECTOR_DIMS
    except Exception as e:
        print(f"[rag] failed to detect db vector dims: {e}")
        return None


def _resolve_embed_model() -> str:
    """
    1) Смотрим dims в БД
    2) Подбираем модель
    3) Если не получилось — берём EMBED_MODEL из env или small по умолчанию
    """
    global _EMBED_MODEL_RESOLVED

    if _EMBED_MODEL_RESOLVED:
        return _EMBED_MODEL_RESOLVED

    dims = _detect_db_vector_dims()
    if dims in DIMS_TO_MODEL:
        _EMBED_MODEL_RESOLVED = DIMS_TO_MODEL[dims]
        print(f"[rag] detected DB vector dims={dims}, using embed model={_EMBED_MODEL_RESOLVED}")
        return _EMBED_MODEL_RESOLVED

    env_model = os.getenv("EMBED_MODEL")
    _EMBED_MODEL_RESOLVED = env_model or "text-embedding-3-small"
    print(f"[rag] could not detect DB dims, using embed model={_EMBED_MODEL_RESOLVED}")
    return _EMBED_MODEL_RESOLVED


def _embed(text: str) -> List[float]:
    model = _resolve_embed_model()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def _to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def search_chunks(query: str, top_k: int = 6, std_pattern: Optional[str] = None) -> List[Chunk]:
    q = (query or "").strip()
    if not q:
        return []

    # 1) Эмбеддинг запроса
    q_emb = _embed(q)
    q_emb_vec = _to_pgvector_literal(q_emb)

    # 2) DENSE поиск
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

    limit = max(top_k * 3, top_k)

    params = {
        "q_emb": q_emb_vec,
        "std_pattern": std_pattern,
        "limit": limit,
    }

    with SessionLocal() as db:
        rows = db.execute(dense_sql, params).mappings().all()

    out: List[Chunk] = []
    for r in rows[:top_k]:
        out.append(
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
    return out
