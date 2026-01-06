# app/rag.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import text as sql_text
from openai import OpenAI

from .db import SessionLocal

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_DB_VECTOR_DIMS: Optional[int] = None
_EMBED_MODEL_RESOLVED: Optional[str] = None

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
    """Смотрим размерность embedding в БД (pgvector: vector_dims)."""
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
    Приоритет:
    1) Если в БД уже есть embeddings -> выбираем модель по dims (чтобы не было mismatch)
    2) Иначе берём OPENAI_EMBEDDING_MODEL (как у тебя в .env)
    3) Иначе EMBED_MODEL
    4) Иначе small
    """
    global _EMBED_MODEL_RESOLVED
    if _EMBED_MODEL_RESOLVED:
        return _EMBED_MODEL_RESOLVED

    dims = _detect_db_vector_dims()
    if dims in DIMS_TO_MODEL:
        _EMBED_MODEL_RESOLVED = DIMS_TO_MODEL[dims]
        print(f"[rag] DB dims={dims} -> embed model={_EMBED_MODEL_RESOLVED}")
        return _EMBED_MODEL_RESOLVED

    env_model = (os.getenv("OPENAI_EMBEDDING_MODEL") or os.getenv("EMBED_MODEL") or "").strip()
    _EMBED_MODEL_RESOLVED = env_model or "text-embedding-3-small"
    print(f"[rag] DB dims not found -> embed model={_EMBED_MODEL_RESOLVED}")
    return _EMBED_MODEL_RESOLVED


def embed_text(text: str) -> List[float]:
    """Публичная функция для ingest.py и для поиска."""
    model = _resolve_embed_model()
    resp = client.embeddings.create(model=model, input=text)
    vec = resp.data[0].embedding

    # Защита от mismatch: если БД уже имеет dims, проверяем
    db_dims = _detect_db_vector_dims()
    if db_dims is not None and len(vec) != db_dims:
        raise RuntimeError(
            f"Embedding dims mismatch: model '{model}' -> {len(vec)}, "
            f"but DB expects {db_dims}. "
            f"Fix: re-ingest DB or set OPENAI_EMBEDDING_MODEL to match."
        )
    return vec


def _to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def search_chunks(query: str, top_k: int = 6, std_pattern: Optional[str] = None) -> List[Chunk]:
    q = (query or "").strip()
    if not q:
        return []

    q_emb = embed_text(q)
    q_emb_vec = _to_pgvector_literal(q_emb)

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
    params = {"q_emb": q_emb_vec, "std_pattern": std_pattern, "limit": limit}

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
