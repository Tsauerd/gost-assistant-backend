# rag.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from sqlalchemy import text as sql_text
from openai import OpenAI

from .db import SessionLocal

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Модели и размерности (важно держать одинаковыми для ingestion и поиска)
MODEL_TO_DIMS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
DIMS_TO_MODEL: Dict[int, str] = {v: k for k, v in MODEL_TO_DIMS.items()}

# Кэш, чтобы не ходить в БД каждый запрос
_DB_VECTOR_DIMS: Optional[int] = None
_EMBED_MODEL_RESOLVED: Optional[str] = None


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


def _detect_db_vector_dims_from_column() -> Optional[int]:
    """
    Пытаемся прочитать тип колонки embedding через format_type:
    вернет 'vector(1536)' или 'vector(3072)' или просто 'vector'.
    """
    q = sql_text("""
        SELECT format_type(a.atttypid, a.atttypmod) AS typ
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE c.relname = 'document_chunks'
          AND a.attname = 'embedding'
          AND n.nspname = current_schema()
        LIMIT 1
    """)

    try:
        with SessionLocal() as db:
            row = db.execute(q).mappings().first()
            typ = (row.get("typ") if row else None) or ""
            m = re.search(r"vector\((\d+)\)", typ)
            return int(m.group(1)) if m else None
    except Exception as e:
        print(f"[rag] failed to detect column vector dims: {e}")
        return None


def _detect_db_vector_dims_from_rows() -> Optional[int]:
    """
    Берем размерность из существующих строк через vector_dims(embedding).
    Работает даже если тип колонки просто 'vector' без фиксированной размерности.
    """
    q = sql_text("""
        SELECT vector_dims(embedding) AS dims
        FROM document_chunks
        WHERE embedding IS NOT NULL
        LIMIT 1
    """)

    try:
        with SessionLocal() as db:
            row = db.execute(q).mappings().first()
            return int(row["dims"]) if row and row.get("dims") else None
    except Exception as e:
        print(f"[rag] failed to detect row vector dims: {e}")
        return None


def _detect_db_vector_dims() -> Optional[int]:
    global _DB_VECTOR_DIMS
    if _DB_VECTOR_DIMS is not None:
        return _DB_VECTOR_DIMS

    # 1) сначала пробуем из типа колонки (лучше всего)
    dims = _detect_db_vector_dims_from_column()
    # 2) если не получилось — из данных
    if dims is None:
        dims = _detect_db_vector_dims_from_rows()

    _DB_VECTOR_DIMS = dims
    return dims


def _resolve_embed_model() -> str:
    """
    Выбор модели для эмбеддинга запроса (и общего embed_text):
    - если в БД уже есть размерность, лучше совпасть с ней
    - иначе берем OPENAI_EMBEDDING_MODEL (или EMBED_MODEL), иначе small
    """
    global _EMBED_MODEL_RESOLVED
    if _EMBED_MODEL_RESOLVED:
        return _EMBED_MODEL_RESOLVED

    env_model = (os.getenv("OPENAI_EMBEDDING_MODEL") or os.getenv("EMBED_MODEL") or "").strip() or None
    db_dims = _detect_db_vector_dims()
    db_model = DIMS_TO_MODEL.get(db_dims) if db_dims else None

    if db_model and env_model and (MODEL_TO_DIMS.get(env_model) != db_dims):
        # безопаснее жить в проде: подстраиваемся под БД, иначе будут ошибки сравнения векторов
        print(f"[rag] WARNING: env embed model={env_model} "
              f"does not match DB dims={db_dims}. Using DB-matched model={db_model}")
        _EMBED_MODEL_RESOLVED = db_model
        return _EMBED_MODEL_RESOLVED

    if env_model:
        _EMBED_MODEL_RESOLVED = env_model
        return _EMBED_MODEL_RESOLVED

    if db_model:
        _EMBED_MODEL_RESOLVED = db_model
        return _EMBED_MODEL_RESOLVED

    _EMBED_MODEL_RESOLVED = "text-embedding-3-small"
    return _EMBED_MODEL_RESOLVED


def _embed(text: str, model: Optional[str] = None) -> List[float]:
    m = model or _resolve_embed_model()
    resp = client.embeddings.create(model=m, input=text)
    return resp.data[0].embedding


def embed_text(text: str, model: Optional[str] = None) -> List[float]:
    """
    Публичная функция для ingestion:
    ingest.py импортирует именно её.
    """
    return _embed(text, model=model)


def _to_pgvector_literal(vec: List[float]) -> str:
    # pgvector умеет читать строку вида [0.1,0.2,...]
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def search_chunks(query: str, top_k: int = 6, std_pattern: Optional[str] = None) -> List[Chunk]:
    q = (query or "").strip()
    if not q:
        return []

    # 1) Эмбеддинг запроса
    q_emb = _embed(q)
    q_dims = len(q_emb)
    q_emb_vec = _to_pgvector_literal(q_emb)

    # 2) ВАЖНО: фильтруем по vector_dims(c.embedding) = q_dims,
    #    чтобы БД НЕ падала, если внутри есть чанки другой размерности (1536/3072).
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
        WHERE c.embedding IS NOT NULL
          AND vector_dims(c.embedding) = :q_dims
          AND (:std_pattern IS NULL OR d.standard_number ILIKE :std_pattern)
        ORDER BY c.embedding <=> (:q_emb)::vector
        LIMIT :limit
    """)

    limit = max(top_k * 3, top_k)
    params = {
        "q_emb": q_emb_vec,
        "q_dims": q_dims,
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
