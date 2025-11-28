import os
from openai import OpenAI
from sqlalchemy import text as sql_text

from .db import SessionLocal

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


def embed_text(text: str):
    # Можно добавить усечение, если кусок слишком длинный
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding


def _embedding_to_literal(vec):
    # для pgvector: '[0.1,0.2,0.3]'
    return "[" + ",".join(str(round(x, 6)) for x in vec) + "]"


def search_chunks(question: str, top_k: int = 3):
    vec = embed_text(question)
    emb_literal = _embedding_to_literal(vec)

    query = sql_text("""
        SELECT id, text, section, paragraph
        FROM document_chunks
        ORDER BY embedding <-> :emb
        LIMIT :k
    """)

    with SessionLocal() as db:
        rows = db.execute(query, {"emb": emb_literal, "k": top_k}).fetchall()
    return rows
