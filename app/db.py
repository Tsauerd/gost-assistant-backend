import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL не найден в переменных окружения.")

# SQLAlchemy не понимает postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SSL: для Neon обычно нужен require. Чтобы не ломать локальный postgres —
# можно управлять через переменную DB_SSLMODE.
connect_args = {}
sslmode = os.getenv("DB_SSLMODE")  # например: require / disable
if sslmode:
    connect_args["sslmode"] = sslmode
else:
    # авто-эвристика: если Neon — включаем require
    if "neon.tech" in DATABASE_URL or "neon" in DATABASE_URL:
        connect_args["sslmode"] = "require"

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,   # лечит "stale connections"
    pool_recycle=280,     # полезно на хостингах
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def ensure_schema() -> None:
    """
    Создаёт таблицу requests если её нет.
    Это важно: если таблицы нет/не та — request_id всегда будет null и рейтинг не будет работать.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS requests (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT now(),
        query_text TEXT NOT NULL,
        task_type TEXT,
        model_used TEXT,
        tokens_in INT,
        tokens_out INT,
        cost_usd DOUBLE PRECISION,
        answer_text TEXT,
        client_id TEXT,
        user_agent TEXT,
        status TEXT,
        error_text TEXT,
        rating INT,
        rating_created_at TIMESTAMPTZ
    );
    """
    with engine.begin() as conn:
        conn.execute(sql_text(ddl))
