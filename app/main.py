from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text as sql_text
from typing import Optional

# Импорты из соседних файлов
# Предполагается, что эти файлы существуют в структуре проекта
from .db import SessionLocal
from .rag import search_chunks
from .llm import call_llm

app = FastAPI(title="GOST Assistant Backend")

# --- CORS: Разрешаем запросы с любого источника (для тестов) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    # Нормально по умолчанию считать, что ищем норму/пункт ГОСТ
    task_type: Optional[str] = "norm"  # norm / procedure / calculation / complaint / letter
    # model_version убрали, теперь решаем внутри

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "gost-assistant-backend"}

def resolve_model_name(task_type: Optional[str]) -> str:
    """
    Выбираем модель тихо, без упоминания во фронте.
    - Для писем/претензий: gpt-4o (более сильная, аналог gpt-4.1 в вашей классификации)
    - Для всего остального: gpt-4o-mini
    """
    tt = (task_type or "").lower()

    # Любые варианты, связанные с письмами/претензиями/юридическими вопросами
    if tt in ("complaint", "claim", "letter", "legal", "complaint_letter"):
        return "gpt-4o"

    # Дефолт для поиска норм и простых ответов
    return "gpt-4o-mini"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.query
    
    # 1. RAG: Ищем больше кусков (top_k=6), чтобы захватить таблицы целиком
    chunks = search_chunks(user_query, top_k=6)

    context_text = "\n\n".join([f"Отрывок: {row.text}" for row in chunks])
    if not context_text.strip():
        context_text = "Нет конкретной информации в базе знаний."

    # 2. Формируем строгий системный промпт
    full_prompt = f"""
Ты — профессиональный технический эксперт и юрист, специализирующийся на стандартах ГОСТ, СНиП и работе с рекламациями.

Твои задачи:
1. Давать точные ответы по нормативам.
2. Помогать составлять юридически грамотные тексты претензий, актов или ответов на жалобы.

Инструкции по работе с контекстом:
1. Отвечай СТРОГО на основе приведённого ниже контекста. Не придумывай нормы, которых нет в базе.
2. ТАБЛИЦЫ: Если в контексте есть таблицы (Markdown), обязательно учитывай их. Если данные в таблице точнее текста — таблица в приоритете.
3. ЧИСЛА И УСЛОВИЯ: Внимательно проверяй условия (температуру, размеры, марки). Если в вопросе есть условия (например, "при -30 градусах"), ищи соответствующее значение в таблицах.

Инструкции по стилю (если вопрос подразумевает составление документа):
1. Используй официально-деловой стиль.
2. Структурируй ответ:
   - **Нормативное обоснование**: Цитата конкретного пункта ГОСТ (с номером пункта/таблицы).
   - **Анализ ситуации**: Сравнение требований ГОСТ с ситуацией пользователя.
   - **Вывод/Рекомендация**: Четкое заключение (соответствует/не соответствует).
3. При составлении претензии используй формулировки: "Согласно п. X ГОСТ Y...", "На основании вышеизложенного требуем...", "Отклонение от нормы составляет...".

Форматирование:
- Используй жирный шрифт для выделения ключевых выводов и номеров пунктов.
- Формулы пиши в LaTeX ($$ ... $$).
- Таблицы выводи в Markdown.

Контекст из базы знаний:
{context_text}

Запрос пользователя:
{user_query}
    """

    # 3. Выбираем модель на основе типа задачи (тихо)
    model_name = resolve_model_name(request.task_type)
    answer, usage = call_llm(full_prompt, model=model_name)

    # 4. Безопасный подсчет токенов и стоимости
    tokens_in = 0
    tokens_out = 0

    if usage:
        # Проверяем атрибуты, так как разные версии либы могут называть их по-разному
        tokens_in = getattr(usage, "prompt_tokens", 0)
        tokens_out = getattr(usage, "completion_tokens", 0)

    # Примерные цены (на текущий момент):
    # gpt-4o-mini: ~$0.15 вход / $0.60 выход за 1М токенов
    # gpt-4o:      ~$2.50 вход / $10.00 выход за 1М токенов
    if "mini" in model_name:
        in_price = 0.15
        out_price = 0.60
    else:
        # Цены для "большой" модели
        in_price = 2.50
        out_price = 10.00

    cost = (tokens_in * in_price + tokens_out * out_price) / 1_000_000

    # 5. Логируем запрос в базу (таблица requests)
    # Обрати внимание: сохраняем request.task_type, а не версию модели из запроса (ее больше нет)
    log_query = sql_text("""
        INSERT INTO requests (query_text, task_type, model_used, tokens_in, tokens_out, cost_usd, status)
        VALUES (:q, :tt, :m, :ti, :to, :c, 'success')
    """)

    try:
        with SessionLocal() as db:
            db.execute(log_query, {
                "q": user_query,
                "tt": request.task_type,
                "m": model_name,
                "ti": tokens_in,
                "to": tokens_out,
                "c": cost
            })
            db.commit()
    except Exception as e:
        print(f"Ошибка логирования в БД: {e}")

    return {
        "answer": answer,
        "model_used": model_name, # Для отладки можно оставить, но фронт может это игнорировать
        "context_used": [row.text for row in chunks]
    }