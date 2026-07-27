FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Системные зависимости для pypdf / python-docx / spacy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app
COPY static ./static
COPY tests ./tests
COPY README.md ./
COPY LICENSE ./
COPY .env.example ./

RUN pip install --upgrade pip && pip install .

# Загрузка русской модели spacy (опционально; фолбэк на regex)
RUN python -c "import spacy" 2>/dev/null && \
    python -m spacy download ru_core_news_sm 2>/dev/null || \
    echo "spacy model not downloaded (fallback to regex tokenizer)"

VOLUME ["/app/storage"]
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
