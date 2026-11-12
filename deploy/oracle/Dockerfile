# finrag API for the Oracle Always Free host (Ampere A1, arm64; builds on amd64 too).
# Ollama is NOT in this image: it runs in its own container (compose.yml).
#   docker compose -f deploy/oracle/compose.yml up -d --build      (from the repo root)

FROM ghcr.io/astral-sh/uv:0.11 AS uv

FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /srv
ENV UV_PROJECT_ENVIRONMENT=/srv/.venv PATH=/srv/.venv/bin:$PATH \
    FASTEMBED_CACHE_PATH=/models/fastembed PYTHONUNBUFFERED=1

# Dependencies first (cached layer), then the code.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev

# The embedder in the image: a restart never downloads it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY deploy/oracle/server.py deploy/oracle/pull_models.py /srv/deploy/
WORKDIR /srv/deploy
EXPOSE 8000
CMD ["uvicorn", "server:build", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--no-access-log"]
