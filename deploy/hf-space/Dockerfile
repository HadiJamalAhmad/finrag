# finrag on a free Hugging Face Space (Docker, CPU). The finrag code is installed at a
# pinned commit; the 45 frozen sheets are indexed at build time with the frozen French
# settings, so the Space answers from its first request and downloads nothing at run time.
FROM python:3.12-slim-trixie

ARG FINRAG_COMMIT=4d728ba
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home user
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

WORKDIR /app
RUN git clone -q https://github.com/HadiJamalAhmad/finrag.git . \
    && git checkout -q "$FINRAG_COMMIT" && rm -rf .git \
    && uv sync --frozen --no-dev

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    FINRAG_DATA=/app/eval/fixtures/fr-interieur \
    FINRAG_DOMAIN=documents \
    FINRAG_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    FINRAG_COLLECTION=docs_fr \
    FINRAG_MIN_SCORE=0.61 \
    FINRAG_TOP_K=3

RUN finrag ingest && chown -R user /app/eval/fixtures/fr-interieur /opt/fastembed
ENV HF_HUB_OFFLINE=1

COPY app.py index.html /app/space/
USER user
EXPOSE 7860
CMD ["uvicorn", "app:app", "--app-dir", "/app/space", "--host", "0.0.0.0", "--port", "7860"]
