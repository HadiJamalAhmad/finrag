# finrag API and ingestion, one image (deploy/k8s runs it as a Deployment and as Jobs).
#
#   docker build -t finrag:local .
#   docker run -p 8000:8000 -e FINRAG_QDRANT_URL=http://... finrag:local
#
# Built to run where nothing may be downloaded at run time (an on-premises cluster):
# the embedding model is baked into the image and Hugging Face is set offline, so a
# pod that tried to fetch it would fail loudly instead of reaching the internet.
# The LLM weights are not in this image: Ollama is its own service (deploy/k8s/ollama).

FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
# The embedder (bge-small-en-v1.5, ONNX), fetched once here and never at run time.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed
RUN /app/.venv/bin/python -c \
    "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
    && chmod -R a+rX /opt/fastembed  # the hub writes some cache files 0600; we run as 10001

FROM python:3.12-slim-bookworm
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin finrag
WORKDIR /app
COPY --from=build /app /app
COPY --from=build /opt/fastembed /opt/fastembed
# The model pins (finrag pull-model checks the digest against src/finrag/models.py).
COPY deploy/models.lock /app/deploy/models.lock
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_OFFLINE=1 \
    FINRAG_DATA=/data
USER 10001
EXPOSE 8000
CMD ["uvicorn", "finrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
