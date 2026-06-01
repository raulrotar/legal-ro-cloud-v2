# HF Spaces entry point — builds and runs the serving package only.
# The processing and dashboard packages are not included in this image.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY packages/core ./packages/core
COPY packages/serving ./packages/serving
COPY config ./config

RUN uv sync --package legalro-serving --no-dev

# Pre-download embedding model at build time to avoid cold-start timeout
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

ENV CONFIG_PATH=/app/config/cloud.yaml
EXPOSE 7860

CMD ["uv", "run", "uvicorn", "legalro_serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
