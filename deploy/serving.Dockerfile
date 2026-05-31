# Serving image — tiny, read-only query API. NO docling/OCR.
# Build from repo root:  docker build -f deploy/serving.Dockerfile -t legalro-serving .
FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app

# Copy the workspace metadata + the packages serving needs (core + serving).
COPY pyproject.toml uv.lock* ./
COPY packages/core ./packages/core
COPY packages/serving ./packages/serving
COPY config ./config

# Install ONLY the serving member (uv resolves the workspace; pulls core, not processing).
RUN uv sync --package legalro-serving --no-dev

ENV CONFIG_PATH=/app/config/cloud.yaml
EXPOSE 7860
CMD ["uv", "run", "uvicorn", "legalro_serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
