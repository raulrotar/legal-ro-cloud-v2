FROM python:3.12-slim

# Docling / OpenCV on Linux needs libGL
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# Copy source
COPY src/ ./src/
COPY config/ ./config/

# Pre-download embedding models at build time to avoid cold-start timeouts.
# nomic is used in prod (cloud.yaml); bge-m3 is used in staging (staging.yaml).
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5'); SentenceTransformer('BAAI/bge-m3')"

ENV PYTHONPATH=/app/src
# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uv", "run", "uvicorn", "legalro.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
