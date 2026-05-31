# Processing image — heavy batch pipeline (Docling, OCR, embedder). Runs on VPS.
# Build from repo root:  docker build -f deploy/processing.Dockerfile -t legalro-processing .
FROM python:3.12-slim

# System deps Docling/PyMuPDF/OCR may need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ghostscript \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY packages/core ./packages/core
COPY packages/processing ./packages/processing
COPY config ./config

RUN uv sync --package legalro-processing --no-dev

ENV CONFIG_PATH=/app/config/vps.yaml
ENTRYPOINT ["uv", "run", "legalro-process"]
CMD ["--help"]
