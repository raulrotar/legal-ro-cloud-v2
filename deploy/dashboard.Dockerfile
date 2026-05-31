# Dashboard image — read-only observability. Base core only (no ML).
# Build from repo root:  docker build -f deploy/dashboard.Dockerfile -t legalro-dashboard .
FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY packages/core ./packages/core
COPY packages/dashboard ./packages/dashboard
COPY config ./config

RUN uv sync --package legalro-dashboard --no-dev

ENV CONFIG_PATH=/app/config/cloud.yaml
EXPOSE 7861
CMD ["uv", "run", "legalro-dashboard"]
