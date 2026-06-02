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

# Pin HF cache to a fixed path so build-time pre-download and runtime both
# resolve model files from the same location, regardless of which user runs
# the container (root at build time, possibly different at runtime on HF Spaces).
ENV HF_HOME=/app/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf_cache/sentence_transformers

# Pre-download embedding model at build time to avoid cold-start timeout.
# _patch_auto_processor_for_text_models() intercepts the ValueError that
# sentence-transformers >=5.5 raises when loading BAAI/bge-m3 (a text-only
# model that has no AutoProcessor), and returns None instead.
RUN uv run python -c "from legalro_core.embeddings import _patch_auto_processor_for_text_models; _patch_auto_processor_for_text_models(); from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-m3'); print('BGE-M3 pre-download OK, dim=', m.get_sentence_embedding_dimension())"

ENV CONFIG_PATH=/app/config/cloud.yaml
EXPOSE 7860

CMD ["uv", "run", "uvicorn", "legalro_serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
