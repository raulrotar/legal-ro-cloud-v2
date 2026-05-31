# LegalRo Cloud v1 — Implementation Plan

> **Goal:** migrate the local MLX-based RAG system to a fully cloud-hosted, zero-cost stack.
> **Last updated:** 2026-05-24

---

## Architecture

```
Ingestion (runs locally or via GitHub Actions)
  ├── Docling  ──────────────── replaces ocrmac (Linux + macOS, same API)
  ├── sentence-transformers ── replaces mlx-embedding-models (cross-platform)
  │     └── nomic-ai/nomic-embed-text-v1.5  (768-dim, same vectors)
  ├── MongoDB Atlas M0 ──────── chunk text + metadata (free, persistent)
  └── Qdrant Cloud free ──────── dense + sparse vectors for hybrid search

Query API (Hugging Face Spaces — free CPU, always-on)
  ├── FastAPI endpoint  /query  /health
  ├── embed question via sentence-transformers (CPU, ~270 MB model)
  ├── Qdrant hybrid search (dense 768-dim + sparse BM25, server-side RRF)
  ├── fetch chunk metadata from MongoDB Atlas M0
  └── Gemini 2.5 Flash (free: 10 RPM / 1 500 RPD / OpenAI-compatible)
```

### Free tier summary

| Service | What it stores/does | Limit | Cost |
|---|---|---|---|
| MongoDB Atlas M0 | gazettes, acts, chunk text + metadata | 512 MB (~50× current corpus) | Free |
| Qdrant Cloud | dense + sparse vectors for search | 1 GB RAM / 4 GB disk | Free |
| Gemini 2.5 Flash | LLM inference | 10 RPM, 1 500 RPD | Free |
| Hugging Face Spaces (CPU Basic) | FastAPI serving | 2 vCPU, 16 GB RAM, 48 h idle sleep | Free |
| sentence-transformers | embeddings (local + HF Space) | — | Free (open weights) |
| Docling | OCR + extraction | — | Free (Apache 2.0) |

---

## Critical facts (verified 2026-05-24)

- **MongoDB Atlas M0 does NOT support `$vectorSearch` or `$search` (Atlas Search).**
  Both require Flex ($8+/mo) or higher. M0 is used only for document/metadata storage.
- **Qdrant Cloud free** supports native hybrid search (dense + sparse BM25 in one query,
  server-side RRF). Data persists; cluster auto-suspends after 1 week of inactivity
  and is deleted after 4 weeks — must keep alive with occasional pings.
- **Gemini 2.0 Flash is deprecated June 1, 2026.** Use `gemini-2.5-flash`.
  OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`) is
  functional but still in beta per Google's docs.
- **HF Spaces free (CPU Basic)**: 2 vCPU, 16 GB RAM, 50 GB ephemeral disk.
  Sleeps after 48 h inactivity; any request wakes it. No persistent disk on free tier
  (model weights re-download from HF Hub on cold start, ~270 MB, cached in session).
- **nomic-embed-text-v1.5**: `trust_remote_code=True` not needed on
  sentence-transformers >= 5.3.0. CPU speed ~20-60 sentences/sec (2 vCPU).
- **Docling on Linux**: install `docling`; fix `libGL` with `opencv-python-headless`.
  Handles digital and scanned PDFs through the same `DocumentConverter.convert()` call.

---

## Accounts to create (before coding)

1. **MongoDB Atlas** — mongodb.com/atlas → create free M0 cluster → create DB user →
   whitelist `0.0.0.0/0` → copy SRV connection string
2. **Qdrant Cloud** — cloud.qdrant.io → create free cluster → copy URL + API key
3. **Google AI Studio** — aistudio.google.com → Get API key (no billing needed)
4. **Hugging Face** — huggingface.co → create account → create new Space
   (Docker runtime) → add secrets: `MONGODB_URI`, `QDRANT_URL`, `QDRANT_API_KEY`,
   `GEMINI_API_KEY`

---

## What changes vs the local version

| File | Change |
|---|---|
| `pyproject.toml` | Remove `ocrmac`, `mlx-lm`, `mlx-embedding-models`, `einops`; add `sentence-transformers`, `qdrant-client[fastembed]` |
| `config.py` | Add `QdrantConfig`; add `provider: "gemini"` to `LLMConfig`; `api_key` field |
| `config/cloud.yaml` | New cloud config values |
| `providers/embeddings.py` | Add `sentence-transformers` path; keep `mlx` path for local dev |
| `providers/store.py` | Add Qdrant client + upsert/query helpers alongside MongoDB |
| `providers/ocr.py` | Make Docling the default; keep `ocrmac` as local-only fallback |
| `retrieval/search.py` | Replace MongoDB `$vectorSearch`/`$search` pipelines with Qdrant hybrid query |
| `generation/agent.py` | `api_key` read from env; no logic change (OpenAIProvider handles Gemini) |
| `cli/app.py` | Guard MLX `start`/`stop` behind provider check |
| `src/legalro/api/app.py` | Wire up `/query` and `/health` endpoints |
| `Dockerfile` | New — HF Spaces Docker container |
| `scripts/migrate_to_qdrant.py` | New — one-off migration of existing chunks to Qdrant |

---

## Implementation phases

### Phase 1 — Dependencies + Config

**`pyproject.toml`** changes:
```toml
# Remove from dependencies
"ocrmac>=1.0.0"
"mlx-lm>=0.31.1"
"mlx-embedding-models>=0.0.11"
"einops>=0.8.2"

# Add to dependencies
"sentence-transformers>=3.0.0"
"qdrant-client[fastembed]>=1.9.0"

# Move to optional (keep for local dev)
[project.optional-dependencies]
local = ["ocrmac>=1.0.0", "mlx-lm>=0.31.1", "mlx-embedding-models>=0.0.11", "einops>=0.8.2"]
docling = ["docling>=2.0.0"]
```

**`config.py`** additions:
```python
class QdrantConfig(BaseSettings):
    url: str = "http://localhost:6333"
    api_key: str = ""
    collection: str = "chunks"

class LLMConfig(BaseSettings):
    provider: str = "gemini"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str = Field(default="", env="GEMINI_API_KEY")
    model: str = "gemini-2.5-flash"
    max_tokens: int = 8192
    temperature: float = 0.1
    agentic_max_tokens: int = 4096
    agentic_timeout: float = 90.0

class EmbeddingsConfig(BaseSettings):
    provider: str = "sentence-transformers"
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    dimensions: int = 768
    batch_size: int = 32

class Settings(BaseSettings):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
```

**`config/cloud.yaml`** (new file):
```yaml
llm:
  provider: "gemini"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  model: "gemini-2.5-flash"
  api_key: ""        # set via GEMINI_API_KEY env var
  max_tokens: 8192
  temperature: 0.1
  agentic_max_tokens: 4096
  agentic_timeout: 90.0

embeddings:
  provider: "sentence-transformers"
  model: "nomic-ai/nomic-embed-text-v1.5"
  dimensions: 768
  batch_size: 32

ocr:
  provider: "docling"
  language: "ro"

mongodb:
  uri: ""            # set via MONGODB_URI env var
  database: "legalro"

qdrant:
  url: ""            # set via QDRANT_URL env var
  api_key: ""        # set via QDRANT_API_KEY env var
  collection: "chunks"

search:
  limit: 10
  parent_doc_top_n: 3
  rrf_k: 60
```

---

### Phase 2 — Embeddings Provider

**`providers/embeddings.py`** — add sentence-transformers path:

```python
_st_model = None

def _get_st_model(settings):
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(settings.embeddings.model)
    return _st_model

def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    if settings.embeddings.provider == "sentence-transformers":
        model = _get_st_model(settings)
        return model.encode(
            texts,
            batch_size=settings.embeddings.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
    elif settings.embeddings.provider == "mlx":
        # existing path unchanged
        ...
```

`normalize_embeddings=True` matches the local nomic-embed behaviour (cosine distance).

---

### Phase 3 — Qdrant Storage + Search

**`providers/store.py`** — add Qdrant helpers:

```python
_qdrant_client = None

def get_qdrant(settings: Settings):
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key or None,
        )
    return _qdrant_client

def ensure_qdrant_collection(settings: Settings):
    from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams
    client = get_qdrant(settings)
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant.collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant.collection,
            vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )

def upsert_chunks_qdrant(chunks: list[dict], dense_vectors: list[list[float]], settings: Settings):
    from qdrant_client.models import PointStruct
    client = get_qdrant(settings)
    points = [
        PointStruct(
            id=i,
            vector={"dense": dense_vectors[i]},
            payload={
                "mongo_id": str(chunk.get("_id", i)),
                **{k: chunk.get(k) for k in [
                    "text", "law_id", "act_number", "act_year", "source_issue_id",
                    "document_type", "title", "act_full_text", "full_path",
                    "act_index_in_issue", "issuing_authority", "locality",
                ]},
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=settings.qdrant.collection, points=points)
```

**`retrieval/search.py`** — replace MongoDB pipelines with Qdrant hybrid query:

```python
def hybrid_search(query: str, settings: Settings, act_type: str | None = None) -> list[dict]:
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery
    from legalro.providers.store import get_qdrant

    query_embedding = embed_texts([query], settings)[0]
    client = get_qdrant(settings)

    qdrant_filter = None
    if act_type:
        qdrant_filter = Filter(
            must=[FieldCondition(key="document_type", match=MatchValue(value=act_type.upper()))]
        )

    # Dense + sparse BM25, server-side RRF
    results = client.query_points(
        collection_name=settings.qdrant.collection,
        prefetch=[
            Prefetch(query=query_embedding, using="dense", limit=40),
            Prefetch(query=query, using="sparse", limit=40),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=settings.search.limit,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    docs = []
    for point in results.points:
        doc = dict(point.payload)
        doc["rrf_score"] = point.score
        doc["_id"] = doc.pop("mongo_id", str(point.id))
        docs.append(doc)

    return _apply_metadata_boost(docs, _parse_query_metadata(query))
```

Note: Qdrant sparse BM25 via `using="sparse"` with a text string requires the
`fastembed` integration (included via `qdrant-client[fastembed]`). On first use it
downloads a small (~23 MB) SPLADE model.

---

### Phase 4 — OCR Provider

**`providers/ocr.py`** — Docling default, ocrmac as local fallback:

```python
def extract_text(pdf_path: str, settings) -> str:
    if settings.ocr.provider == "docling":
        return _extract_docling(pdf_path)
    elif settings.ocr.provider == "ocrmac":
        return _extract_ocrmac(pdf_path, settings)
    raise ValueError(f"Unknown OCR provider: {settings.ocr.provider}")

def _extract_docling(pdf_path: str) -> str:
    from docling.document_converter import DocumentConverter
    result = DocumentConverter().convert(pdf_path)
    return result.document.export_to_text()
```

Linux fix: replace `opencv-python` with `opencv-python-headless` in the Dockerfile
to avoid `libGL.so.1` errors.

---

### Phase 5 — LLM Provider (Gemini)

No code change to `generation/agent.py` — `OpenAIProvider` with configurable
`base_url` already works as a Gemini client. Only the config drives it:

```yaml
llm:
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  model: "gemini-2.5-flash"
  api_key: ""   # GEMINI_API_KEY env var
```

The `chat_template_kwargs: {"enable_thinking": False}` field in Stage B is
MLX-specific and silently ignored by Gemini — no issue.

---

### Phase 6 — FastAPI App

**`src/legalro/api/app.py`**:

```python
import os
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from legalro.config import load_settings
from legalro.generation.agent import run_query_hybrid
from legalro.providers.store import get_qdrant

settings = load_settings(os.getenv("CONFIG_PATH", "config/cloud.yaml"))

@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(_keep_qdrant_alive())
    yield

app = FastAPI(title="LegalRo API", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str
    act_type: str = ""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest):
    try:
        return {"answer": run_query_hybrid(req.question, settings)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _keep_qdrant_alive():
    """Ping Qdrant every 6 h to prevent free cluster suspension."""
    while True:
        await asyncio.sleep(6 * 3600)
        try:
            get_qdrant(settings).get_collections()
        except Exception:
            pass
```

---

### Phase 7 — Dockerfile

```dockerfile
FROM python:3.12-slim

# Docling / OpenCV on Linux needs libGL
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev

COPY src/ ./src/
COPY config/cloud.yaml ./config/

ENV PYTHONPATH=/app/src
EXPOSE 7860

CMD ["uv", "run", "uvicorn", "legalro.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

HF Spaces injects secrets as env vars. The `MongoDBConfig` and `QdrantConfig` fields
must read from env:
```python
class MongoDBConfig(BaseSettings):
    uri: str = Field(default="", env="MONGODB_URI")
    database: str = "legalro"

class QdrantConfig(BaseSettings):
    url: str = Field(default="", env="QDRANT_URL")
    api_key: str = Field(default="", env="QDRANT_API_KEY")
    collection: str = "chunks"
```

---

### Phase 8 — Migration Script

**`scripts/migrate_to_qdrant.py`** — one-off: reads chunks from MongoDB, upserts
to Qdrant (run locally against cloud services):

```python
from legalro.config import load_settings
from legalro.providers.store import get_db, ensure_qdrant_collection, upsert_chunks_qdrant
from legalro.providers.embeddings import embed_batch

settings = load_settings("config/cloud.yaml")
ensure_qdrant_collection(settings)
chunks = list(get_db(settings).chunks.find({}))
print(f"Migrating {len(chunks)} chunks...")
embeddings = embed_batch([c["text"] for c in chunks], settings)
upsert_chunks_qdrant(chunks, embeddings, settings)
print("Done.")
```

---

### Phase 9 — CLI Updates

**`cli/app.py`** — guard MLX-specific commands:

```python
@app.command()
def start():
    """Start local services. No-op in cloud mode."""
    settings = load_settings()
    if settings.llm.provider != "mlx":
        typer.echo("Cloud mode — no local services to start.")
        return
    # existing MLX start logic...
```

---

## Verification plan

1. Set env vars locally, point config to cloud services:
   ```bash
   export GEMINI_API_KEY=... MONGODB_URI=... QDRANT_URL=... QDRANT_API_KEY=...
   uv run python scripts/migrate_to_qdrant.py
   uv run python test_questions.py --config config/cloud.yaml
   ```
2. Compare QA results to local baseline (22/28 CORECT, 0 EROARE target)
3. Build and test Docker image locally:
   ```bash
   docker build -t legalro-cloud .
   docker run -p 7860:7860 --env-file .env legalro-cloud
   curl -X POST localhost:7860/query -d '{"question":"Ce este un HG?"}'
   ```
4. Deploy to HF Spaces — add secrets in Space settings, push Dockerfile
5. Confirm `/health` returns `{"status":"ok"}` and `/query` answers correctly

---

## Risk register

| Risk | Mitigation |
|---|---|
| Gemini 2.5 Flash OpenAI endpoint still in beta | Fall back to `google-generativeai` SDK if pydantic-ai compat breaks |
| Qdrant cluster deleted after 4 weeks inactivity | Keep-alive ping every 6 h from HF Space |
| HF Space cold start: model download ~270 MB | First request ~30 s; acceptable for POC |
| sentence-transformers CPU slow (~20-60 sent/sec) | Batch queries; fine for low-traffic POC |
| Gemini 1 500 RPD limit hit | Log usage; switch primary to Groq Llama 3.3 70B as overflow |
| Qdrant fastembed SPLADE model download on first start | Pre-warm in Dockerfile or accept cold-start delay |
