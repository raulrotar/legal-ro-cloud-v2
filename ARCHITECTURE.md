# LegalRo — Architecture Overview

## System map

```
┌──────────────────────────────────────────────────────────────────────────┐
│  USER MACHINE                                                            │
│                                                                          │
│  uv run legalro <command>                                                │
│         │                                                                │
│  cli/app.py (Typer)  ──loads── .env                                     │
│         │                                                                │
│  cli/client.py (httpx)                                                   │
│         │  Authorization: Bearer <HF_TOKEN>   ← HF proxy auth           │
│         │  X-API-Token: <API_TOKEN>            ← app-level auth         │
└─────────┼────────────────────────────────────────────────────────────────┘
          │ HTTPS
          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  HUGGING FACE SPACES  (Docker container, port 7860)                      │
│                                                                          │
│  HF proxy  ──validates── Authorization: Bearer <HF_TOKEN>               │
│         │                                                                │
│  api/app.py (FastAPI)                                                    │
│    _require_auth()  ──validates── X-API-Token: <API_TOKEN>              │
│         │                                                                │
│    POST /query ──────────────────────────────────────────────────────►  │
│    │  embed question  (sentence-transformers nomic-text-v1.5, in RAM)    │
│    │  hybrid_search() ────────────────────────────────────────────────► │──► MongoDB Atlas
│    │  run_query_hybrid() ─────────────────────────────────────────────► │──► Gemini API
│    │  return answer                                                       │
│         │                                                                │
│    POST /ingest ─────────────────────────────────────────────────────►  │
│    │  save PDF to /tmp                                                    │
│    │  return {job_id}  (immediately)                                      │
│    │  background task:                                                    │
│    │    extract_module.run_extraction()  ── scanned pages ─────────────► │──► Mistral OCR API
│    │      → GazetteDocument JSON                                          │
│    │    ingest_module.run_ingestion()                                     │
│    │      → chunk + embed + write ──────────────────────────────────────► │──► MongoDB Atlas
│         │                                                                │
│    GET /jobs/{id}  → poll job status                                     │
└──────────────────────────────────────────────────────────────────────────┘
          │                         │                          │
          ▼                         ▼                          ▼
   MongoDB Atlas M0          Gemini API                 Mistral OCR API
   (vector + BM25)        (gemini-3.1-flash-lite)      (scanned PDFs only)
                         Stage A: native SDK
                         Stage B: OpenAI-compat REST
```

---

## Components

### CLI (`src/legalro/cli/`)

- `app.py` — Typer commands: `query`, `chat`, `ingest`, `status`, `start`, `stop`
- `client.py` — Thin httpx wrapper. Sends two auth headers per request; polls `/jobs/{id}` after ingest

When `LEGALRO_API_URL` is set in `.env`, all commands default to remote mode. Pass `--local` to run in-process.

### API server (`src/legalro/api/app.py`)

FastAPI app running on port 7860 inside the HF Spaces Docker container.

| Endpoint | Auth | Mode | Description |
|---|---|---|---|
| `GET /health` | none | sync | MongoDB ping |
| `POST /query` | required | sync | Embed → search → generate |
| `POST /ingest` | required | async | Upload PDF → background job |
| `GET /jobs/{id}` | required | sync | Poll job status |
| `POST /extract` | required | sync | PDF → JSON (no DB write) |
| `POST /ingest-json` | required | async | JSON → MongoDB (background) |

**Auth:** `X-API-Token` header checked against `API_TOKEN` env var. If `API_TOKEN` is unset, auth is skipped (dev mode).

### Ingestion pipeline (`src/legalro/ingestion/`)

Two-phase design with a JSON handoff:

```
Phase 1 — extract_module.run_extraction()
  PDF → era detection → text extraction → act segmentation → metadata → GazetteDocument → JSON

Phase 2 — ingest_module.run_ingestion()
  JSON → SHA-256 dedup → chunk → embed → MongoDB (gazettes, acts, chunks collections)
```

`pipeline.process_gazette()` is a thin orchestrator that calls both phases in sequence. The `--local` CLI path calls the orchestrator directly; the remote path uploads the PDF and the server runs both phases.

**Era detection** classifies each gazette as one of:
- `SCANNED` — image-only scans; routed to Mistral OCR (cloud) or ocrmac (local)
- `HYBRID` — mix of digital text and scanned facsimile pages
- `BROKEN_2002` / `BROKEN_2007` — mojibake encoding from early digital era
- `MODERN` — clean born-digital PDFs; extracted with PyMuPDF

### Retrieval (`src/legalro/retrieval/`)

Hybrid search combining two MongoDB Atlas indexes on the `chunks` collection:

- `$vectorSearch` — ANN cosine search on 768-dim nomic-embed vectors, top-40 candidates
- `$search` — BM25 full-text search with Romanian analyzer on `text` + `title` fields, top-40 candidates
- RRF merge — Reciprocal Rank Fusion with weights: text 0.7, vector 0.3
- Metadata boost — exact act-type match scores higher

Results are expanded to parent document context before being passed to the LLM.

### Generation (`src/legalro/generation/agent.py`)

Two-stage answering:

- **Stage A** (agentic) — pydantic-ai `Agent` with a `search_law` tool; issues up to 3 retrieval calls, refines query based on intermediate results. Uses `GeminiModel` + `GoogleProvider` (native Gemini SDK) — required for reliable tool-calling support.
- **Stage B** (single-turn) — direct RAG; used as fallback or when `--no-agentic` is passed. Calls the Gemini REST API via OpenAI-compatible endpoint. Also handles MLX locally (Stage A is skipped for MLX since its OpenAI-compat server does not support function calling).

Both stages target `gemini-3.1-flash-lite` in cloud mode.

### Embedding (`src/legalro/providers/embeddings.py`)

`sentence-transformers` with `nomic-ai/nomic-embed-text-v1.5` (768 dimensions, matryoshka). Runs in the HF Spaces container RAM on CPU. Local mode can use MLX embedding models on Apple Silicon.

---

## Data model

```
gazettes collection
  _id, sha256, filename, issue_number, date, source_pdf, ...

acts collection
  _id, gazette_id, act_type, act_number, year, authority, title, full_text, ...

chunks collection
  _id, act_id, gazette_id, text, embedding[768], chunk_index, ...
  indexes: chunks_vector ($vectorSearch), chunks_search_ro ($search BM25)
```

---

## Authentication flow (private HF Space)

```
Client                   HF proxy                 FastAPI app
  │                          │                         │
  │  Authorization: Bearer   │                         │
  │    <HF_TOKEN>  ─────────►│ validates HF token      │
  │  X-API-Token:            │  ──forwards request────►│
  │    <API_TOKEN>           │                         │ checks X-API-Token
  │                          │                         │  vs API_TOKEN env var
  │◄─────────────────────────────────────────────────── response
```

Two separate tokens, two separate headers — no conflict.

---

## Deployment

```
git push origin main
        │
        ▼
GitHub Actions (.github/workflows/deploy.yml)
        │  git push hf main --force  (using HF write token)
        ▼
HF Spaces git repo
        │  detects push → rebuilds Docker image
        ▼
Container running FastAPI on port 7860
        │  reads secrets from HF Space settings
        │  MONGODB_URI, GEMINI_API_KEY, MISTRAL_API_KEY, API_TOKEN
        ▼
Live at https://rraul99-legalro.hf.space
```

Secrets never touch the git repo or the Docker image — they are injected at container start by HF Spaces.
