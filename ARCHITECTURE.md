# LegalRo — Architecture Overview

## System map

```
┌──────────────────────────────────────────────────────────────────────────┐
│  VPS / LOCAL MACHINE  (processing)                                        │
│                                                                           │
│  legalro-process extract                                                  │
│    PDF → Markdown  (GLM-OCR for scanned era, Docling otherwise)          │
│                    cached in md_cache/ (sha256-keyed)                    │
│    MD  → MdActBlock list  (md_segmenter)                                  │
│    Block → GazetteDocument JSON  (md_rule_extractor regex + optional     │
│                    vision repair pass; cached in extracted/)              │
│    JSON → chunks → bge-m3 embed  (build.py)                              │
│    Emit on-disk bundle  (bundle_writer.py → out/)                        │
│         │                                                                 │
│  legalro-process load                                                     │
│    Bundle → MongoDB Atlas ──────────────────────────────────────────────►│
└──────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                   MongoDB Atlas M0
                               (gazettes + chunks + graph_edges)
                                          │
┌──────────────────────────────────────── ▼ ────────────────────────────────┐
│  HUGGING FACE SPACES  (serving package, Docker, port 7860)                │
│                                                                           │
│  HF proxy  ──validates── Authorization: Bearer <HF_TOKEN>               │
│         │                                                                 │
│  FastAPI app  (legalro_serving.app)                                      │
│    _require_auth()  ──validates── X-API-Token: <API_TOKEN>              │
│         │                                                                 │
│    POST /query ──────────────────────────────────────────────────────►   │
│    │  embed question (bge-m3, in-container RAM)                           │
│    │  _python_rrf_search():                                               │
│    │    $vectorSearch (cosine, 1024-dim, top-80 candidates) ────────────►│── MongoDB Atlas
│    │    $search (BM25 Romanian analyzer, top-80)            ────────────►│
│    │    Python RRF merge + metadata boost                                │
│    │  run_query_hybrid():                                                 │
│    │    Stage A — pydantic-ai Agent (GoogleModel/Gemini, tool-calling) ──►│── Gemini API
│    │    Stage B — single-turn RAG fallback (httpx, OpenAI-compat)  ─────►│
│    │  return QueryResponse {answer, sources, latency_ms, chunks_used}    │
│         │                                                                 │
│    GET /acts        — paginated act listing (aggregation on chunks)       │
│    GET /acts/{id}   — single act detail                                   │
│    GET /health      — MongoDB ping                                        │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Packages

### `legalro-core` (`packages/core/`)

Shared contract imported by all other packages. Contains no ML-heavy dependencies.

| Module | Role |
|---|---|
| `config.py` | `Settings` dataclass; YAML + env override loader |
| `schema.py` | Version stamps, collection names, deterministic `_id` derivation |
| `models.py` | `Era` enum: `SCANNED / HYBRID / MODERN / BROKEN_2002 / BROKEN_2007` |
| `store.py` | MongoDB connection pool (`get_db`) |
| `embeddings.py` | bge-m3 embed wrapper; `embed_texts` / `embed_batch` |
| `llm_client.py` | Shared OpenAI-compat LLM client (Gemini / Ollama) |
| `normalize.py` | Diacritics, mojibake repair, BM25 text normalization |
| `md_normalize.py` | Markdown-specific normalization helpers |
| `act_number.py` | Act number parsing and canonicalisation |
| `bundle.py` | On-disk bundle read helpers (manifest, checksums, JSONL.gz) |
| `retrieval/search.py` | Hybrid search: `$vectorSearch` + `$search` + Python RRF + metadata boost |
| `retrieval/context.py` | Parent-doc expansion + context string assembly |

### `legalro-processing` (`packages/processing/`)

VPS/batch package. Handles the entire extraction → embedding → load pipeline.

**CLI: `legalro-process`**

| Command | Description |
|---|---|
| `extract` | Stage A: PDF → bundle (full pipeline) |
| `extract-json` | PDF → GazetteDocument JSON only (no embeddings) |
| `load` | Stage B: bundle → MongoDB |
| `setup-indexes` | Create/update Atlas Search + vector indexes |
| `reset-db` | Drop all collections |

**Extraction pipeline:**

```
Primary path (deterministic, no LLM):
  PDF → era detection
    SCANNED → GLM-OCR (glm-ocr:latest) → Markdown → md_cache/
    other   → Docling                  → Markdown → md_cache/
  Markdown → MdActBlock list (md_segmenter, era-aware thresholds)
  MdActBlock → md_rule_extractor (regex) → rule_draft fields
            → secondary_analyzer (PyMuPDF closing-sig recovery)
            → sumar positional fallback
  → GazetteDocument → extracted/

Repair pass (on-demand, flagged acts only):
  Acts failing inline validation (ACT_NUMBER_ZERO, DOC_TYPE_UNKNOWN, …)
  → llm_repair.py (llama3.2-vision:11b / glm-ocr) patches only flagged fields
```

**Era detection** classifies each gazette as:
- `SCANNED` — image-only scans (pre-1997); routed to GLM-OCR
- `HYBRID` — mix of digital text and scanned facsimile pages
- `BROKEN_2002` / `BROKEN_2007` — mojibake encoding from early digital era; Docling visual render recovers glyphs
- `MODERN` — clean born-digital PDFs; extracted with Docling

**Build stage (`prepare/build.py`):**
- Deterministic `_id` / `chunk_id` / `act_id` from `schema.py`
- Modality + embedding_version + publication_date facets on every chunk
- `act_full_text` capped at 12 000 chars for parent-doc retrieval

**Load stage (`loader.py`):**
- Idempotent: `UpdateOne` upserts via deterministic `_id`s
- Resumable: `.load_state.json` tracks loaded docs
- Works identically against Atlas Local (Docker) and Atlas M0

### `legalro-serving` (`packages/serving/`)

Read-only FastAPI app deployed to HF Spaces. Imports only `legalro-core` — no Docling, no embedding batch code.

| Endpoint | Auth | Mode | Description |
|---|---|---|---|
| `GET /health` | none | sync | MongoDB ping |
| `POST /query` | required | sync | Embed → hybrid search → Gemini → answer |
| `GET /acts` | required | sync | Paginated act listing with filters |
| `GET /acts/{id}` | required | sync | Single act detail |

**Generation (`generation.py`):**

Two-stage answering:
1. **Stage A (agentic)** — pydantic-ai `Agent` with `search_law` tool; up to 3 retrieval calls; uses `GoogleModel` + `GoogleProvider` for reliable tool-calling
2. **Stage B (single-turn)** — direct RAG via httpx; fallback on timeout / error / MLX provider

**Auth:** `X-API-Token` header vs `API_TOKEN` env var. If `API_TOKEN` is unset, auth is skipped (dev mode).

### `legalro-dashboard` (`packages/dashboard/`)

Read-only observability FastAPI app. Imports only `legalro-core`.

| Endpoint | Description |
|---|---|
| `GET /health` | MongoDB ping |
| `GET /runs` | Per-batch processing stats (from `COLL_RUNS`) |
| `GET /coverage` | Issues with extraction warnings (QA worklist) |

---

## Data model

```
gazettes collection
  _id             MO_PI_{issue}_{year}  (deterministic)
  issue_number, part, date, year, era, modality
  filename, sha256, page_count, act_count
  schema_version, pipeline_version
  sumar[]         [{act_number, doc_type, title, page_start, page_end}]
  extraction_warnings[]

chunks collection  (retrieval unit)
  _id             {issue_id}::act-{N}::chunk-{N}  (deterministic)
  source_issue_id, act_index_in_issue, law_id
  document_type, act_number, act_year, title, issuing_authority
  modality, publication_date, locality
  chunk_type      preamble | article
  article_number, alineat, litera, full_path
  text, text_normalized, text_embedded
  act_full_text   (capped at 12 000 chars — for parent-doc retrieval)
  embedding[1024] (bge-m3 cosine)
  embedding_version, embedding_model, embedding_dim, tokens
  indexes:
    chunks_vector   ($vectorSearch — 1024-dim cosine + filter fields)
    chunks_search_ro ($search — BM25 Romanian analyzer on text/title)

graph_edges collection
  edge_id, source_law_id, target_law_id, edge_type, gazette_id

runs collection
  started_at, finished_at, ok, failed, coverage_min, ...
```

---

## Hybrid search

```
query string
    │
    ├── embed_texts([query], bge-m3)  →  query_embedding[1024]
    │
    ├── $vectorSearch  numCandidates=200, limit=80  →  vector_results (by cosine)
    │
    ├── $search chunks_search_ro  limit=80          →  text_results   (by BM25)
    │
    ├── Python RRF merge
    │     score = vector_weight / (rrf_k + rank) + text_weight / (rrf_k + rank)
    │     defaults: vector_weight=0.3, text_weight=0.7, rrf_k=60
    │
    ├── metadata boost (applied BEFORE top-N cutoff)
    │     +0.01  exact act_number match
    │     +0.005 exact act_year match
    │     +0.15  exact MO issue + year match  (15× multiplier)
    │
    └── top-N (default 20)  →  parent-doc context assembly  →  LLM
```

---

## Authentication flow (private HF Space)

```
Client                   HF proxy                 FastAPI app
  │                          │                         │
  │  Authorization: Bearer   │                         │
  │    <HF_TOKEN>  ─────────►│ validates HF token      │
  │  X-API-Token:            │                         │
  │    <API_TOKEN>           │  ──forwards request────►│ checks X-API-Token
  │                          │                         │  vs API_TOKEN env var
  │◄─────────────────────────────────────────────────── response
```

---

## Deployment

```
git push origin main
        │
        ▼
GitHub Actions (.github/workflows/deploy-serving.yml)
        │  git push hf main --force  (HF write token)
        ▼
HF Spaces git repo
        │  detects push → rebuilds Docker image (serving package only)
        ▼
Container (FastAPI on port 7860)
        │  reads secrets: MONGODB_URI, GEMINI_API_KEY, API_TOKEN
        ▼
Live at https://rraul99-legalro.hf.space
```

Secrets never touch the git repo or the Docker image — injected at container start by HF Spaces.

---

## Version stamps

Defined in `legalro_core/schema.py` and stamped on every written document:

| Stamp | Bump when | Consequence |
|---|---|---|
| `SCHEMA_VERSION` | Field shapes change | Full re-extraction + re-ingest |
| `PIPELINE_VERSION` | Extraction logic changes | Full re-extraction + re-ingest |
| `EMBEDDING_VERSION` | Chunker or embedding model changes | Re-embed only |

Current: `SCHEMA_VERSION=2.0.0`, `PIPELINE_VERSION=2.0.0`, `EMBEDDING_VERSION=chunker-2.0+bge-m3-1024`
