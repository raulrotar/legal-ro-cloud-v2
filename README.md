---
title: LegalRo
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# LegalRo — Romanian Legal RAG System

Ingests Monitorul Oficial gazettes (PDF), stores structured chunks in MongoDB Atlas, and answers Romanian legal questions using hybrid search (BM25 + vector) and Gemini.

**Hosted on:** Hugging Face Spaces (Docker) · MongoDB Atlas · Gemini API

---

## Table of Contents

1. [Architecture overview](#architecture-overview)
2. [Quick start](#quick-start)
3. [Configuration](#configuration)
4. [CLI reference](#cli-reference)
5. [API reference](#api-reference)
6. [Ingest pipeline](#ingest-pipeline)
7. [Deploy to HF Spaces](#deploy-to-hf-spaces)
8. [Local dev mode](#local-dev-mode)
9. [Project structure](#project-structure)
10. [QA accuracy](#qa-accuracy)

---

## Architecture overview

The system is split into four deployable packages in a uv workspace:

```
packages/
  core/        legalro-core        Shared schema, config, embeddings, store, retrieval
  processing/  legalro-processing  VPS/batch: PDF → JSON → chunks → MongoDB
  serving/     legalro-serving     HF Spaces: read-only FastAPI query API
  dashboard/   legalro-dashboard   Read-only observability FastAPI app
```

**Data flow:**

```
YOUR MACHINE (or VPS)                         EXTERNAL SERVICES
─────────────────────                         ─────────────────
legalro-process extract                       Docling / LlamaParse OCR
  PDF → Markdown (md_cache/)
  Markdown → GazetteDocument JSON (extracted/)
  JSON → chunk → bge-m3 embed
  Emit on-disk bundle (out/)
                                              │
legalro-process load                          ▼
  Bundle → MongoDB Atlas ──────────────────► Atlas M0
                                              (gazettes + chunks collections)
                                              │
HF SPACES (Docker, port 7860)                 │
  legalro-serving                             │
    POST /query ──── embed + hybrid search ──►│
                     Gemini generation        │
                     ◄────────────────────────┘
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/raulrotar/legalRo-cloud.git
cd legalRo-cloud
pip install uv
uv sync --no-dev

# 2. Configure credentials
cp .env.example .env
# Edit .env — fill in MONGODB_URI, GEMINI_API_KEY, LEGALRO_API_URL, LEGALRO_API_TOKEN

# 3. Query the live API
uv run legalro query "Ce drepturi are un salariat?"
uv run legalro chat
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```dotenv
# MongoDB Atlas connection string
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=<app>

# Gemini API key (LLM generation + optional extraction LLM)
GEMINI_API_KEY=your_gemini_key

# Remote client — points CLI at the live HF Space
LEGALRO_API_URL=https://rraul99-legalro.hf.space
LEGALRO_API_TOKEN=your_api_token      # must match API_TOKEN secret in HF Space

# HF read token — required when the Space is private
HF_TOKEN=hf_your_read_token
```

**HF Space secrets** (Space → Settings → Variables and secrets):
- `MONGODB_URI`
- `GEMINI_API_KEY`
- `API_TOKEN` — must match `LEGALRO_API_TOKEN` in your local `.env`

**Extraction LLM** (optional, VPS-side):
- `EXTRACTION_LLM_ENABLED=true` — activates LLM-assisted extraction
- `EXTRACTION_LLM_BASE_URL` / `EXTRACTION_LLM_MODEL` / `EXTRACTION_LLM_API_KEY` — route to a separate endpoint (e.g. local vLLM)

---

## CLI reference

### Serving CLI (`legalro`)

| Command | Description |
|---|---|
| `uv run legalro query "<question>"` | Ask a question (remote by default) |
| `uv run legalro query "<question>" --local` | Run in-process (needs local deps) |
| `uv run legalro chat` | Interactive multi-turn chat |
| `uv run legalro ingest <path>` | Ingest PDF or directory (remote by default) |
| `uv run legalro status` | Show MongoDB connection + corpus counts |

### Processing CLI (`legalro-process`)

| Command | Description |
|---|---|
| `uv run legalro-process extract --root laws/ --out out/` | Stage A: PDF → bundle (extract + embed) |
| `uv run legalro-process extract-json --root laws/ --extracted-dir extracted/` | PDF → GazetteDocument JSON only (no embeddings) |
| `uv run legalro-process load --root out/ --mongo "$MONGODB_URI"` | Stage B: bundle → MongoDB |
| `uv run legalro-process setup-indexes --mongo "$MONGODB_URI"` | Create/update Atlas Search + vector indexes |
| `uv run legalro-process reset-db --mongo "$MONGODB_URI" --yes` | Drop all collections (data wipe) |

---

## API reference

Base URL: `https://rraul99-legalro.hf.space`

All protected endpoints require:
- `Authorization: Bearer <HF_TOKEN>` — HF proxy auth (private Space)
- `X-API-Token: <API_TOKEN>` — app-level auth

### `GET /health` — public
```bash
curl https://rraul99-legalro.hf.space/health
# → {"mongodb": true}
```

### `POST /query`
```bash
curl -X POST https://rraul99-legalro.hf.space/query \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "X-API-Token: $LEGALRO_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "Cine a semnat HG 1908/2006?", "act_type": ""}'
# → {"answer": "...", "sources": [...], "latency_ms": 1200, "chunks_used": 10}
```

### `GET /acts`
List legal acts with optional filters:
```bash
curl "https://rraul99-legalro.hf.space/acts?type=LEGE&year_from=2020&limit=20" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "X-API-Token: $LEGALRO_API_TOKEN"
```

### `GET /acts/{act_id}`
Retrieve a single act by its `law_id`:
```bash
curl "https://rraul99-legalro.hf.space/acts/LEGE_123_2024" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "X-API-Token: $LEGALRO_API_TOKEN"
```

---

## Ingest pipeline

PDFs must follow the Monitorul Oficial naming convention:

```
MO_PI_{issue_number}_{YYYY-MM-DD}.pdf
```

Organize under a directory:
```
laws/
  2026/
    04/
      20/
        MO_PI_311_2026-04-20.pdf
```

### Two-stage pipeline

**Stage A — Extract + embed (VPS / local):**

```bash
uv run legalro-process extract --root laws/ --out out/ --extracted-dir extracted/
```

1. PDF → Markdown (Docling; cached in `md_cache/`)
2. Markdown → per-act blocks (segmenter)
3. Blocks → GazetteDocument JSON (LLM structurer; cached in `extracted/`)
4. JSON → chunks → bge-m3 embeddings
5. Emit on-disk bundle under `out/`

**Stage B — Load into MongoDB (network only):**

```bash
uv run legalro-process load --root out/ --mongo "$MONGODB_URI"
```

Idempotent: deterministic `_id`s mean re-loading a bundle is a no-op. A `.load_state.json` lets partial runs resume.

**One-time index setup:**

```bash
uv run legalro-process setup-indexes --mongo "$MONGODB_URI"
```

---

## Deploy to HF Spaces

Every push to `main` that touches `packages/serving/`, `packages/core/`, `Dockerfile`, or `config/cloud.yaml` auto-deploys via `.github/workflows/deploy-serving.yml`.

### One-time setup

**1. GitHub repository secret:**
Settings → Secrets and variables → Actions → New repository secret
- Name: `HF_TOKEN`
- Value: HuggingFace **write** token

**2. HF Space secrets** (Space → Settings → Variables and secrets):
- `MONGODB_URI`
- `GEMINI_API_KEY`
- `API_TOKEN`

**3. Push to main:**
```bash
git push origin main
# GitHub Actions pushes code to HF → HF rebuilds Docker → app live
```

---

## Local dev mode

```bash
# Install all extras for local dev
uv sync

# Stage A: extract + embed locally
uv run legalro-process extract --root laws/ --out out/ --config config/local.yaml

# Stage B: load into local Atlas Docker (see docker-compose.yml)
docker compose up -d
uv run legalro-process setup-indexes --mongo "mongodb://localhost:27017"
uv run legalro-process load --root out/ --mongo "mongodb://localhost:27017"

# Query locally
uv run legalro query "..." --local
```

---

## Project structure

```
packages/
  core/src/legalro_core/
    config.py           Settings loader (YAML + env overrides)
    schema.py           Version stamps, collection names, deterministic _id helpers
    models.py           Era enum (SCANNED / HYBRID / MODERN / BROKEN_*)
    store.py            MongoDB connection pool
    embeddings.py       bge-m3 embed wrapper (sentence-transformers)
    normalize.py        Diacritics + mojibake repair + BM25 text normalization
    bundle.py           On-disk bundle read/write helpers
    retrieval/
      search.py         $vectorSearch + $search + Python RRF + metadata boost
      context.py        Parent-doc expansion + context string assembly

  processing/src/legalro_processing/
    cli.py              Typer CLI: extract / extract-json / load / setup-indexes / reset-db
    extract_module.py   Standalone: PDF → GazetteDocument JSON (sha256 cache)
    bundle_writer.py    Emit gazette + chunks + edges to on-disk bundle
    loader.py           Idempotent bulk-upsert bundle → MongoDB
    extract/
      gazette_extractor.py  PDF → GazetteDocument (regex pipeline)
      pipeline.py           Option C orchestrator: MD → LLM → GazetteDocument
      era.py                Era detection (SCANNED / HYBRID / MODERN / BROKEN_*)
      extract.py            Era-routed text extraction
      md_extractor.py       PDF → full Markdown (Docling / LlamaParse)
      md_cache.py           sha256-keyed Markdown cache
      md_segmenter.py       Full MD → per-act MdActBlock list
      llm_structurer.py     MdActBlock → metadata + corrected text (LLM)
      llm_extract.py        Option B: LLM metadata / segmentation extraction
      gazette_schema.py     GazetteDocument, LegalAct, SumarEntry dataclasses
      metadata.py           Doc type, number, authority, title extraction
      segment.py            Act segmentation via sumar pages + header patterns
      sumar.py              TOC state-machine parser → page boundaries per act
      structure.py          Header / footer / page-number stripping
      ocr.py                Mistral / LlamaParse / docling / ocrmac dispatcher
    prepare/
      build.py              GazetteDocument → bundle docs (gazette + chunks)
      chunk.py              Token-aware chunking (article / paragraph / window)
    audit/
      coverage.py           Coverage ratio: chars mapped vs raw PyMuPDF stream
    graph/
      edges.py              Citation / promulgation edge extraction (GraphRAG)

  serving/src/legalro_serving/
    app.py              FastAPI: /query /acts /acts/{id} /health
    generation.py       Agentic (pydantic-ai GoogleModel) + single-turn RAG fallback
    cli.py              Thin CLI wrapper (legalro query / chat / ingest / status)
    client.py           Thin httpx client for remote mode
    llm.py              LLM provider helpers

  dashboard/src/legalro_dashboard/
    app.py              FastAPI: /runs /coverage /query-log (read-only observability)

config/
  cloud.yaml            Production config (Gemini 2.5 Flash, Atlas, bge-m3, LlamaParse)
  local.yaml            Local dev config (local MongoDB, docling OCR)
  staging.yaml          Staging config (separate Atlas DB)

scripts/
  benchmark_local_llm.py  Benchmark local LLM extraction quality
  compare_extractions.py  Side-by-side regex vs Option C accuracy comparison
  reembed_bge_m3.py       Re-embed all chunks after model change
  reingest_targeted_mos.py Targeted re-ingest of specific gazette issues

tools/
  ops/
    extract_gazette.py  CLI wrapper for standalone Phase 1 extraction
    reembed.py          Re-embed all chunks
    reingest.py         Re-ingest from cached JSONs (no re-OCR)
    reset.py            Drop DB + clear cache + re-ingest
    reclassify.py       Reclassify era for existing gazette records
    setup_indexes.py    Create Atlas vector + full-text indexes
    validate.py         Component health check
  test_questions.py     Local question evaluation harness
  test_questions_cloud.py Cloud question evaluation harness
  evaluate_retrieval.py   Retrieval quality evaluation
  generate_golden_set.py  Generate golden QA set

tests/
  test_chunk.py         Chunker unit tests
  test_era.py           Era detection unit tests
  test_normalize.py     Normalization unit tests
  test_segment.py       Segmenter unit tests

.github/workflows/
  deploy-serving.yml    Auto-deploy serving package to HF Spaces on push to main
  deploy-dashboard.yml  Auto-deploy dashboard package
  ci-processing.yml     CI for processing package

Dockerfile              HF Spaces image: serving package only, port 7860
docker-compose.yml      Local MongoDB Atlas Local for dev
```

---

## QA accuracy

Evaluated on 52 Romanian legal questions across all eras:

| Run | Extraction pipeline | LLM | n | Correct | Partial | Wrong | Error |
|---|---|---|---|---|---|---|---|
| Local baseline | Regex | Qwen 9B MLX | 28 | 22 | 3 | 0 | 3 |
| Cloud v1 | Regex | Gemini Flash Lite | 28 | 23 | 4 | 1 | 0 |
| Cloud v2 | Regex | Gemini Flash Lite | 52 | 48 | 2 | 2 | 0 |
| Option C | Docling→MD→LLM | Llama 3.1 8B | 52 | 50 | 2 | 0 | 0 |

Option C (Docling→Markdown→LLM structuring) achieves **96.8% accuracy** on the full test set with Llama 3.1 8B local extraction and Gemini for generation.
