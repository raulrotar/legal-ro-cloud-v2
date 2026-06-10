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

### Extraction architecture

```
PDF
 │
 ├─ era=SCANNED (1989) ────→ GLM-OCR (glm-ocr:latest, 2.2 GB)
 │                            page-by-page vision, ~13 s/page
 │                            correct diacritics (ț/ș/ă/î)
 │
 ├─ era=BROKEN_2007 ─────→ Docling
 │                            visual render recovers mojibake glyphs
 │
 └─ era=MODERN/HYBRID ───→ Docling
                             text layer + TableFormer (no OCR)
                                   │
                            md_cache (sha256-keyed .md files)
                                   │
                      fitz_enrich — PyMuPDF injects missing
                       closing blocks from PDF text layer
                                   │
                      md_segmenter → per-act MdActBlock list
                       (era-aware thresholds)
                                   │
               ┌── per-act loop ──────────────────────────┐
               │  Stage 1:   md_rule_extractor (regex)    │
               │             act_number, act_year,         │
               │             doc_type, issuing_authority   │
               │  Stage 1.5: secondary_analyzer            │
               │             (fitz closing-sig recovery)   │
               │  Stage 1.6: sumar positional fallback     │
               │  Stage 2:   build from rule_draft — NO LLM│
               └──────────────────────────────────────────┘
                                   │
                          phantom dedup
                                   │
               ┌── inline validation + vision repair ─────┐
               │  validate_act_inline() per act            │
               │                                           │
               │  CLEAN ✅ → pass through                  │
               │  FLAGGED ⚠️ (ACT_NUMBER_ZERO,             │
               │              DOC_TYPE_UNKNOWN, …)         │
               │        ↓                                  │
               │  llama3.2-vision:11b (7.8 GB)            │
               │  receives: PDF page images + broken       │
               │   fields + error codes                    │
               │  patches ONLY flagged fields              │
               │  records repair_flag:true                 │
               └───────────────────────────────────────────┘
                                   │
                          annex propagation
                                   │
                          GazetteDocument JSON
```

**Memory profile — single phase, no prewarm needed:**

| Stage | Model | RAM |
|-------|-------|-----|
| MD extraction (scanned) | GLM-OCR | 2.2 GB |
| MD extraction (other eras) | Docling | ~3 GB |
| Per-act build | None — pure regex | ~0 |
| Repair pass (flagged acts only) | llama3.2-vision:11b | 7.8 GB |

Docling and the vision repair model never coexist — Docling is GC'd before the repair pass fires.

**Required models:**
```bash
ollama pull glm-ocr:latest          # 2.2 GB — scanned era OCR
ollama pull llama3.2-vision:11b     # 7.8 GB — vision repair pass
```

### Two-stage pipeline

**Stage A — Extract + embed (VPS / local):**

```bash
uv run legalro-process extract --root laws/ --out out/ --extracted-dir extracted/
```

1. PDF → Markdown via GLM-OCR (scanned era) or Docling (other eras); cached in `md_cache/`
2. Markdown → per-act blocks (segmenter)
3. Blocks → GazetteDocument JSON (regex extraction + inline vision repair for flagged acts)
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
    llm_client.py       Shared OpenAI-compat LLM client (Gemini / Ollama)
    normalize.py        Diacritics + mojibake repair + BM25 text normalization
    md_normalize.py     Markdown-specific normalization helpers
    act_number.py       Act number parsing and canonicalisation
    bundle.py           On-disk bundle read/write helpers
    retrieval/
      search.py         $vectorSearch + $search + Python RRF + metadata boost
      context.py        Parent-doc expansion + context string assembly

  processing/src/legalro_processing/
    cli.py              Typer CLI: extract / extract-json / load / setup-indexes / reset-db
    extract_module.py   Standalone: PDF → GazetteDocument JSON (sha256 cache)
    ingest_module.py    End-to-end ingest helper (extract + embed + load)
    bundle_writer.py    Emit gazette + chunks + edges to on-disk bundle
    loader.py           Idempotent bulk-upsert bundle → MongoDB
    pipeline.py         Top-level orchestrator coordinating extract→embed→load
    extraction_validator.py  Field-level validation with error codes
    gazette_validator.py     Gazette-level cross-act consistency checks
    fallback_merge.py        Orphan-act merge heuristics
    run_ledger.py            Run tracking / audit log
    extract/
      gazette_extractor.py   PDF → GazetteDocument (regex pipeline)
      pipeline.py            Option C orchestrator: GLM-OCR/Docling→MD→regex→repair
      era.py                 Era detection (SCANNED / HYBRID / MODERN / BROKEN_*)
      extract.py             Era-routed text extraction
      md_extractor.py        PDF → full Markdown (GLM-OCR for scanned, Docling otherwise)
      docling_extractor.py   Docling-specific extraction wrapper
      md_cache.py            sha256-keyed Markdown cache
      md_segmenter.py        Full MD → per-act MdActBlock list
      md_rule_extractor.py   Deterministic regex extractor (primary path)
      md_table_extractor.py  Table-dense page triage and extraction
      secondary_analyzer.py  PyMuPDF closing-sig recovery (Stage 1.5)
      blocks.py              MdActBlock data structure
      roles.py               Line-role classification helpers
      gazette_schema.py      GazetteDocument, LegalAct, SumarEntry dataclasses
      metadata.py            Doc type, number, authority, title extraction
      segment.py             Act segmentation via sumar pages + header patterns
      sumar.py               TOC state-machine parser → page boundaries per act
      structure.py           Header / footer / page-number stripping
      ocr.py                 LlamaParse / Docling / GLM-OCR dispatcher
      llm_repair.py          Vision-LLM repair pass (on-demand, flagged acts only)
      llm_extract.py         Option B: LLM metadata / segmentation extraction
      llm_structurer.py      Legacy LLM structurer (unused; kept for reference)
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
  cloud.yaml            Production config (gemini-3.1-flash-lite, Atlas, bge-m3, LlamaParse)
  local.yaml            Local dev config (local MongoDB, Docling + GLM-OCR)

scripts/
  compare_extractions.py   Side-by-side diff of extraction outputs
  reingest_targeted_mos.py Targeted re-ingest of specific gazette issues

tools/
  ops/
    extract_gazette.py   CLI wrapper for standalone extraction
    glm_ocr_compare.py   GLM-OCR vs Docling output comparison
    prewarm_md_cache.py  Pre-populate md_cache without full extraction
    verify_md_coverage.py Check md_cache coverage vs laws/ directory
    reembed.py           Re-embed all chunks after model change
    reingest.py          Re-ingest from cached JSONs (no re-OCR)
    reset.py             Drop DB + clear cache + re-ingest
    reclassify.py        Reclassify era for existing gazette records
    setup_indexes.py     Create Atlas vector + full-text indexes
    validate.py          Component health check
  test_questions.py      Local question evaluation harness
  test_questions_cloud.py Cloud question evaluation harness
  evaluate_retrieval.py  Retrieval quality evaluation
  generate_golden_set.py Generate golden QA set

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
| Current (regex + vision repair) | GLM-OCR/Docling→MD→regex+repair | Gemini 3.1 Flash Lite | 52 | 48+ | — | — | — |

The current pipeline replaces the per-act text-LLM with deterministic regex extraction (primary path, zero LLM calls on clean acts) and a vision repair pass (`llama3.2-vision:11b`) that fires only for acts that fail inline validation. Memory peak during extraction is ~7.8 GB (repair pass, on-demand only); query-side uses Gemini 3.1 Flash Lite.
