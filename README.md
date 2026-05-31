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

Ingests Monitorul Oficial gazettes (PDF), stores them in MongoDB Atlas, and answers Romanian legal questions using hybrid search (BM25 + vector) and Gemini.

**Hosted on:** Hugging Face Spaces (Docker) · MongoDB Atlas · Gemini API · Mistral OCR

---

## Table of Contents

1. [Quick start](#quick-start)
2. [How it works](#how-it-works)
3. [Configuration](#configuration)
4. [CLI reference](#cli-reference)
5. [API reference](#api-reference)
6. [Ingest PDFs](#ingest-pdfs)
7. [Deploy to HF Spaces](#deploy-to-hf-spaces)
8. [Local dev mode](#local-dev-mode)
9. [Project structure](#project-structure)

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
# Edit .env — fill in LEGALRO_API_URL, LEGALRO_API_TOKEN, HF_TOKEN

# 3. Use the app
uv run legalro query "Ce drepturi are un salariat?"
uv run legalro chat
uv run legalro ingest path/to/gazette.pdf
```

The CLI reads `.env` automatically — no manual `export` needed.

---

## How it works

```
YOUR MACHINE                    HF SPACES (Docker)              EXTERNAL SERVICES
────────────                    ──────────────────              ─────────────────

uv run legalro ingest file.pdf
  │
  │  POST /ingest  (HF_TOKEN + X-API-Token headers)
  ├────────────────────────────► receive PDF
  │                              ├── born-digital → PyMuPDF / docling (in-container)
  │  {job_id}                    ├── scanned pages ────────────────────────────► Mistral OCR API
  │◄───────────────────────────  ├── chunk + embed (sentence-transformers in RAM)
  │                              └── write to Atlas ──────────────────────────► MongoDB Atlas
  │  GET /jobs/{id}
  ├────────────────────────────► {status: done, chunks: N}
  │◄───────────────────────────

uv run legalro query "..."
  │
  │  POST /query  (HF_TOKEN + X-API-Token headers)
  ├────────────────────────────► embed question (in-container)
  │                              ├── $vectorSearch + $search ──────────────────► MongoDB Atlas
  │  {answer}                    └── generate answer ──────────────────────────► Gemini API
  │◄───────────────────────────
```

**Your machine is a thin client.** All processing — OCR, chunking, embedding, search, generation — runs in the cloud. The only thing that runs locally is the CLI.

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```dotenv
# MongoDB Atlas connection string
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=<app>

# Gemini API key (LLM generation)
GEMINI_API_KEY=your_gemini_key

# Remote client — points CLI at the live HF Space
LEGALRO_API_URL=https://rraul99-legalro.hf.space
LEGALRO_API_TOKEN=your_api_token      # must match API_TOKEN secret in HF Space

# HF read token — required when the Space is private
HF_TOKEN=hf_your_read_token           # huggingface.co/settings/tokens → New token → Read
```

The CLI auto-loads `.env` on startup — no shell exports needed.

**HF Space secrets** (Space → Settings → Variables and secrets):
- `MONGODB_URI`
- `GEMINI_API_KEY`
- `MISTRAL_API_KEY`
- `API_TOKEN` — must match `LEGALRO_API_TOKEN` in your local `.env`

---

## CLI reference

| Command | Description |
|---|---|
| `uv run legalro query "<question>"` | Ask a question (remote by default) |
| `uv run legalro query "<question>" --local` | Run in-process (needs local deps) |
| `uv run legalro query "<question>" --no-agentic` | Single-turn RAG, skip agentic stage |
| `uv run legalro query "<question>" --act-type ORDIN` | Filter by document type |
| `uv run legalro chat` | Interactive multi-turn chat |
| `uv run legalro chat --local` | Chat in-process locally |
| `uv run legalro ingest <path>` | Ingest PDF or directory (remote by default) |
| `uv run legalro ingest <path> --local` | Ingest in-process locally |
| `uv run legalro status` | Show MongoDB connection + corpus counts |
| `uv run legalro start` | Start local MLX LLM server (local dev only) |
| `uv run legalro stop` | Stop local MLX LLM server (local dev only) |

Remote mode is used automatically when `LEGALRO_API_URL` is set in `.env`. Pass `--local` to force in-process execution.

---

## API reference

Base URL: `https://rraul99-legalro.hf.space`

All protected endpoints require two headers:
- `Authorization: Bearer <HF_TOKEN>` — authenticates with the HF Spaces proxy (private Space)
- `X-API-Token: <API_TOKEN>` — authenticates with the FastAPI app

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
# → {"answer": "..."}
```

### `POST /ingest`
Upload a PDF. Returns a `job_id` immediately; processing runs in background.
```bash
curl -X POST https://rraul99-legalro.hf.space/ingest \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "X-API-Token: $LEGALRO_API_TOKEN" \
  -F "file=@MO_PI_311_2026-04-20.pdf"
# → {"job_id": "abc123"}
```

### `GET /jobs/{job_id}`
Poll ingestion status.
```bash
curl https://rraul99-legalro.hf.space/jobs/abc123 \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "X-API-Token: $LEGALRO_API_TOKEN"
# → {"job_id": "abc123", "status": "done", "chunks_created": 48, "detail": "completed"}
```
Status values: `queued` → `running` → `done` | `error`

### `POST /extract`
PDF → GazetteDocument JSON (synchronous). Returns raw extraction without writing to MongoDB.

### `POST /ingest-json`
GazetteDocument JSON → MongoDB (background). Use when you already have an extracted JSON.

---

## Ingest PDFs

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

```bash
# Single file
uv run legalro ingest laws/2026/04/20/MO_PI_311_2026-04-20.pdf

# Whole directory (recursive)
uv run legalro ingest laws/

# Re-ingest after changes (delete cached JSON first)
rm extracted/2026/04/20/MO_PI_311_2026-04-20.json
uv run legalro ingest laws/2026/04/20/MO_PI_311_2026-04-20.pdf
```

The pipeline has two phases:
1. **Extract** — PDF → GazetteDocument JSON (cached in `extracted/`, no DB writes)
2. **Ingest** — JSON → chunk → embed → MongoDB

Already-ingested files are skipped via SHA-256 deduplication.

---

## Deploy to HF Spaces

Every push to `main` auto-deploys via `.github/workflows/deploy.yml`.

### One-time setup

**1. GitHub repository secret:**
Settings → Secrets and variables → Actions → New repository secret
- Name: `HF_TOKEN`
- Value: HuggingFace **write** token from huggingface.co/settings/tokens

**2. HF Space secrets** (Space → Settings → Variables and secrets):
- `MONGODB_URI`
- `GEMINI_API_KEY`
- `MISTRAL_API_KEY`
- `API_TOKEN`

**3. Push to main:**
```bash
git push origin main
# GitHub Actions pushes code to HF → HF rebuilds Docker → app live
```

---

## Local dev mode

For development on macOS Apple Silicon using local MLX models:

```bash
# Install local extras
uv sync --no-dev --extra local

# Start MongoDB + MLX LLM server
uv run legalro start

# Run fully locally
uv run legalro ingest laws/ --local
uv run legalro query "..." --local

# Check status
uv run legalro status

# Stop everything
uv run legalro stop
```

Local mode uses Apple Vision OCR (`ocrmac`) for scanned pages and an MLX LLM (Qwen 9B 4-bit) for generation.

---

## Project structure

```
src/legalro/
  ingestion/
    era.py                Era detection (SCANNED / HYBRID / MODERN / BROKEN_*)
    extract.py            Era-routed text extraction (PyMuPDF / docling / Mistral OCR)
    extract_module.py     Standalone: PDF → GazetteDocument JSON (no DB, no embeddings)
    ingest_module.py      Standalone: JSON → chunk → embed → MongoDB (no PDF)
    pipeline.py           Orchestrator: extract_module → ingest_module
    gazette_extractor.py  PDF parsing → GazetteDocument dataclass
    gazette_schema.py     GazetteDocument, LegalAct, SumarEntry dataclasses
    normalize.py          Diacritics + mojibake fix + BM25 text normalization
    structure.py          Header / footer / page-number stripping
    sumar.py              TOC state-machine parser → page boundaries per act
    segment.py            Act segmentation via sumar pages + header patterns
    metadata.py           Doc type, number, authority, title extraction
    chunk.py              Token-aware chunking (article / paragraph / window)
  providers/
    embeddings.py         sentence-transformers / MLX embed wrapper
    ocr.py                Mistral / LlamaParse / docling / ocrmac dispatcher
    docling_extractor.py  Era-specific docling config + markdown cleanup
    store.py              MongoDB connection pool + insert helpers
  retrieval/
    search.py             $vectorSearch + $search + RRF + metadata boost
    context.py            Parent-doc expansion + context string assembly
  generation/
    agent.py              Stage A (pydantic-ai agentic) + Stage B (single-turn RAG)
  cli/
    app.py                Typer CLI: query / chat / ingest / status / start / stop
    client.py             Thin HTTP client for remote mode
  api/
    app.py                FastAPI: /query /ingest /jobs/{id} /extract /ingest-json /health
  config.py               Settings loader (YAML + env overrides)
  models.py               GazetteResult, Era enum

config/
  cloud.yaml              Production config (Gemini, Atlas, sentence-transformers, Mistral OCR)
  local.yaml              Local dev config (MLX LLM, local MongoDB, ocrmac)

scripts/
  setup_indexes.py        Create Atlas vector + full-text search indexes (run once)
  validate.py             Component health check
  reset.py                Drop DB + clear cache + re-ingest from scratch
  reingest.py             Re-run Phase 2 from cached JSONs (no re-OCR)
  reembed.py              Re-embed all chunks (after model change)
  reclassify.py           Reclassify era for existing gazette records
  extract_gazette.py      CLI wrapper for standalone Phase 1 extraction
  start_mlx.sh            Shell helper to launch MLX LLM server

.github/
  workflows/
    deploy.yml            Auto-deploy to HF Spaces on push to main

extracted/                Phase 1 JSON cache (gitignored)
laws/                     PDF input directory (gitignored)
Dockerfile                HF Spaces image: python:3.12-slim, port 7860
docker-compose.yml        Local MongoDB for dev
```

---

## QA accuracy

Evaluated on 28 Romanian legal questions:

| Run | Model | Infra | n | Correct | Partial | Wrong | Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local baseline | Qwen 9B MLX | Mac | 28 | 22 | 3 | 0 | 3 |
| Cloud v1 | Gemini 3.1 Flash Lite | HF Spaces | 28 | 23 | 4 | 1 | 0 |
| Cloud v2 | Gemini 3.1 Flash Lite | HF Spaces | 52 | 48 | 2 | 2 | 0 |

Cloud eliminates context-window overflow errors. v2 tested on an expanded 52-question set after extraction pipeline improvements (two-column detection, ANCPI doc_type inference, title in FTS index).
