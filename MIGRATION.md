# LegalRo v2 — Monorepo migration

This repo is the restructured (monorepo) successor to `legal-Ro-poc-cloud-v1`.
v1 is untouched and remains the running reference system. v2 splits the codebase
into independently-deployable packages that share one contract package.

## Layout

```
legalro/  (uv workspace, virtual root)
├── packages/
│   ├── core/        legalro-core        shared contract — imported by all others
│   ├── processing/  legalro-processing  VPS/batch: PDF → JSON → chunks+embeddings → bundle → Mongo
│   ├── serving/     legalro-serving     read-only query API (HF/AWS)
│   └── dashboard/   legalro-dashboard   read-only observability
├── deploy/          one Dockerfile per deployable
├── .github/workflows/  path-filtered: a change to one package builds only that one
├── config/          shared yaml (local.yaml, cloud.yaml; add vps.yaml)
├── tools/           eval harness (test_questions*) + ops/ scripts
├── tests/
└── laws/, extracted/   local data (gitignored)
```

## The contract (`legalro-core`) — never duplicate these

- `schema.py` — version stamps, collection names, deterministic `_id`s, Era→modality.
- `embeddings.py` — THE embedder (same model/prefix/normalization on both sides → no drift).
- `store.py`, `models.py`, `normalize.py`, `retrieval/` — DB + read logic.
- `bundle.py` — on-disk Stage A/B bundle format.

## Dependency isolation

| Package | depends on | heavy deps |
|---|---|---|
| core | — | (core[ml] adds sentence-transformers) |
| processing | core[ml] | pymupdf, docling, tiktoken, llama-parse, mistralai |
| serving | core[ml] | fastapi, pydantic-ai |
| dashboard | core (base) | fastapi only |

`uv sync --package legalro-serving` builds serving + core only — never pulls docling/OCR.

## What changed from v1

- `src/legalro/**` was split across the four packages (see git moves).
- All `from legalro.x` imports rewritten to `legalro_core.* / legalro_processing.* / legalro_serving.*`.
- `serving/app.py` is now **read-only** — the `/ingest`, `/extract`, `/ingest-json`
  endpoints were removed so serving no longer imports processing. Ingestion runs
  out-of-band via `legalro-process`.
- New Stage A/B split: `core/bundle.py` + `processing/bundle_writer.py` (write) and
  `processing/loader.py` (idempotent load). This replaces the old inline
  "ingest_module writes straight to Mongo".

## Status: Stage A runs end-to-end (PDF → bundle, with real bge-m3 embeddings)

Done:
- [x] Package structure, manifests, workspace, Dockerfiles, path-filtered CI.
- [x] Imports rewritten; serving decoupled from processing.
- [x] Contract modules (`schema`, `bundle`) + Stage B loader implemented.
- [x] `uv sync` green (266 packages); all four packages import at runtime.
- [x] `legalro-process extract` wired: PDF → JSON → chunk → embed → bundle.
      Verified on 2017 PDFs: deterministic ids, facets, version stamps, real
      1024-dim bge-m3 vectors, per-doc coverage_ratio, manifest + checksums.
      `prepare/build.py` does gazette→docs (deterministic ids + facets +
      modality, NO act_full_text bloat, real coverage vs raw PyMuPDF stream).

TODO (next passes):
1. Run `pytest tests` and fix any test fallout from the moves.
2. Retire the legacy `ingest_module.py` (its DB-write role is now bundle + loader;
   `prepare/build.py` superseded its chunk-build half).
3. Point **MongoDB Atlas Local** (Docker) at a pilot bundle:
   `legalro-process load --root out/ --mongo "mongodb://localhost:27017/?directConnection=true"`,
   then `tools/test_questions.py` to validate accuracy before paying for Atlas.
4. Accuracy work: per-doc coverage is ~0.95–0.98 today — the audit shows act
   segmentation drops a few % of raw text. Tighten reading-order/segmentation
   (spec §3.6/§3.8) to push toward 1.0.

## Verified commands

```
uv sync                                                     # green
uv run legalro-process extract --root laws/2017 --out out/ --no-embed   # fast structural run
uv run legalro-process extract --root laws/2017/01/30 --out out/ --limit 1   # with bge-m3
```

## Pilot flow (target)

```
VPS:    legalro-process extract --root laws/ --out out/      # → bundle
        rsync out/ → local
local:  docker run mongodb/mongodb-atlas-local
        legalro-process load --root out/ --mongo "mongodb://localhost:27017/?directConnection=true"
        uv run python tools/test_questions.py                # score accuracy
```
