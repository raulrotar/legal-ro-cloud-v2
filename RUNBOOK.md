# LegalRo — Full Pipeline Runbook

Step-by-step reference for a clean extraction → ingest cycle.
Run every command from the **repository root**.

---

## Prerequisites

### Environment variables

```bash
export MONGODB_URI="mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?appName=<app>"
export GEMINI_API_KEY="<your-gemini-key>"
export LLAMA_CLOUD_API_KEY="<your-llamaparse-key>"   # required for cloud OCR
```

Or place them in `.env` at the repo root (already gitignored):

```
MONGODB_URI=mongodb+srv://...
GEMINI_API_KEY=...
LLAMA_CLOUD_API_KEY=...
```

> `.env` is loaded automatically by the pipeline at startup.

### Source PDFs

Place gazette PDFs under `laws/` following the convention:

```
laws/{YYYY}/{MM}/{DD}/MO_PI_{issue}_{YYYY}-{MM}-{DD}.pdf
```

Example: `laws/2026/04/14/MO_PI_295_2026-04-14.pdf`

---

## When to run a full clean cycle

Run the full pipeline (all steps below) when:

- Code changes to `md_segmenter.py`, `metadata.py`, `pipeline.py`, or any extraction module
- New PDFs added to `laws/`
- MongoDB schema or chunk format changed
- Search index definition changed
- After `git pull` that includes extraction or chunking fixes

For serving-only changes (prompts, temperature, search weights) a redeploy is enough — no re-extraction needed.

---

## Step 0 — Wipe local artifacts

Remove all cached intermediate files so extraction runs fresh.

```bash
rm -rf db/md_cache/
rm -rf db/bundle_bge-m3/by_doc db/bundle_bge-m3/manifest.jsonl
rm -f .load_state.json
rm -rf db/extracted/
```

`db/md_cache/` — Docling/LlamaParse Markdown output (Option C intermediate).
`db/bundle_bge-m3/` — embedded chunk bundles (input to `load`).
`db/extracted/` — raw GazetteDocument JSONs (structural cache, re-created by `extract`).
`.load_state.json` — resume-state file; delete so `load` doesn't skip anything.

---

## Step 1 — Reset MongoDB

Drop all LegalRo collections (gazettes, chunks, graph_edges, runs).

```bash
echo "y" | uv run legalro-process reset-db --mongo "$MONGODB_URI"
```

Expected output:
```
dropped: gazettes
dropped: chunks
dropped: graph_edges
dropped: runs
[reset-db] done
```

> Skip this step if you only want to add new issues without removing existing ones.
> The `load` command is idempotent (upserts by deterministic `_id`), so re-loading
> existing issues without resetting is safe.

---

## Step 2 — Extract: PDF → chunks → embeddings (local)

Runs all PDFs through OCR → segmentation → extraction → embedding and writes
on-disk bundles under `db/bundle_bge-m3/`.  Everything runs locally — no cloud calls
except optional OCR via LlamaParse.

```bash
uv run legalro-process extract \
  --root laws/ \
  --out db/bundle_bge-m3/ \
  --extracted-dir db/extracted/
```

Key flags:

| Flag | Default | When to change |
|------|---------|----------------|
| `--root` | required | Point to a subdirectory (e.g. `laws/1989/`) for partial runs |
| `--out` | `db/bundle_bge-m3` | Keep default unless testing |
| `--extracted-dir` | `db/extracted` | Keep default unless testing |
| `--config` | `config/local.yaml` | Use `config/cloud.yaml` for cloud LLM extraction |
| `--no-embed` | off | Add to skip embedding (structural-only test run) |
| `--limit N` | 0 (all) | Process only first N PDFs (useful for smoke tests) |

Expected final line:
```
[extract] done: 21 ok, 0 failed, coverage_min=0.791
[extract] manifest: db/bundle_bge-m3/manifest.jsonl
```

Validation warnings (`DOC_TYPE_UNKNOWN`, `ACT_NUMBER_ZERO`) for individual acts
are normal — they are annotated in the JSON and don't block ingestion.

---

## Step 3 — Load: bundles → MongoDB (push to cloud)

Upserts all locally built bundles from `db/bundle_bge-m3/` into MongoDB Atlas.

```bash
uv run legalro-process load \
  --root db/bundle_bge-m3/ \
  --mongo "$MONGODB_URI" \
  --no-resume
```

Use `--no-resume` on a fresh ingest (after `reset-db`) so the load-state file
is ignored. For incremental top-ups (adding new issues to an existing DB) use
the default `--resume` to skip already-loaded bundles.

Expected output:
```
[load] loaded=21 skipped=0
```

---

## Step 4 — Create / update search indexes

Creates (or updates) the Atlas Search and vector indexes required for hybrid search.

```bash
uv run legalro-process setup-indexes \
  --mongo "$MONGODB_URI" \
  --no-wait
```

> Use `--wait` on Atlas M10+ clusters where `$listSearchIndexes` is available.
> On M0/M2 (free/shared tier) use `--no-wait` — indexes build in the background
> and are typically READY within 1–3 minutes.

Indexes created:
- `chunks_vector` — Atlas Vector Search (bge-m3, 1024-dim, cosine)
- `chunks_search_ro` — Atlas Search full-text (Romanian analyser, BM25)

---

## Step 5 — Verify

Quick sanity check — confirm documents are in MongoDB and search works:

```bash
source .env
python - <<'EOF'
from legalro_core.config import load_settings
from legalro_core.store import get_db
from legalro_core.retrieval.search import hybrid_search

s = load_settings()  # defaults to config/local.yaml; set LEGALRO_ENV=cloud for cloud config
db = get_db(s)
print("chunks:", db.chunks.count_documents({}))
print("gazettes:", db.gazettes.count_documents({}))

results = hybrid_search("Decretul-lege nr. 2/1989", s)
print("search results:", len(results))
for r in results[:3]:
    print(" -", r.get("source_issue_id"), r.get("title", "")[:60])
EOF
```

Expected: `chunks` ≥ 800, `gazettes` ≥ 21, `search results` = 20.

---

## Step 6 — Run QA test suite

```bash
source .env
uv run python tools/test_questions_cloud.py
```

The serving app must be running (local or HF Spaces). Set `BASE_URL` if needed:

```bash
BASE_URL=http://localhost:7860 uv run python tools/test_questions_cloud.py
```

Target: **≥ 48/52 CORECT**.

---

## Partial re-ingest (single year or issue)

Re-extract one year without touching other issues in MongoDB:

```bash
# Extract only 1989 issues (local)
uv run legalro-process extract \
  --root laws/1989/ \
  --out db/bundle_bge-m3/

# Push to cloud (--resume skips already-loaded)
uv run legalro-process load \
  --root db/bundle_bge-m3/ \
  --mongo "$MONGODB_URI"
```

To force reload of specific issues after a code fix, remove their entries from
`.load_state.json` before running `load`:

```bash
# Example: force reload MO_PI_2_1989 and MO_PI_4_1989
python - <<'EOF'
import json, pathlib
f = pathlib.Path(".load_state.json")
state = json.loads(f.read_text()) if f.exists() else {}
for key in list(state):
    if "1989" in key:
        del state[key]
f.write_text(json.dumps(state, indent=2))
print("Removed 1989 entries from load state")
EOF

uv run legalro-process load --root db/bundle_bge-m3/ --mongo "$MONGODB_URI"
```

---

## Deploying serving changes (no re-ingest)

For changes to `generation.py`, `search.py`, `app.py`, or `config/cloud.yaml`
(temperature, prompts, weights):

```bash
git push origin main
# Then redeploy HF Space from the HF dashboard or via:
# huggingface-cli repo sync --repo-type space <space-name>
```

No extraction or MongoDB changes needed.

---

## Corpus gaps (known missing data)

See `notes/corpus_gaps.md` for gazette issues that are not yet in `laws/`
and which questions they block (currently Q16 and Q17 — HG 1908/2006).
