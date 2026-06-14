# Feasibility: adapting "Accurately Extract Everything from Documents" prompts (2026-06-15)

Source article: *How to Accurately Extract Everything from Documents Using AI* —
Umair Ali Khan, AI Advances, Apr 2026
(https://ai.gopubby.com/how-to-accurately-extract-everything-from-documents-using-ai-cf12d0125238).
It repackages LlamaIndex's **ParseBench** parsing prompts into a `MultimodalParser`
Python class.

**Decision: analysis only — no prompt change applied, no extraction rerun this
session.** This document records the assessment and the ready-to-run change +
commands for later.

## What the article actually proposes

1. Send the **whole PDF** (base64) to a **frontier cloud VLM** — Gemini 3
   Flash / GPT / Claude — via each provider's native API.
2. A shared system+user prompt (`prompts.py`) that asks the model to:
   - output **HTML tables** (`<table><tr><th><td>`) with **`colspan`/`rowspan`**
     instead of Markdown tables, to preserve merged cells & multi-level headers;
   - convert **charts/graphs → tables** with flat combined column headers;
   - wrap **every layout element** in `<div data-bbox="[…]" data-label="…">`
     (normalized 0–1000 coords; Gemini uses `[y_min,x_min,y_max,x_max]`,
     OpenAI/Claude use `[x1,y1,x2,y2]`);
   - a `merge_table` toggle that appends "combine tables split across pages".
3. `create_html` renders the cleaned markdown to a styled HTML document.

The article's benchmark target is **table-dense business documents** (purchase
orders, BOMs) where Markdown tables genuinely lose structure.

## Why most of it does not transfer to this pipeline

This is a **Romanian legal-gazette** pipeline (Monitorul Oficial), **local-only**
by design (Ollama; 50k-PDF target; cloud is a P3 "strategic" item only). Content
is ~95% **prose** legal acts; tables appear only in **annexes** (financial
tables, nomenclatures, party-financing lists).

| Article idea | Verdict | Reason (file:line) |
|---|---|---|
| HTML `<table>` + `colspan`/`rowspan` | **Narrow** | Only the annex/financial path benefits. That path is **deterministic PyMuPDF `find_tables()` → `_to_markdown()`** (`annex_tables.py:26,86`) — *not an LLM prompt*. So there is no "prompt" to swap; the analogue is a `_to_markdown`→`_to_html` renderer change. |
| `<div data-bbox>` / `data-label` | **Harmful / N/A** | Nothing downstream reconstructs spatial layout. Segmentation is **heading-driven** on `##` act titles (`md_extractor.py:163` `_OCR_PROMPT_STRUCTURED`; `md_segmenter.py`). bbox divs would break the segmenter and only add token cost. |
| Charts/graphs → tables | **N/A** | Gazettes contain no charts/graphs. |
| `merge_table` across pages | **Already shipped** | `annex_tables.py` stitches multi-page tables by repeated-header / column-count match (`extract_annex_tables`, docstring lines 9–13); enabled via `extraction.annex_tables_fitz` (`pipeline.py:128-130`). |
| Cloud multi-provider VLM (OpenAI/Claude/Google) | **Conflicts** | Local-only constraint. Whole-doc cloud parsing is explicitly out of scope (action plan P3, `known_limitations_action_plan_2026-06-12.md`). |

### Key structural finding
The article's whole premise — "frontier VLMs mangle Markdown tables, so prompt
for HTML" — **is already sidestepped here**: ruled annexes go through
deterministic `find_tables(strategy="lines_strict")`, which the action plan
benchmarks as superior to Camelot (~73%) and TableFormer-in-Markdown
(5× inflation, per-page fragmentation). The remaining weakness is not *table
detection* but the **Markdown rendering** of detected cells (merged header cells
flatten). That — and only that — is where the article's HTML-table idea adds
value here.

## The one on-strategy adaptation (proposed, NOT applied)

Swap the annex renderer from pipe-Markdown to HTML with `colspan`/`rowspan`,
keeping the same `Table` dataclass and stitch logic. PyMuPDF cell geometry can
detect horizontal spans (cells wider than one column → `colspan`).

Proposed addition to `annex_tables.py` (alongside `_to_markdown`, switched in
`_flush`):

```python
def _to_html(header: list, rows: list[list]) -> str:
    import re
    def cell(tag: str, c) -> str:
        return f"<{tag}>{re.sub(r'\s+', ' ', str(c or '')).strip()}</{tag}>"
    head = "<tr>" + "".join(cell("th", c) for c in header) + "</tr>"
    body = "".join("<tr>" + "".join(cell("td", c) for c in r) + "</tr>" for r in rows)
    return f"<table>{head}{body}</table>"
```

> Note: emitting true `colspan`/`rowspan` requires reading PyMuPDF cell bbox
> widths (not yet captured by the current row-list flow). A first cut emits flat
> HTML; spans are a follow-up once cell geometry is threaded through.

**Risk to check before merging:** downstream chunking/coverage and the QA suite
assume pipe-table Markdown (`build_issue_docs`, `chunk_type='financial_table'`,
`gazette_schema.py:58`). HTML tables must still be counted by the coverage fix
(P0 item 1) and remain searchable in OpenSearch — verify before adopting.

## Exact rerun + compare commands (run when Ollama is up)

Prereqs: `ollama serve` running; models pulled (glm-ocr OCR model, gemma4
summarizer, vision-repair model per `config/local.yaml`).

```bash
# 0. Baseline FIRST (current prompts), into a separate dir to diff against:
uv run legalro-processing extract \
  --root laws --out db/bundle_baseline --extracted-dir db/extracted_baseline \
  --embed false                         # fast structural run, no embeddings

uv run python tools/ops/audit_extraction.py \
  --json-dir db/extracted_baseline --md-cache db/md_cache \
  --out reports/audit_baseline.md

# 1. Apply the _to_html change, then re-extract.
#    Annex tables are deterministic (no md_cache invalidation), so this is fast.
uv run legalro-processing extract \
  --root laws --out db/bundle_html --extracted-dir db/extracted_html \
  --embed false

uv run python tools/ops/audit_extraction.py \
  --json-dir db/extracted_html --md-cache db/md_cache \
  --out reports/audit_html.md

# 2. Compare.
diff -u reports/audit_baseline.md reports/audit_html.md
#    then the 52-question QA suite (success bar: QA >= 51/52, coverage_min >= 0.79,
#    0 unmatched sumar entries in modern issues — same gates as the action plan).
```

Note: `db/extracted` JSONs are sha-cached — extract into **fresh** dirs (as
above) or stale JSONs get skipped (known gotcha).

## Recommendation
Adopt **only** the HTML-table renderer for the deterministic annex path, gated
behind a config flag, and validate against the QA suite before default-on. Skip
bbox-div wrapping, chart-to-table, and the cloud multi-provider design — they are
N/A or conflict with the local-only architecture. The article's most useful
contribution here is conceptual confirmation that **HTML tables beat Markdown for
merged cells**, which the pipeline can apply at the renderer (deterministic)
layer rather than via prompts.
