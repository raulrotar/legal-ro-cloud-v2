# QA Spec — HTML-table extraction & retrieval correctness (2026-06-15)

Purpose: a stable, reusable QA set to measure whether the HTML-`<table>`
(colspan/rowspan) work **improves** the app, split into two independently
checkable layers:

1. **Extraction accuracy/correctness** — structural assertions on the extracted
   `Table` objects / chunks. Runnable offline (no services), deterministic.
2. **Retrieval correctness** — end-to-end RAG questions in the existing harness
   tuple format (`tools/test_questions_cloud.py`). Needs the serving stack
   (Atlas + embeddings + Gemini) — see "Runtime" below.

Tie-in: this measures the **latent capability** the impact team flagged — the
current 51-question suite has NO Nomenclator cell-lookup question, so the
merged-header capability is currently unmeasured. These questions close that gap.

## Runtime / how to run the baseline
- **Extraction checks**: `uv run pytest tests/test_table_html_regression.py`
  (to be created with the feature; spec in §3). No services required.
- **Retrieval checks**: start the serving app (needs Ollama OR Gemini creds +
  Mongo with the bundle ingested), then
  `LEGALRO_API_URL=http://localhost:7861 uv run python tools/test_questions_tables.py`.
- **Baseline status (2026-06-15)**: live-RAG baseline NOT captured — Ollama down,
  no Gemini/Mongo creds in env. Extraction-level baseline IS captured below from
  `db/extracted/2026/04/14/MO_PI_294Bis_2026-04-14.json` (34 tables, 345,076
  table-markdown chars). Re-run the retrieval set once services are up to get the
  "before" numbers, then again after each implementation stage.

## §1 — Captured extraction baseline (the "before")
Source: `db/extracted/.../MO_PI_294Bis_2026-04-14.json`, `tables[0]`
(`page` index 2 = PDF page 3; the `page` field is 0-based).
The Nomenclator is a **transposed multi-level-header** table: attribute rows
(Domeniu detaliat ISCED, Cod ISCED, Număr de credite, Specializarea, Cod S, …),
specialization columns. Current defects to be fixed:
- **Merged-header flattening**: `Domeniu detaliat ... | Matematică | Matematică | Matematică | Dezvoltarea... | Dezvoltarea...` — one logical header band repeated as N leaf columns; the colspan that says "Matematică groups 3 specializations" is lost.
- **Text-bleed / word-disorder in cells**: e.g. `"Ingineria substanțelor și protecția anorganice mediului"`, `"Tehnologia informației și (TIC) comunicațiilor"` — adjacent cell fragments concatenated out of order.
- **Hierarchy lost**: cannot recover that specialization "Inteligență artificială" sits under domain "Informatică" → answer-LLM and BM25 see column-soup.

Known-good cell values (verifiable from `tables[0]`, use as assertion oracle):
| Specializarea (S) | Cod ISCED F-2013 | Credite (ECTS) | Cod S |
|---|---|---|---|
| Matematică | 0541 | 180 | 10 |
| Matematici aplicate | 0541 | 180 | 20 |
| Informatică | 0613 | 180 | 10 |
| Inteligență artificială | 0619 | 180 | 30 |
| Securitate informatică și știința datelor | 0613 | 180 | 40 |
| Fizică | 0533 | 180 | 10 |
(Domain "Inginerie chimică și procese" → ISCED 0711, 240 credite — `tables[1]`.)

## §2 — Retrieval-correctness questions (RAG, harness format)
Defined in `tools/test_questions_tables.py`. Expected answers are grounded in §1.
Q-ids continue the existing suite (last is Q55).

- **QT1 (Nomenclator cell lookup — credite)**: "Câte credite ECTS are specializarea «Inteligență artificială» din Nomenclatorul aprobat prin HG 191/2026 (MO 294 bis/14.IV.2026)?" → **180**
- **QT2 (Nomenclator cell lookup — cod ISCED de domeniu)**: "Ce cod ISCED F-2013 corespunde domeniului «Matematică» în Nomenclator?" → **0541**
- **QT3 (cross-column / hierarchy)**: "În ce domeniu de studii se încadrează specializarea «Securitate informatică și știința datelor» și câte credite are?" → **Informatică; 180 credite**
- **QT4 (different-credit domain)**: "Câte credite ECTS sunt prevăzute pentru specializările din domeniul «Inginerie chimică și procese»?" → **240**
- **QT5 (party-financing table total — exercises a clean small table)**: "Care este cuantumul total al cotizațiilor primite în luna ianuarie, conform tabelului din MO nr. 311/20.IV.2026?" → numeric total per the cotizatii table (oracle: sum of the ianuarie column).
- **QT6 (beneficiary/annex list recall — MO 822)**: "Apare beneficiarul/ rândul «…» în lista din MO nr. 822/3.XII.2007?" → row presence check (recall on a clean ruled annex; positive regression guard).

Note: QT5/QT6 should already pass today (clean tables) — they are **positive
regression guards** so the change does not break the cases that work. QT1–QT4
exercise the new capability and are expected to be PARTIAL/WRONG on the baseline.

## §3 — Extraction-correctness assertions (offline pytest)
To be implemented as `tests/test_table_html_regression.py` alongside the feature.
Each asserts on the extracted `Table` for 294Bis (`tables[0]`):

1. **No merged-header text-bleed**: no cell text contains a known bleed bigram
   (e.g. "anorganice mediului", "și (TIC) comunicațiilor"). Tokens within a cell
   appear in source order.
2. **Header spans preserved (HTML path)**: `Table.html` contains `colspan` > 1 on
   the domain header row (e.g. a `<th colspan="3">Matematică</th>`), proving the
   merged band is encoded rather than repeated.
3. **Hierarchy recoverable**: from `Table.html`, the specialization "Inteligență
   artificială" resolves to domain "Informatică" and credite "180".
4. **Oracle cells exact**: the six (specializare → ISCED, credite, Cod S) rows in
   §1 are recoverable exactly from the table.
5. **Flattened view is clean & tag-free**: `text_flat` / `text_embedded` for the
   table contains no `<`/`>`/`colspan`, and reads in source order (the bge-m3 /
   BM25 payload). HTML must NOT appear in `text_embedded`, `text`,
   `text_normalized`, or the coverage `mapped_texts`.
6. **Coverage gate intact**: re-extracting 294Bis keeps `coverage` within
   [baseline, 1.0] and `coverage_min ≥ 0.79` across the dev corpus (HTML tags
   must not inflate the metric — guardrail #2).
7. **Column-count sanity**: per-page header band collapses repeated leaf columns;
   distinct domain count per page ≈ 8 (not the ~24 flattened leaves).

## Success criteria (per implementation stage)
- Extraction: assertions §3.1, §3.4, §3.5, §3.6 pass (span assertions §3.2/§3.3
  pass once the true-span stage lands; flat-HTML stage may xfail them).
- Retrieval: QT5/QT6 stay correct (no regression); QT1–QT4 improve from
  baseline; existing 51-question suite stays ≥ 51/52; `coverage_min ≥ 0.79`.
