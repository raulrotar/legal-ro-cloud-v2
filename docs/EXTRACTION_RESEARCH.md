# Extraction Quality — Audit & Research Report (2026-06-10)

> **Status update (same day):** the local-first subset of §3 is IMPLEMENTED —
> see §0 below for results on the 21-PDF batch.  Cloud OCR options were
> dropped per the low-cost/local constraint (50k-PDF corpus → local Mac now,
> rented VPS later).

## 0. Implemented fixes & batch results

| Fix | Module | Result on batch |
|---|---|---|
| Tiled VLM-OCR under Ollama's 2048px clip limit + retry ladder (full-page → column-split → Tesseract text) | `extract/page_tiles.py`, `extract/md_extractor.py` | 1989 content +36…+110% per issue (MO_PI_4: 12.2K→25.4K chars); 17/20 pages pass the gate, 3 flagged with reasons |
| Tesseract `-l ron` oracle gate per page (under/over-emission, tandem-repetition, duplicate-block collapse) + `.verify.json` sidecars | `extract/ocr_verify.py` | silent omission is impossible now — every scanned page is scored |
| ToUnicode CMap + `/Differences` glyph-name rewrite for hacked QuarkXPress fonts (fixed PDFs in `db/pdf_fixed/`) | `extract/cmap_fix.py` | broken_2007 intra-word mojibake 6.4–7.6% → **0.00%** |
| Whole-act recovery from the PDF text layer when Docling drops column content (body-probe guarded) | `extract/md_act_recovery.py` | recovers acts Docling discards (MO 74/2017 lost 7 bodies) |
| Context-anchored closing-block injection (replaces name-anchor that fails when all acts share a signatory) | `extract/secondary_analyzer.py` | MO 74/2017 act numbers: scrambled with dups → clean 20–49 |
| Sumar↔acts reconciliation (completeness oracle) + title backfill | `extract/sumar_reconcile.py`, wired in `extract/pipeline.py` | missing/phantom acts now flagged in `extraction_warnings`; generic titles ("DECRET") → 0 for all sumar-bearing issues |
| Batch quality audit tool | `tools/ops/audit_extraction.py` | per-gazette table: coverage, mojibake, reconciliation, titles |

**Round 2 (same day):** scanned-era sumar parser (`_build_sumar_from_scanned_text`,
numbers/types only — titles come from act bodies via `backfill_title_from_body`
+ `sanitize_title` run-on truncation); `dedup_repeated_acts` (full-body
containment — prefix matching would merge template-twin decrees);
`repair_numbers_from_sumar` (duplicate numbers on template twins corrected
from the sumar sequence); fallback merge no longer re-appends acts whose
numbers the primary already has nor adopts run-on titles; ruled annex tables
via PyMuPDF `find_tables` with multi-page stitching (`extract/annex_tables.py`,
opt-in `extraction.annex_tables_fitz`, 294Bis: 34 stitched tables / 3,251 rows
in 8.5 s vs minutes of TableFormer).

**Known remaining limitations (flagged, not silent):**
- Generic titles: 4 left in the whole batch (PI_1/PI_2 1989 communiqués, one
  PI_75 act); scanned-era over-segmentation is flagged as phantoms
  (PI_6_1989: 13) but not pruned.
- 3 scanned pages still fail the OCR gate (glm-ocr repetition on dense
  pages; Tesseract fallback text used where it beats the VLM).
- A few real `MISSING act` flags remain (PI_1_2007: 3 — incl. Decret 1.422
  and CCR Decizia 831 whose number extraction picked 1816; PI_294/PI_311: 2
  each). Twin-number repair is positional — adjacent twins could swap
  numbers (warned per act).
- `extract-json` skips gazettes whose JSON already exists (sha-cache):
  delete `db/extracted/` (or the affected subtree) after pipeline changes.
- New config keys must be added to `legalro_core/config.py` too — Settings
  forbids unknown YAML keys (extra_forbidden).


Scope: PDF → MD → JSON for all 21 gazettes in `laws/` (eras: scanned 1989,
broken_2007, modern 2007/2017/2026). Local audit of `db/md_cache/` and
`db/extracted/` against the source PDFs, plus three web-research tracks
(scanned-OCR, native-PDF/encoding/tables, MD→JSON structuring & QA).

---

## 1. Audit findings (what is actually broken today)

### 1.1 Scanned 1989 — GLM-OCR via Ollama: **silent page-level data loss**

Evidence (verified against rendered pages):

| Gazette | Failure |
|---|---|
| MO 6/1989 p.1 | Page holds 21-entry two-column SUMAR **+ full text of Decret-Lege nr. 3** (electricity tariffs). MD has masthead + **one** sumar line (320 chars of a ~5,000-char page). DL 3 is absent from the JSON entirely; JSON has 0 sumar entries and a duplicated "DECRET 20". |
| MO 4/1989 p.1 | Full two-column body of Decret-Lege nr. 1 (abrogation of ~16 acts) dropped; only header + sumar survived. JSON has 1 act for a 4-act issue. |
| MO 3/1989 p.2 | Repetition loop — footer block transcribed 3×. |
| All 1989 | OCR misspellings in captured text (`decretăză`, `inființeažă`, `Salvăriii`); titles never captured; sumar=0 in every JSON. |

**Root cause (confirmed by research): bugs in the *Ollama port* of glm-ocr, not the model.**
- [ollama#14117](https://github.com/ollama/ollama/issues/14117) — generation aborts early ("token repeat limit reached") → truncation + silent omission.
- [ollama#14114](https://github.com/ollama/ollama/issues/14114) — images >2048×2048 are clipped; our 200 DPI page is ~1650×2340 px.
- The underlying [GLM-OCR model](https://huggingface.co/zai-org/GLM-OCR) is #1 on OmniDocBench v1.5 (94.62). Served correctly (vLLM/SGLang or the [Z.AI API](https://docs.z.ai/guides/vlm/glm-ocr), ≈$0.07/1k pages) it does not have these failures.
- There is **no completeness guard** anywhere in the pipeline, so the loss was invisible.

### 1.2 Broken_2007 — Docling text layer + static replacement table: **~7% of words still mojibake**

- Intra-word `'` (=ă) and `,` (=â) survive normalization: `urm'tor`, `Rom,niei`,
  `Hot'r,re`, `v,nzarea` — measured 6.4–7.6% of words in the Jan-2007 issues.
  These glyphs are ambiguous with real punctuation, so a blind table can never fix them.
  Corrupted text flows into chunks → embeddings → BM25.
- Two-column SUMAR collapses into one giant single-cell Markdown table.
- Docling drops closing blocks (16 had to be re-injected from the fitz layer in MO 1/2007).

**Root cause:** "hacked" Romanian DTP fonts (QuarkXPress era): diacritic glyph
*shapes* live in punctuation codepoints; the correct identity exists only at the
**(font, glyph_id)** level, which text-level normalization cannot see.

### 1.3 Modern — Docling + TableFormer: **reading order + lost titles + table bloat**

- Two-column reading order scrambled: MO 74/2017 act sequence comes out
  20, 21, 22, **25, 24, 27, 26, 31, 28, 33, 30, 35** … (column interleave).
  Risk: act bodies stitched across the wrong column.
- Orphan fragments mid-text (MO 294/2026: dangling `de studii universitare` after Art. 3).
- MO 294Bis/2026 (146-page table annex): MD inflates to 5× the PDF text volume;
  TableFormer is the wrong tool for ruled, regular, repeated-header tables.

### 1.4 MD→JSON (all eras)

- **Titles not extracted**: acts get `title:"DECRET"` even when the sumar parsed
  the real title ("Decret pentru numirea unui judecător"). Sumar titles are never
  joined back to acts.
- **Sumar↔acts reconciliation is advisory only**: 24 sumar vs 22 acts passes
  silently (MO 1/2007); phantom + missing acts both undetected.
- Gazette-level metadata (part, issue, date, era, sha256) **is** captured correctly.

---

## 2. Recommended target architecture

Classifier per document/page (extends existing era detection):
1. Born-digital vs scanned: text-layer coverage per page (PyMuPDF).
2. Broken-font class: `pdffonts` uni column + QuarkXPress font-name fingerprints
   + `ro_unigrams` hit-rate < ~85% + presence of `∫ ˛ Ó`.
3. Table-annex pages: ruled-line density / table-area ratio.
4. Post-extraction column-interleave sanity score.

Routing:

| Class | Pipeline |
|---|---|
| SUMAR page (all eras) | **Dedicated deterministic SUMAR parser** (pdfplumber words + x-gap column split + regex; for scanned era, a one-shot schema-constrained LLM call) → per-issue **act manifest** used as completeness oracle |
| Modern body pages | Docling **≥2.5x with `docling-layout-heron`** (new layout model, +20–24% mAP, built for multi-column) — may fix much of the reading order for free; TableFormer `FAST` |
| Broken-font era (2000–2008) | **Per-font glyph remap**: PyMuPDF `get_texttrace()` → (font, glyph_id) → render each glyph once → classify → cached font table → rewrite ToUnicode with pikepdf → Docling sees clean text. Residue → Romanian dictionary check + batched cheap-LLM diacritic repair ([arXiv:2511.13182](https://arxiv.org/pdf/2511.13182)). Unfixable PDFs → re-OCR (MinerU 2.5 / PaddleOCR-VL) |
| Scanned 1989 | GLM-OCR served correctly (**Z.AI API ≈$0.07/1k pages** or vLLM on rented GPU, 300 DPI) — or Gemini 3 Flash for best accuracy (~$5–9/1k pages). Local-only alternative: Qwen3-VL-8B with region-wise OCR after Surya layout |
| Annex table pages | **Camelot 2.0 lattice + `stack_contiguous()`** (multi-page stitch, drop repeated headers) → JSON directly; GMFT fallback for unruled tables; one batched-LLM pass for column-schema labeling only |
| Reading-order failures | Per-page escalation to **MinerU 2.5** (reading-order edit-dist 0.130, best-in-class), merge back |

### Always-on verification layer (the most important change)

1. **Tesseract `-l ron --psm 1` oracle on every scanned page** (free, local,
   cannot hallucinate or silently drop): gate = VLM output word count ≥75% of
   Tesseract count AND no 40-char shingle repeated ≥3× AND ≥90% of Tesseract's
   confident words fuzzy-found in output. Fail → re-OCR region-wise → escalate.
2. **Sumar as completeness oracle** (bidirectional): every sumar entry must
   match exactly one act (type+number+year, then title similarity) — unmatched
   sumar entry = missing act (the OCR-drop case, now *detectable*); unmatched
   act = phantom candidate. Page-range vs text-length sanity per act.
3. **Gold set of 30–50 hand-checked issues** (stratified by era) + field-level
   precision/recall in CI as a regression gate; CER sampling on 1% of bodies.
4. Sampled **LLM-as-judge** (different model than extractor) on rendered page
   images vs extracted JSON.

### MD→JSON structuring

- Keep rule-based segmentation as Tier 1. Widen the LLM repair trigger to:
  field-validation failure ∪ sumar-reconciliation failure ∪ page-coverage anomaly.
- Repair/sumar model: **Gemini 3 Flash** (native `response_schema`, 1M ctx,
  Batch API −50%) with **Claude Haiku 4.5** as cross-check judge. Two-step
  "reason free-text first, format JSON second" (+10–15% quality).
- **Hard rule: title must be non-empty and ≠ doc-type keyword** — backfill from
  the sumar manifest (positional + number match) before any LLM call.
- Schema: stay custom JSON, but adopt Akoma Ntoso *names* for fixed hierarchy
  levels (`titlu/capitol/sectiune/articol/alineat/litera`, each with `num`,
  `heading`, `text`, stable eId like `art_5__alin_2__lit_a`) — no recursive
  schema (constrained decoding doesn't support it). Add an **ELI-style URI**
  per act (`/eli/ro/{type}/{year}/{number}`) and provenance fields
  (extraction path, validation status, sumar-match status).
- Chunking: articol-level, prepend one-line act header to embedded text
  (Summary-Augmented Chunking), `references_out`/`modifies` citation edges
  via regex (Romanian citation syntax is regular).

---

## 3. Prioritized implementation plan

1. **Stop using glm-ocr through Ollama.** Re-OCR the 1989 set via Z.AI GLM-OCR
   API (or Gemini 3 Flash) at 300 DPI; add the Tesseract coverage gate +
   repetition-shingle detector so silent omission can never pass again.
2. **Sumar-as-oracle reconciliation** (deterministic sumar parser → act
   manifest → bidirectional match; kills phantom *and* missing acts) +
   title backfill from sumar.
3. **Upgrade Docling to layout-heron**; re-extract one 2017 issue and check the
   act-order scramble before/after.
4. **Per-font glyph remap for broken_2007** (get_texttrace + cached font tables
   + pikepdf ToUnicode rewrite); dictionary-gated LLM repair for the residue.
5. **Camelot lattice route for table annexes** (294Bis-class), bypassing
   TableFormer.
6. **Gold set + CI metrics gate**; eId/ELI identifiers + chunk header
   augmentation + citation edges.

Full source citations are embedded in the three research-agent reports
(scanned-OCR, native-PDF, MD→JSON) — key ones inline above.
