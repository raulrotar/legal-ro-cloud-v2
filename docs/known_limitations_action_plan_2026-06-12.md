# Known Limitations — Expert Review & Action Plan (2026-06-12)

Three parallel investigations: 1989 scanned-era forensics, modern-era
forensics, and a 2025–2026 technology radar (web research). This document
synthesizes them into a prioritized plan. Full agent evidence (file:line) is
embedded below; no code was changed during the review.

## Headline findings

1. **The two coverage outliers are a metric artifact, not data loss.**
   `MO_PI_294Bis_2026` (0.070) and `MO_PI_822_2007` (0.288) have their annex
   content fully extracted as `financial_table` chunks (34 stitched tables /
   3,251 rows; 894-row beneficiary list) — but `build_issue_docs`
   (`prepare/build.py:224`) never adds table markdown to `mapped_texts`, so
   the coverage ratio ignores it. Reconstructed: 20,237/275,009 = 0.0736 ✓
   and 26,098/87,084 = 0.2997 ✓.
2. **`'999/726'` is legitimate joint-ministerial numbering** (Ordin comun:
   MS nr. 999 + CNAS nr. 726). `fold_act_number` (`act_number.py:41`)
   already preserves compound numbers; `is_malformed_act_number`
   (`act_number.py:99`) doesn't know the rule and false-flags it — and the
   false flag triggers a wasted glm-ocr repair call that returns garbage.
3. **Most 1989 validation errors are regex morphology bugs, not OCR limits.**
   The 1989 gazettes print *COMUNICATE / COMUNICATUL CĂTRE ȚARĂ / DECRETE-LEGE*
   (plural/articulated); the classifiers match only singular `COMUNICAT\b`,
   `DECRET\b` (`metadata.py:14-24`, `md_segmenter.py:31-45`,
   `normalize.py:281-305`). Plus one OCR-split token (`DE CRET`). Fixing the
   morphology resolves ~5 of 7 annotated acts; `ACT_NUMBER_ZERO` then
   auto-downgrades to INFO for communiqués (`extraction_validator.py:171`).
4. **The vision-repair pass runs with Ollama's default 4096 context** —
   `llm_repair.py:184` sets only `temperature: 0`, no `num_ctx`, although a
   72-dpi page image is ~4K tokens (the OCR path sets `num_ctx: 16384`,
   `md_extractor.py:224`). Prompt truncation explains today's 4 malformed
   repair JSONs on PI_6. No `seed` is passed anywhere.
5. **The only true data loss found:** the 1989 COMUNICAT bodies.
   `_split_by_headings` (`md_segmenter.py:254`) discards all text before the
   first promoted boundary, and communiqué headings are never promoted —
   `MO_PI_5_1989` sumar entry 5 correctly flags the loss (the oracle works;
   the body side loses the act).
6. **The PI_5 "phantoms" are glm-ocr fence-echo duplicates** (re-emitted
   ```markdown blocks duplicating earlier content). Existing dedup only
   collapses adjacent ≤8-line blocks (`ocr_verify.py:111`), echoes sit
   15–25 lines away.
7. **Tech radar:** the 1989 OCR ceiling is an *Ollama serving artifact*, not
   a model ceiling. The same GLM-OCR weights are #1 on OmniDocBench v1.5
   (94.62) when served via vLLM/SGLang, which also offers true deterministic
   decoding (seed + batch-invariant kernels). Whole-scanned-corpus re-OCR on
   a vast.ai GPU costs single-digit dollars; Z.AI's API runs $0.03/MTok.
   **Camelot benchmarks at ~73% vs Docling TableFormer's 91%+ — drop the
   open item.** For degraded historical print, Stanford's CHURRO (3B, 46
   language clusters) is the one specialist model worth a bake-off.

## Prioritized action plan

### P0 — trivial, immediate (each ≤ ~10 lines)
| # | Fix | Where | Effect |
|---|---|---|---|
| 1 | Count table markdown in coverage (or emit separate `table_coverage`) | `prepare/build.py:179-225` | coverage outliers disappear; metric becomes honest |
| 2 | Joint-ministerial rule in `is_malformed_act_number` + exclude pattern from LLM-repair fixable set | `act_number.py:99`, `llm_repair.py:37` | kills false `ACT_NUMBER_MALFORMED` + wasted repair calls |
| 3 | `options={"temperature":0, "seed":42, "num_ctx":16384, "num_predict":512}` in repair; plumb fields through `RepairLLMConfig` | `llm_repair.py:184`, `config.py:98` | likely converts failed repairs into usable patches; removes silent truncation |

### P1 — small (a day, one re-extraction + audit diff to validate)
| # | Fix | Where | Effect |
|---|---|---|---|
| 4 | Widen keyword morphology: `DECRET(?:E)?(?:\s*-\s*LEGE)?`, `COMUNICAT(?:UL|E)?`, OCR-split `DE\s?CRET`; add communiqué cues (`al Consiliului`, `către țară`) to `_ACT_NUMBER_CUE`; keep plurals in `_SKIP_HEADING` as section markers | `metadata.py`, `md_segmenter.py`, `normalize.py` | resolves ~5/7 of the 1989 DOC_TYPE_UNKNOWN / ACT_NUMBER_ZERO annotations |
| 5 | `annex_tables_fitz` trigger: replace "≥20 regions" with row-based threshold (≥N total rows or any region ≥100 rows) | `pipeline.py:130` | 822-class single-giant-annex issues get the better fitz extraction |
| 6 | Reconcile pass 3: replace positional `zip` with compatible-type + `_norm_nr` search; prefer longer body on duplicate numbers | `sumar_reconcile.py:163` | recovers MO_PI_1_2007's HG 1.919 match (body already extracted) |

### P2 — medium (scanned-era robustness)
| # | Fix | Where | Effect |
|---|---|---|---|
| 7 | Fence-echo collapse at normalize (scanned era): strip ``` fences, fold non-adjacent duplicate paragraphs within ±1 page, gate on `Nr. N.` closings | `pipeline.py:549` (`_normalize_gazette_md`) | kills PI_5 phantoms, PI_1 duplicate page, PI_6 loop residue at the source |
| 8 | Sumar-driven recovery of pre-first-boundary text: when a MISSING entry's title tokens appear in the discarded region, synthesize the block | `md_segmenter.py:254` + `sumar_reconcile` | recovers COMUNICAT bodies — the only real data loss found; respects legacy_junk_filter |
| 9 | Party-financing filings (Legea 334/2006): match number-less sumar entries against extracted tables by title/page, or mint lightweight table-only acts | `sumar_reconcile.py:111`, `table_triage` | clears MO_PI_311's "missing" entry; QA Q55-class questions gain a citable act |

### P3 — strategic (when scaling to the 50k corpus / VPS phase)
| # | Move | Cost | Effect |
|---|---|---|---|
| 10 | Re-serve GLM-OCR via vLLM (greedy + seed + batch-invariant, pinned version/GPU) on a vast.ai rental for the scanned subset; keep Ollama for dev only | single-digit $ per full re-OCR | determinism + escapes the documented Ollama glm-ocr bugs (fence echo, repetition loops, tile drops) |
| 11 | CHURRO bake-off on the 50 worst 1989 pages; Mistral OCR 3 batch (~$0.50–1/1k pages) as disagreement arbiter | hours + pennies | evidence-based choice for the historical-print ceiling |
| 12 | Drop the Camelot open item; if a table fallback is ever needed, use a VLM table pass (GLM-OCR/PaddleOCR-VL) with header-match stitching | — | avoids integrating a tool that benchmarks 18+ points below what's already shipped |

## Implementation status (same day)

**P0 — done.** Coverage counts tables (`coverage_min` 0.070 → **0.954**, both
outliers at 1.000); joint-ministerial numbers clean (0 MALFORMED flags, repair
no longer wastes calls on them); repair runs with `seed/num_ctx/num_predict`
(5 successful field patches vs ~0 usable before; leftover unparseables are the
glm-ocr JSON-discipline ceiling → P3).

**P1 — done, plus two pulled-forward fixes.** 1989 morphology widened
(metadata + segmenter + promoter cue); annex-fitz row-based trigger (fires on
822, kept the richer markdown per the existing quality guard); reconcile pass
3a number-aware. Implementing surfaced three deeper blockers, all fixed:
(a) blank-line gate — act headings directly under a bare section line
(`COMUNICATE`) now promote; (b) `_is_artefact` publisher-footer rule killed a
3,190-char communiqué ending with the colophon — size-gated to <1000 chars +
CFSN body signal added; (c) **pre-boundary discard** (P2 item 8 pulled
forward): scanned-era OCR emits headingless acts before the masthead — kept as
a lead block when ≥300 chars with an act-body signal; recovered the PI_2
cease-fire communiqué (QA Q34) and PI_4/PI_5 fragments (phantom-flagged).
RECTIFICĂRI errata, previously silently dropped, now extracted and classified.

**Scoreboard:** validator-clean issues 11 → **14**/21 (1989 era fully clean
except the quarantined colophon); coverage_min 0.070 → **0.954**; QA
**51/52 effective** (Q43 unchanged — generation-layer; Q24 flips
CORECT/PARTIAL across runs with correct data in Mongo, same generation-layer
class). Remaining annotated acts are P2 items: fence-echo duplicates (item 7),
reconcile-after-fallback ordering for HG 1.919 (item 6 extension), party
filings (item 9), plus modern-era fallback fragments.

**P2 — done.** Fence-echo collapse at normalize (scanned era: strips glm-ocr's
re-emitted ```markdown fences + non-adjacent duplicate paragraphs ≥120
word-chars without Nr. closings — PI_5's 7 echoes gone at source);
party-financing filings and wrapped continuation lines downgraded MISSING →
INFO in reconcile (anchored on the Legea 334/2006 section pattern — Table.page
proved unpopulated, and categories arrive as wrapped fragments);
post-fallback-merge MISSING resolution in `fallback_merge.py` (HG 1.919 class —
now also matched at pipeline time by pass 3a). Validation forced two more
fixes: lead-block trim at the masthead (sumar lines contain the CFSN phrase
and minted a phantom), and **position-based doc_type classification** in
`md_rule_extractor.py` (earliest keyword in the block wins, DCC kept as a
context override on DECIZIE) — without it the plural-widened DECRET pattern
re-classified the PI_5 communiqué from the next-section "DECRETE" line.
`având în vedere` joined the promoter cue, recovering PI_5's
extraordinary-tribunals communiqué: **PI_5 reconciles 5/5 — the last known
real data loss is closed**. Final QA: **51/52, only Q43** (generation-layer)
— Q24/Q34 both pass.

## Validation loop for P0–P2

After each batch: full re-extraction (md_cache retained), then
`tools/ops/audit_extraction.py` diff against today's baseline
(`/tmp/extract_gemma4.log` figures: 11/21 clean, 7 annotated 1989 acts,
3 unmatched sumar entries, 2 coverage outliers), then the 52-question QA
suite. Success criteria: ≥17/21 clean validations, 0 unmatched sumar entries
in modern issues, coverage_min ≥ 0.79, QA ≥ 51/52.

## Sources (tech radar)

- GLM-OCR: github.com/zai-org/GLM-OCR · vLLM recipe: docs.vllm.ai/projects/recipes/en/latest/GLM/GLM-OCR.html · Z.AI pricing: docs.z.ai/guides/overview/pricing
- vLLM determinism: docs.vllm.ai/en/latest/usage/reproducibility/ · batch invariance: docs.vllm.ai/en/latest/features/batch_invariance/ · thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Ollama nondeterminism: github.com/ollama/ollama/issues/586, /issues/5321
- CHURRO: arxiv.org/pdf/2509.19768 · github.com/stanford-oval/Churro
- olmOCR-2: allenai.org/blog/olmocr-2 · DeepSeek-OCR-2: github.com/deepseek-ai/DeepSeek-OCR-2
- PaddleOCR-VL: huggingface.co/PaddlePaddle/PaddleOCR-VL
- Mistral OCR 3: mistral.ai/news/mistral-ocr-3/ · Gemini pricing: ai.google.dev/gemini-api/docs/pricing
- Docling/TableFormer vs Camelot: arxiv.org/pdf/2408.09869 · codecut.ai/docling-vs-marker-vs-llamaparse/
- Gazette projects: github.com/okfn-brasil/querido-diario · github.com/mkoniari/LegalParser (Greek FEK sumar-as-oracle analogue) · github.com/worldwidelaw/legal-sources
