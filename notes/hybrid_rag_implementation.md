# Hybrid RAG Implementation — Generation + Retrieval

**Date recorded:** 2026-05-23  
**QA score:** 22/28 CORECT, 4 PARTIAL, 2 GRESIT, 0 EROARE  
**Model:** `mlx-community/Qwen3.5-9B-4bit` (local, MLX, thinking enabled for Stage A)

---

## What changed vs the pydantic-ai agent baseline

| Component | Before | After |
|-----------|--------|-------|
| Generation path | `agent.run_sync()`, no timeout | `run_query_hybrid()`: async agentic + fallback |
| Agentic budget | unlimited / `max_tokens` from config | 4096 tokens + 90s wall-clock timeout |
| Fallback | none | `run_query()`: single-turn, no thinking |
| RRF score on chunks | not attached (lost after merge) | `rrf_score` field on every returned chunk |
| Parent-doc expansion | score-threshold (`vector_score >= 0.75`) | rank-based (`rank < parent_doc_top_n=3`) |
| Pipeline candidate limit | 20 per source | 40 per source |
| Metadata boost | none | additive `rrf_score` boost for `act_number`/`act_year`/MO matches |

---

## Architecture

### Generation (`src/legalro/generation/agent.py`)

```
run_query_hybrid(question, settings)
├── Stage A: asyncio.wait_for(agent.run(..., max_tokens=4096), timeout=90s)
│   ├── Qwen3 thinking ON
│   ├── pydantic-ai Agent with search_law tool
│   └── OpenAI http client timeout = agentic_timeout + 10s (backstop)
└── on UnexpectedModelBehavior / TimeoutError / httpx error → Stage B
    └── run_query(question, settings)
        ├── hybrid_search → assemble_context
        ├── max_tokens=2048, thinking OFF (chat_template_kwargs)
        └── httpx direct call, timeout=120s
```

Stage A was NOT triggered as fallback in the 2026-05-23 QA run — all 28 questions completed via the agentic path within budget. Stage B exists as a safety net.

### Retrieval (`src/legalro/retrieval/search.py`)

- `hybrid_search()` calls `_python_rrf_search()` (or `_rankFusion` if enabled).
- `_python_rrf_search()` runs MongoDB `$vectorSearch` (limit 40) + `$search` BM25 (limit 40), then `_rrf_merge()`.
- `_rrf_merge()` now attaches `doc["rrf_score"] = score` to every returned doc.
- After merge, `_apply_metadata_boost()` parses `nr. <num>/<year>` and `MO nr. <num>` from the query and adds a small boost (`_METADATA_BOOST = 0.005`) to chunks whose `act_number`/`act_year`/`source_issue_id` match.
- Final result: `settings.search.limit` (10) docs, each with `rrf_score`.

### Context assembly (`src/legalro/retrieval/context.py`)

- Top `parent_doc_top_n` (default 3) chunks by RRF rank with `act_full_text` expand to the full act.
- Full act text is truncated to `max_parent_chars` (8000 chars) to prevent token overflow.
- Deduplication: same act appears only once even if multiple chunks qualify.

### Config (`src/legalro/config.py`, `config/local.yaml`)

```yaml
llm:
  agentic_max_tokens: 4096   # max tokens for Stage A thinking + answer
  agentic_timeout: 90.0      # wall-clock seconds before Stage A is cancelled
search:
  parent_doc_top_n: 3        # top-N chunks that expand to full act text
```

---

## QA score comparison

| Run | CORECT | PARTIAL | GRESIT | EROARE | Notes |
|-----|-------:|--------:|-------:|-------:|-------|
| Agentic + thinking (4096) | 22 | 3 | 0 | 3 | Crashes on Q11/Q23/Q26 |
| Single-turn, no thinking | 22 | 4 | 2 | 0 | Q18 regression |
| **Hybrid (current)** | **22** | **4** | **2** | **0** | Q11 CORECT, Q18 CORECT |

---

## Known limitations

### Q9 — CORECT → PARTIAL regression
**Expected:** "George Daniel Subțirelu, Tribunalul București, Secția a V-a civilă, Dosar nr. 22.190/299/2006"  
**Issue:** The hybrid agentic path returns a PARTIAL — likely missing the dosar number. Possibly the rank-based parent-doc expansion for top-3 pulls in a large act that buries the specific dosar detail, or the 8k truncation cuts it off.  
**Fix direction:** Reduce `parent_doc_top_n` to 2, or increase `max_parent_chars`, and re-run QA.

### Q23 — CORECT (single-turn) → PARTIAL (hybrid)
**Expected:** "90 lei de persoană"  
**Issue:** HG 1447/2007 Norme financiare sport is ~54k chars. When the act is included as parent doc (truncated to 8k), the relevant article (Art. 4 alin. (2) at ~90 lei) may fall beyond the 8k cut — or the agentic model reasons differently than the single-turn path.  
**Fix direction:** Ensure the specific chunk containing "90 lei" is within the top-3 and not displaced by parent-doc expansion of a different act.

### Q26 — GRESIT (persists)
**Expected:** List of documents for taxi authorization  
**Issue:** Fundamental question/corpus mismatch — Q26 asks about "Ordinul nr. 346/2007" but the actual taxi-norme act in the corpus is **ORDIN 356/2007** (law_id `ORDIN_356_2007`, source `PI_820_2007`). The literal "346" search surfaces ORDIN_346_2007 (strategic exports) first. The metadata boost helps with MO match (both in PI_820) but not enough to override the act-number mismatch for specific content queries.  
**Fix direction:** Either correct the question numbering (346→356) or add a `law_id` alias mapping in the corpus.

### Q28 — PARTIAL → GRESIT regression
**Expected:** "Art. 108 din Constituție și art. 30 alin. (8) din Legea nr. 199/2023"  
**Issue:** HG 191/2026 is not ingested (corpus gap — MO 294/2026 missing). The agentic model, finding no relevant chunks, fabricates a plausible-sounding answer instead of saying "not found." The single-turn path (which just passes whatever chunks exist) was better here because it at least found related context that produced a PARTIAL.  
**Fix direction:** Ingest MO 294/2026 (parent gazette of HG 191/2026). Alternatively, add a "not found" guardrail to the system prompt.

### Q11/Q25/Q26 structural issue — act number mismatch
The test questions reference "Ordinul nr. 346/2007" for the taxi norme, but the corpus has this act as **ORDIN 356/2007**. This is a persistent source of confusion for all three questions. The metadata boost mitigates it for Q11/Q25 (where topic context is strong) but not Q26.

### Stage A never fell back in this run
All 28 questions completed via Stage A (agentic) within the 90s/4096-token budget. This means:
- Stage B (single-turn fallback) is untested under real QA conditions.
- Questions that previously crashed (Q11/Q23/Q26) now complete agentic successfully — possibly because the 40-candidate pool returns better chunks, reducing the model's uncertainty and thus thinking length.
- If model behavior changes (e.g. different prompt, longer thinking), Stage B will activate.

---

## Next steps to reach 23+/28

1. **Ingest MO 294/2026** → fixes Q27/Q28 corpus gap (potential +2 CORECT/PARTIAL).
2. **Tune `parent_doc_top_n`** from 3→2 → potentially recovers Q9 and Q23.
3. **Fix Q26 data** — correct act number in test questions (346→356) or add a corpus alias.
4. **Add "not found" guardrail** to SYSTEM_PROMPT to prevent hallucination on corpus gaps (Q28).
