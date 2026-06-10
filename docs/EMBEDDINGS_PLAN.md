# Embeddings & Chunking — Research Findings + Implementation Plan (2026-06-11)

Inputs: two research reports (local Romanian embedding models; chunking
necessity/strategy) + the extraction-evaluator's embedding-readiness audit.
Full citations in the research transcripts; key sources inline.

## Verdicts (evidence-backed)

| Question | Verdict |
|---|---|
| Switch embedder? | **No — keep bge-m3 dense via Ollama.** No Romanian benchmark exists; switching buys ~1–3 pts at real re-embed cost. A/B Qwen3-Embedding-0.6B later on the QA set. |
| Is chunking still necessary? | **Yes.** Positional bias + dilution make whole-act embedding lose late content (annexes). 8K ctx is for not truncating long articles, not for skipping chunking. |
| Chunk unit | **Article-level, ≤512 tokens** (bge-m3 dense sweet spot; legal benchmarks converge on cited-unit boundaries). Structure-aware chunking ≈ doubles nDCG vs fixed splitting with bge-m3 (arXiv 2603.06976). |
| Advanced technique | **SAC** (summary-augmented chunking, arXiv 2510.06999): 1–2 sentence *generic* act summary prepended to every chunk — halves wrong-act retrieval; one local-LLM call per act. Skip late chunking (underperforms on bge-m3), per-chunk contextual retrieval (cost), RAPTOR, semantic-breakpoint chunking. |
| bge-m3 sparse/ColBERT | Skip — Ollama can't emit them; our BM25 already covers the +1–2 pt lexical gain. |
| Reranker | **bge-reranker-v2-m3 on top-20, behind an A/B gate** — strongest single lever in most studies (+12–17pp recall/MRR legal) but LegalBench-RAG saw rerankers *hurt* legal retrieval → measure on our QA set before keeping. |
| Storage | **binData int8 vectors now** (float32 arrays = 2.6GB at 200k chunks; int8 ≈ 205MB). M0 cannot hold the 50k corpus text regardless → Atlas Flex ($8–30/mo) or local Qdrant/LanceDB at scale. |
| Metadata prefix | Keep — it's Anthropic's contextual-embedding pattern (−35% failures). Enrich with the SAC summary. |
| Overlap | 0 for structural chunks; ≤80 tokens for fallback splits (current 100 is wasted index). |
| Parent expansion | Fill the 12K budget **centered on the retrieved chunk** (header + article ± neighbors), not head-truncated full text. |

## Implementation plan (phases, each gated on the QA question set)

**Phase 0 — pre-embedding cleanup (from the extraction evaluation; do first):**
- chunk-time regex strip: running headers (`Monitorul Oficial al României, Partea I, Nr.…`), literal `page-break` markers, EDITOR colophon tails;
- quarantine acts from pages whose `verify.json` failed (MO_PI_6 loop act);
- fuzzy-body dedup at build time (MO_PI_75 duplicated annex; 1989 phantom twins).

**Phase 1 — chunking overhaul (`prepare/chunk.py`):**
- raise `articles[]` parsing coverage 30%→≥90% (deterministic Art./alin./Anexă regexes — biggest payoff per hour);
- chunk = article; split >512-token articles at alineat boundaries; merge tiny consecutive articles into 256–512-token groups;
- MAX_TOKENS 1024→512, OVERLAP 100→0 structural / 80 fallback;
- fallback = recursive splitting on paragraph boundaries (not semantic chunking).

**Phase 2 — SAC summaries:**
- one generic 1–2 sentence summary per act via local Qwen3/llama (Ollama), cached in the JSON;
- `text_embedded` = `[type | authority | MO id | year | summary] chunk`; also append summary to `text_normalized` for the contextual-BM25 effect.

**Phase 3 — embed + store:**
- bge-m3 unchanged; write vectors as BSON binData int8 (`quantization: scalar` index);
- bump `EMBEDDING_VERSION` (chunker-3.0+bge-m3-int8); full re-embed (~200 chunks now, trivial).

**Phase 4 — retrieval:**
- chunk-centered parent expansion;
- reranker A/B: bge-reranker-v2-m3 (sentence-transformers/ONNX int8, inputs truncated 256 tokens) over top-20 RRF → keep only if QA accuracy improves.

**Phase 5 — scale prep (50k):**
- A/B Qwen3-Embedding-0.6B vs bge-m3 on the QA set;
- decide Atlas Flex vs local vector store; reranker ONNX on VPS.
