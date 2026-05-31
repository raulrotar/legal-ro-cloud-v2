# Future: BGE-M3 Sparse Retrieval

## What is it
BGE-M3 natively outputs three vector types from a single forward pass:
- **Dense** (1024-dim) — used in current implementation
- **Sparse** (SPLADE-style weights) — lexical retrieval, morphologically aware
- **Multi-vector / ColBERT** — token-level late interaction (impractical at scale)

## Why it matters for Romanian legal text
The current BM25 index (`chunks_search_ro`) uses Atlas Search with a Snowball stemmer for Romanian. The stemmer is adequate but misses morphological edge cases (diacritics, legal jargon, verb forms). BGE-M3's sparse head learns subword frequency directly from training data — it is natively aware of Romanian morphology without a hand-crafted analyzer.

## What would need to change
1. Switch from `sentence-transformers` to `FlagEmbedding` (`BGEM3FlagModel`) to access sparse outputs
2. Store sparse vector per chunk in MongoDB (as a dict of `{token_id: weight}` or a sparse array)
3. Either:
   - Replace Atlas `$search` (BM25) with a sparse vector dot-product query
   - Or run both and merge with RRF (dense + BGE-M3-sparse, drop Atlas BM25 entirely)
4. Update `scripts/setup_indexes.py` to create a sparse-compatible index
5. Update `retrieval/search.py` to handle the new sparse retrieval path

## Open questions to research
- Does MongoDB Atlas support sparse vector storage and dot-product queries natively, or does it require a workaround?
- Storage cost: sparse vectors are variable-length dicts — what is the avg size per chunk on Romanian legal text?
- FlagEmbedding vs sentence-transformers API: which is better maintained and HF Spaces compatible?
- Benchmark: does BGE-M3 sparse actually outperform the Romanian Snowball BM25 on Gazette-style queries?

## References
- [BGE-M3 paper](https://arxiv.org/abs/2309.07597)
- [FlagEmbedding BGEM3FlagModel docs](https://github.com/FlagOpen/FlagEmbedding)
- [MongoDB Atlas sparse vector support](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
