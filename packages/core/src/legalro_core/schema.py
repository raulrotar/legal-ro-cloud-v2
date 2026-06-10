"""Canonical schema contract — the single source of truth shared by every package.

Anything that, if duplicated, would silently break retrieval lives here:
  * version stamps (so re-processing paths are independent and auditable),
  * collection names,
  * deterministic _id derivation,
  * modality mapping (Era -> M1/M2/M3).

processing STAMPS these onto every document it writes; serving + dashboard READ
them. Never re-define collection names or id formats anywhere else.
"""
from __future__ import annotations

# ── Version triple (see spec §4.3) ────────────────────────────────────────────
# Bump SCHEMA_VERSION when field shapes change           -> full re-extraction.
# Bump PIPELINE_VERSION when extraction logic changes     -> full re-extraction.
# Bump EMBEDDING_VERSION when chunker OR model changes     -> re-embed only.
SCHEMA_VERSION = "2.0.0"
PIPELINE_VERSION = "2.0.0"
# chunker-3.0: Phase-0 pre-embedding cleanup + 512-token article chunks
# (docs/EMBEDDINGS_PLAN.md)
EMBEDDING_VERSION = "chunker-3.0+bge-m3-1024"

# ── MongoDB collections ───────────────────────────────────────────────────────
# Keep the v1 names that the current retrieval code already queries.
COLL_GAZETTES = "gazettes"      # one doc per issue (metadata + sumar)
COLL_CHUNKS = "chunks"          # retrieval unit: text + embedding + facets
COLL_EDGES = "graph_edges"      # citations / promulgation (GraphRAG)
COLL_RUNS = "runs"              # one doc per processing batch (dashboard source)
COLL_QUERY_LOG = "query_log"    # serving: who/when/query/returned ids (dashboard)

# ── Atlas search index names (must match scripts/setup_indexes) ───────────────
INDEX_VECTOR = "chunks_vector"
INDEX_SEARCH = "chunks_search_ro"

# ── Modality (spec §1.1) derived from the existing Era enum ───────────────────
# M1 = scanned (OCR), M2 = legacy born-digital (mojibake repair), M3 = modern.
ERA_TO_MODALITY = {
    # Era enum values are lowercase strings (Era.SCANNED.value == "scanned")
    "scanned": "M1_SCANNED",
    "broken_2002": "M2_LEGACY_DIGITAL",
    "broken_2007": "M2_LEGACY_DIGITAL",
    "hybrid": "M3_MODERN_DIGITAL",   # mixed; treated as modern unless a page is facsimile
    "modern": "M3_MODERN_DIGITAL",
}


def modality_for_era(era: str) -> str:
    return ERA_TO_MODALITY.get(era.lower(), "M3_MODERN_DIGITAL")


# ── Deterministic id derivation (idempotent upserts depend on this) ───────────
def doc_id(issue_number: int, year: int, is_bis: bool = False, part: str = "I") -> str:
    """Stable issue id, e.g. 'MO_PI_295_2026' or 'MO_PI_294Bis_2026'."""
    suffix = "Bis" if is_bis else ""
    return f"MO_P{part}_{issue_number}{suffix}_{year}"


def act_id(issue_id: str, act_index: int) -> str:
    return f"{issue_id}::act-{act_index:03d}"


def chunk_id(act_identifier: str, chunk_index: int) -> str:
    return f"{act_identifier}::chunk-{chunk_index:03d}"
