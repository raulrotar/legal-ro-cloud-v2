"""Hybrid search: vector + BM25 + RRF fusion."""
import re
from legalro_core.config import Settings
from legalro_core.embeddings import embed_texts
from legalro_core.store import get_db

_ACT_NR_RE = re.compile(r'\bnr\.?\s*([\d.]+\d)/(\d{4})\b', re.IGNORECASE)
_MO_NR_RE = re.compile(r'\bMO\b.*?nr\.?\s*(\d+)', re.IGNORECASE)
_PI_RE = re.compile(r'\bPI_(\d+)_(\d{4})\b', re.IGNORECASE)

_PIPELINE_LIMIT = 80  # candidates per pipeline before RRF merge
_METADATA_BOOST = 0.005  # small additive boost for act-number/MO matches

# Bare "Decizia 922 2007" or "Decizia nr. 922 din 2007" — no slash
_ACT_NR_BARE_RE = re.compile(
    r'\b(?:decizia|hotararea|hotărârea|ordinul|decretul|legea)\s+(?:nr\.?\s*)?(\d+)(?:\s+din)?\s+(\d{4})\b',
    re.IGNORECASE,
)


def _parse_query_metadata(query: str) -> dict:
    """Extract act number, year, and MO number hints from a natural-language query."""
    meta = {}
    m = _ACT_NR_RE.search(query)
    if m:
        meta["act_number"] = m.group(1).replace(".", "")
        meta["act_year"] = int(m.group(2))
    if not meta.get("act_number"):
        # Try bare form: "Decizia 922 2007" (no slash)
        m = _ACT_NR_BARE_RE.search(query)
        if m:
            meta["act_number"] = m.group(1).replace(".", "")
            meta["act_year"] = int(m.group(2))
    m = _MO_NR_RE.search(query)
    if m:
        meta["mo_number"] = m.group(1)
    return meta


def _apply_metadata_boost(docs: list[dict], meta: dict) -> list[dict]:
    """Add a small boost to chunks whose metadata matches query hints."""
    if not meta:
        return docs
    act_num = meta.get("act_number")
    act_year = meta.get("act_year")
    mo_num = meta.get("mo_number")
    for doc in docs:
        boost = 0.0
        if act_num and str(doc.get("act_number", "")) == act_num:
            boost += _METADATA_BOOST
        if act_year and doc.get("act_year") == act_year:
            boost += _METADATA_BOOST * 0.5
        if mo_num:
            issue = doc.get("source_issue_id", "")
            if mo_num in issue:
                boost += _METADATA_BOOST
        if boost:
            doc["rrf_score"] = doc.get("rrf_score", 0.0) + boost
    # Re-sort after boost
    if any(meta.values()):
        docs.sort(key=lambda d: d.get("rrf_score", 0.0), reverse=True)
    return docs


def hybrid_search(
    query: str,
    settings: Settings,
    act_type: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict]:
    db = get_db(settings)
    query_embedding = embed_texts([query], settings, is_query=True)[0]

    vector_filter = {}
    if act_type:
        vector_filter["document_type"] = act_type.upper()
    if year_from is not None or year_to is not None:
        year_range = {}
        if year_from is not None:
            year_range["$gte"] = year_from
        if year_to is not None:
            year_range["$lte"] = year_to
        vector_filter["act_year"] = year_range

    if settings.search.use_rank_fusion:
        results = _rank_fusion_search(db, query, query_embedding, vector_filter, settings)
    else:
        results = _python_rrf_search(db, query, query_embedding, vector_filter, settings,
                                     year_from=year_from, year_to=year_to)

    meta = _parse_query_metadata(query)
    return _apply_metadata_boost(results, meta)


def _python_rrf_search(
    db, query: str, query_embedding: list[float],
    vector_filter: dict, settings: Settings,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict]:
    vector_pipeline = [
        {
            "$vectorSearch": {
                "index": "chunks_vector",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": settings.search.num_candidates,
                "limit": _PIPELINE_LIMIT,
                **({"filter": vector_filter} if vector_filter else {}),
            }
        },
        {"$addFields": {"vector_score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0}},
    ]
    vector_results = list(db.chunks.aggregate(vector_pipeline))

    text_pipeline: list[dict] = [
        {
            "$search": {
                "index": "chunks_search_ro",
                "text": {
                    "query": query,
                    "path": ["text", "text_normalized", "title"],
                },
            }
        },
        {"$addFields": {"text_score": {"$meta": "searchScore"}}},
        {"$project": {"embedding": 0}},
        {"$limit": _PIPELINE_LIMIT},
    ]
    if year_from is not None or year_to is not None:
        year_match: dict = {}
        if year_from is not None:
            year_match["$gte"] = year_from
        if year_to is not None:
            year_match["$lte"] = year_to
        text_pipeline.insert(-1, {"$match": {"act_year": year_match}})
    text_results = list(db.chunks.aggregate(text_pipeline))

    return _rrf_merge(vector_results, text_results, settings)


def _rrf_merge(
    vector_results: list[dict],
    text_results: list[dict],
    settings: Settings,
) -> list[dict]:
    k = settings.search.rrf_k
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        doc_id = str(doc["_id"])
        scores[doc_id] = scores.get(doc_id, 0) + settings.search.vector_weight / (k + rank + 1)
        docs[doc_id] = doc

    for rank, doc in enumerate(text_results):
        doc_id = str(doc["_id"])
        scores[doc_id] = scores.get(doc_id, 0) + settings.search.text_weight / (k + rank + 1)
        docs[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for doc_id, score in ranked[:settings.search.limit]:
        doc = docs[doc_id]
        doc["rrf_score"] = score
        result.append(doc)
    return result


def _rank_fusion_search(
    db, query: str, query_embedding: list[float],
    vector_filter: dict, settings: Settings
) -> list[dict]:
    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": [
                            {
                                "$vectorSearch": {
                                    "index": "chunks_vector",
                                    "path": "embedding",
                                    "queryVector": query_embedding,
                                    "numCandidates": settings.search.num_candidates,
                                    "limit": settings.search.limit,
                                    **({"filter": vector_filter} if vector_filter else {}),
                                }
                            }
                        ],
                        "text": [
                            {
                                "$search": {
                                    "index": "chunks_search_ro",
                                    "text": {"query": query, "path": ["text", "text_normalized", "title"]},
                                }
                            },
                            {"$limit": settings.search.limit},
                        ]
                    }
                }
            }
        },
        {"$addFields": {"rrf_score": {"$meta": "score"}}},
        {"$project": {"embedding": 0}},
        {"$limit": settings.search.limit},
    ]
    return list(db.chunks.aggregate(pipeline))
