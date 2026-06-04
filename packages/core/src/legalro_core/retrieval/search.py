"""Hybrid search: vector + BM25 + RRF fusion."""
import re
from legalro_core.config import Settings
from legalro_core.embeddings import embed_texts
from legalro_core.store import get_db

_ACT_NR_RE = re.compile(r'\bnr\.?\s*([\d.]+\d)/(\d{4})\b', re.IGNORECASE)
# Captures MO issue number AND optionally the year from queries like:
#   "MO nr. 2/1989"                   → (2, 1989)
#   "Monitorul Oficial nr. 820/3.XII.2007"  → (820, 2007)
#   "Monitorul Oficial nr. 4 din 27 decembrie 1989" → (4, 1989)
# Group 1 = MO issue number, Group 2 = year (4 digits, optional).
_MO_NR_RE = re.compile(
    r'(?:\bMO\b|Monitorul\s+Oficial)\s*(?:nr\.?\s*|,\s*)(\d+)'
    r'(?:'
    r'[/_](?:\d{1,2}[\./](?:[IVXLCDM]+|[A-Za-z]+)[\./])?(\d{4})'  # N/D.M.YYYY or N/YYYY
    r'|'
    r'\s+din\s+\d{1,2}\s+\w+\s+(\d{4})'                            # N din DD MMMM YYYY
    r')?',
    re.IGNORECASE,
)
_PI_RE = re.compile(r'\bPI_(\d+)_(\d{4})\b', re.IGNORECASE)

_PIPELINE_LIMIT = 80  # candidates per pipeline before RRF merge
_METADATA_BOOST = 0.01   # base additive boost for act-number/MO matches
# Multiplier applied when the MO number AND year both match exactly — much
# stronger signal, used to disambiguate issues with the same number across years
# (e.g. PI_2_1989 vs PI_2_2007) and to overcome BM25 noise from documents that
# share date/keyword overlap with the target issue (e.g. MO_3/1989 vs MO_2/1989).
_MO_EXACT_MULTIPLIER = 15
# Title-keyword agreement boost applied when query words appear in the chunk
# title. This disambiguates cases where two acts share the same act_number in
# the same year (e.g. two ORDIN 346/2007 candidates) by favouring the one whose
# title overlaps with query content words (e.g. "taxi" vs "export").
_TITLE_KEYWORD_BOOST = 0.015

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
        year = m.group(2) or m.group(3)  # group 2: N/YYYY form, group 3: "din DD MMMM YYYY" form
        if year:
            meta["mo_year"] = int(year)
    return meta


def _compute_boost(doc: dict, meta: dict, query: str = "") -> float:
    """Compute the metadata+title boost for a single chunk doc.

    Centralised so both the $rankFusion path (_apply_metadata_boost) and the
    Python-RRF path (_rrf_merge inline loop) use identical logic.

    Act-number boost rules (year-gated):
    - Only boost when the chunk's own act_number matches the queried number AND
      (no year was parsed from the query OR the chunk's act_year matches it).
      This prevents the boost from reinforcing an act that merely happens to carry
      the same number in a different year.
    - Additionally, if the query contains title-level keywords (content words
      beyond digits/prepositions) that also appear in the chunk title, apply a
      small extra boost. This disambiguates same-number/same-year acts whose
      subject differs (e.g. ORDIN 346/2007 export-controls vs. taxi).
    """
    boost = 0.0
    act_num = meta.get("act_number")
    act_year = meta.get("act_year")
    mo_num = meta.get("mo_number")
    mo_year = meta.get("mo_year")

    if act_num and str(doc.get("act_number", "")) == act_num:
        # Year-gate: only boost when year matches or query has no year
        if not act_year or doc.get("act_year") == act_year:
            boost += _METADATA_BOOST
            # Title-keyword agreement: extract content words from the query
            # (min 4 chars, not pure digits) and check overlap with chunk title.
            if query:
                chunk_title = (doc.get("title") or "").lower()
                qwords = {
                    w.lower() for w in re.findall(r'[A-Za-zÀ-ÿ]{4,}', query)
                    if not w.isdigit()
                }
                if qwords and any(w in chunk_title for w in qwords):
                    boost += _TITLE_KEYWORD_BOOST

    if act_year and doc.get("act_year") == act_year:
        boost += _METADATA_BOOST * 0.5

    if mo_num:
        issue = doc.get("source_issue_id", "")
        if mo_year:
            # Exact issue-ID match supporting both v1 (PI_N_Y) and v2 (MO_PI_N_Y)
            # formats. Strong boost — decisively favours the right issue when the
            # query explicitly states both MO number and year.
            exact_ids = (
                f"MO_PI_{mo_num}_{mo_year}", f"MO_PI_{mo_num}Bis_{mo_year}",
                f"PI_{mo_num}_{mo_year}", f"PI_{mo_num}Bis_{mo_year}",
            )
            if issue in exact_ids:
                boost += _METADATA_BOOST * _MO_EXACT_MULTIPLIER
            elif mo_num in issue and str(mo_year) in issue:
                # Loose match: issue contains both number and year
                boost += _METADATA_BOOST * 2
        elif mo_num in issue:
            # Fallback: no year in query — use weak substring boost
            boost += _METADATA_BOOST

    return boost


def _apply_metadata_boost(docs: list[dict], meta: dict, query: str = "") -> list[dict]:
    """Add metadata+title boost to each doc and re-sort. Used by the $rankFusion path."""
    if not meta:
        return docs
    for doc in docs:
        b = _compute_boost(doc, meta, query)
        if b:
            doc["rrf_score"] = doc.get("rrf_score", 0.0) + b
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

    # Only act_type in vector_filter — act_year is not in the Atlas filterable
    # index, so year filtering is applied as a Python post-filter after merge.
    vector_filter = {}
    if act_type:
        vector_filter["document_type"] = act_type.upper()

    meta = _parse_query_metadata(query)

    if settings.search.use_rank_fusion:
        results = _rank_fusion_search(db, query, query_embedding, vector_filter, settings)
        return _apply_metadata_boost(results, meta, query)
    else:
        # Pass meta into _python_rrf_search so the boost is applied BEFORE
        # the top-N cutoff — chunks ranked just outside limit can be rescued.
        return _python_rrf_search(db, query, query_embedding, vector_filter, settings,
                                  year_from=year_from, year_to=year_to, meta=meta)


def _python_rrf_search(
    db, query: str, query_embedding: list[float],
    vector_filter: dict, settings: Settings,
    year_from: int | None = None,
    year_to: int | None = None,
    meta: dict | None = None,
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

    return _rrf_merge(vector_results, text_results, settings, meta=meta, query=query)


def _rrf_merge(
    vector_results: list[dict],
    text_results: list[dict],
    settings: Settings,
    meta: dict | None = None,
    query: str = "",
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

    # Apply metadata+title boost to ALL candidates BEFORE the top-N cutoff so
    # that a strong boost (e.g. exact MO+year match) can rescue chunks that would
    # otherwise be eliminated at the search.limit boundary.
    # Uses the shared _compute_boost helper (same logic as the $rankFusion path).
    if meta:
        for doc_id, doc in docs.items():
            b = _compute_boost(doc, meta, query)
            if b:
                scores[doc_id] = scores.get(doc_id, 0.0) + b

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
