"""FastAPI application — READ-ONLY query service (serving deployable).

This service is intentionally decoupled from `legalro_processing`: it never
imports Docling/OCR/embedding-batch code and only needs read access to MongoDB.
Ingestion (PDF -> JSON -> chunks -> Mongo) lives entirely in the `processing`
package and runs out-of-band (VPS / batch). The DB is the only contract between
the two. See MIGRATION.md.
"""
import os
import time

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from legalro_core.config import load_settings
from legalro_core.store import get_db
from legalro_core.retrieval.search import hybrid_search
from legalro_serving.generation import run_query_hybrid

settings = load_settings(os.getenv("CONFIG_PATH") or None)

app = FastAPI(title="LegalRo API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────
_API_TOKEN = os.environ.get("API_TOKEN", "")


def _require_auth(x_api_token: str | None = Header(default=None)):
    if not _API_TOKEN:
        return
    if x_api_token != _API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    act_type: str = ""
    year_from: int | None = None
    year_to: int | None = None


class SourceItem(BaseModel):
    document_type: str | None = None
    act_number: str | None = None
    act_year: int | None = None
    title: str | None = None
    issuing_authority: str | None = None
    source_issue_id: str | None = None
    rrf_score: float | None = None
    excerpt: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    latency_ms: int = 0
    chunks_used: int = 0


class ActItem(BaseModel):
    id: str | None = None
    document_type: str | None = None
    act_number: str | None = None
    act_year: int | None = None
    title: str | None = None
    issuing_authority: str | None = None
    source_issue_id: str | None = None
    gazette_date: str | None = None
    status: str | None = None
    signed_by: str | None = None


class ActListResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: list[ActItem]


class ActDetailResponse(ActItem):
    full_authority: str | None = None
    summary: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_to_source(chunk: dict) -> SourceItem:
    text = chunk.get("text", "")
    return SourceItem(
        document_type=chunk.get("document_type"),
        act_number=str(chunk.get("act_number", "")) or None,
        act_year=chunk.get("act_year"),
        title=chunk.get("title"),
        issuing_authority=chunk.get("issuing_authority"),
        source_issue_id=chunk.get("source_issue_id"),
        rrf_score=chunk.get("rrf_score"),
        excerpt=text[:300] if text else None,
    )


def _gazette_date(db, source_issue_id: str | None) -> str | None:
    if not source_issue_id:
        return None
    try:
        g = db.gazettes.find_one({"_id": source_issue_id}, {"date": 1})
        if g and g.get("date"):
            d = g["date"]
            return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
    except Exception:
        pass
    return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    status: dict = {"mongodb": False}
    try:
        get_db(settings).command("ping")
        status["mongodb"] = True
    except Exception:
        pass
    return status


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_require_auth)])
def query(req: QueryRequest):
    t0 = time.time()
    act_type = req.act_type or None

    # Run hybrid search first to collect sources (same params as generation fallback)
    try:
        chunks = hybrid_search(
            req.question, settings,
            act_type=act_type,
            year_from=req.year_from,
            year_to=req.year_to,
        )
    except Exception:
        chunks = []

    try:
        answer = run_query_hybrid(req.question, settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.time() - t0) * 1000)
    sources = [_chunk_to_source(c) for c in chunks]
    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=latency_ms,
        chunks_used=len(chunks),
    )


@app.get("/acts", response_model=ActListResponse, dependencies=[Depends(_require_auth)])
def list_acts(
    type: str | None = Query(default=None, description="Comma-separated document types"),
    year_from: int | None = Query(default=None),
    year_to: int | None = Query(default=None),
    q: str | None = Query(default=None, description="Substring search on title, act_number, issuing_authority"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    sort: str = Query(default="date_desc", pattern="^(date_desc|date_asc|type)$"),
):
    db = get_db(settings)

    match: dict = {"law_id": {"$exists": True, "$nin": [None, ""]}}

    if type:
        types = [t.strip().upper() for t in type.split(",") if t.strip()]
        if types:
            match["document_type"] = {"$in": types}
    if year_from is not None or year_to is not None:
        year_range: dict = {}
        if year_from is not None:
            year_range["$gte"] = year_from
        if year_to is not None:
            year_range["$lte"] = year_to
        match["act_year"] = year_range
    if q:
        match["$or"] = [
            {"title": {"$regex": q, "$options": "i"}},
            {"act_number": {"$regex": q, "$options": "i"}},
            {"issuing_authority": {"$regex": q, "$options": "i"}},
        ]

    sort_spec: dict
    if sort == "date_asc":
        sort_spec = {"act_year": 1, "_id": 1}
    elif sort == "type":
        sort_spec = {"document_type": 1, "act_year": -1}
    else:  # date_desc
        sort_spec = {"act_year": -1, "_id": -1}

    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": "$law_id",
            "document_type": {"$first": "$document_type"},
            "act_number": {"$first": "$act_number"},
            "act_year": {"$first": "$act_year"},
            "title": {"$first": "$title"},
            "issuing_authority": {"$first": "$issuing_authority"},
            "source_issue_id": {"$first": "$source_issue_id"},
            "signed_by": {"$first": "$signed_by"},
        }},
        {"$facet": {
            "total": [{"$count": "n"}],
            "items": [
                {"$sort": sort_spec},
                {"$skip": (page - 1) * limit},
                {"$limit": limit},
            ],
        }},
    ]

    try:
        result = list(db.chunks.aggregate(pipeline))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total = result[0]["total"][0]["n"] if result and result[0]["total"] else 0
    raw_items = result[0]["items"] if result else []

    items = [
        ActItem(
            id=doc.get("_id"),
            document_type=doc.get("document_type"),
            act_number=str(doc.get("act_number", "")) or None,
            act_year=doc.get("act_year"),
            title=doc.get("title"),
            issuing_authority=doc.get("issuing_authority"),
            source_issue_id=doc.get("source_issue_id"),
            gazette_date=_gazette_date(db, doc.get("source_issue_id")),
            signed_by=doc.get("signed_by"),
        )
        for doc in raw_items
    ]

    return ActListResponse(total=total, page=page, limit=limit, items=items)


@app.get("/acts/{act_id}", response_model=ActDetailResponse, dependencies=[Depends(_require_auth)])
def get_act(act_id: str):
    db = get_db(settings)

    chunk = db.chunks.find_one(
        {"law_id": act_id},
        {"embedding": 0},
        sort=[("chunk_index", 1)],
    )
    if not chunk:
        raise HTTPException(status_code=404, detail="Act not found")

    act_full_text = chunk.get("act_full_text", "") or ""
    summary = act_full_text[:300].strip() if act_full_text else None

    return ActDetailResponse(
        id=chunk.get("law_id"),
        document_type=chunk.get("document_type"),
        act_number=str(chunk.get("act_number", "")) or None,
        act_year=chunk.get("act_year"),
        title=chunk.get("title"),
        issuing_authority=chunk.get("issuing_authority"),
        source_issue_id=chunk.get("source_issue_id"),
        gazette_date=_gazette_date(db, chunk.get("source_issue_id")),
        signed_by=chunk.get("signed_by"),
        summary=summary,
    )
