"""Read-only observability dashboard.

Reads (never writes) the status collections that processing + serving populate:
  * COLL_RUNS      — per-batch processing stats + failures + coverage
  * COLL_GAZETTES  — per-issue extraction warnings / coverage
  * COLL_QUERY_LOG — serving query log (who/when/query/returned ids)

Deliberately depends on base legalro-core only (no ML deps) so the image is tiny.
TODO(pilot): add HTML views (jinja2) for runs + coverage distribution; for now
JSON endpoints are enough to validate the pilot.
"""
from __future__ import annotations

import os

from fastapi import FastAPI

from legalro_core.config import load_settings
from legalro_core.store import get_db
from legalro_core import schema

settings = load_settings(os.getenv("CONFIG_PATH") or None)
app = FastAPI(title="LegalRo Dashboard", version="2.0.0")


@app.get("/health")
def health():
    try:
        get_db(settings).command("ping")
        return {"mongodb": True}
    except Exception:
        return {"mongodb": False}


@app.get("/runs")
def runs(limit: int = 20):
    db = get_db(settings)
    docs = list(db[schema.COLL_RUNS].find().sort("started_at", -1).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.get("/coverage")
def coverage():
    """Issues whose extraction flagged warnings — the pilot QA worklist."""
    db = get_db(settings)
    docs = list(db[schema.COLL_GAZETTES].find(
        {"extraction_warnings.0": {"$exists": True}},
        {"filename": 1, "extraction_warnings": 1, "era": 1},
    ).limit(200))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7861")))


if __name__ == "__main__":
    main()
