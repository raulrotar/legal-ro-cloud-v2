"""Run ledger: one record per processing batch, written to COLL_RUNS.

This is the data source the dashboard reads to show "how did the last run go".
It is also your pilot accuracy/QA surface — coverage_ratio per PDF and the
failure list are exactly what you stare at while validating 500-1000 PDFs.

TODO(pilot): call begin()/record_doc()/finish() from the processing CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from legalro_core import schema


@dataclass
class RunRecord:
    run_id: str
    started_at: str
    pipeline_version: str = schema.PIPELINE_VERSION
    schema_version: str = schema.SCHEMA_VERSION
    embedding_version: str = schema.EMBEDDING_VERSION
    ended_at: str = ""
    stats: dict = field(default_factory=lambda: {
        "pdfs_seen": 0, "pdfs_processed": 0, "pdfs_skipped": 0, "pdfs_failed": 0,
        "chunks_written": 0, "coverage_min": 1.0, "coverage_mean": 1.0,
    })
    failures: list[dict] = field(default_factory=list)


def new_run() -> RunRecord:
    ts = datetime.now(timezone.utc)
    return RunRecord(run_id=f"run-{ts.strftime('%Y-%m-%dT%H-%M-%SZ')}", started_at=ts.isoformat())


def persist(db, run: RunRecord) -> None:
    run.ended_at = datetime.now(timezone.utc).isoformat()
    db[schema.COLL_RUNS].replace_one({"_id": run.run_id}, {"_id": run.run_id, **asdict(run)}, upsert=True)
