"""Stage B: load an on-disk bundle into MongoDB. Idempotent + resumable.

This is the ONLY thing that needs the Atlas/Atlas-Local connection. It is
network-bound, GPU-free, and re-runnable: deterministic _ids mean re-loading an
already-loaded doc is a no-op, and a .load_state.json lets a partial run resume.

Works identically against:
  * MongoDB Atlas Local (Docker)  -> pilot validation, no cost
  * MongoDB Atlas (paid)          -> production, same bundle, unchanged

CLI: legalro-process load --root out/ --mongo "$MONGO_URI"
"""
from __future__ import annotations

import json
from pathlib import Path

from pymongo import MongoClient, UpdateOne

from legalro_core import bundle, schema


def load_bundle(out_root: Path, mongo_uri: str, db_name: str = "legalro",
                resume: bool = True, batch: int = 1000) -> dict:
    cli = MongoClient(mongo_uri)
    db = cli[db_name]

    state_path = out_root / ".load_state.json"
    state = json.loads(state_path.read_text()) if (resume and state_path.exists()) else {}

    loaded, skipped = 0, 0
    for meta in bundle.read_manifest(out_root):
        key = f"{meta.doc_id}::{meta.sha256}::{meta.pipeline_version}::{meta.embedding_version}"
        if state.get(key) == "done":
            skipped += 1
            continue

        d = bundle.doc_dir(out_root, meta.doc_id)
        bundle.verify_checksums(d, meta.checksums)

        gazette = json.loads((d / "gazette.json").read_text(encoding="utf-8"))
        db[schema.COLL_GAZETTES].replace_one({"_id": meta.doc_id}, gazette, upsert=True)

        chunks_file = next((d / n for n in ("chunks.jsonl.gz", "chunks.jsonl") if (d / n).exists()), None)
        if chunks_file:
            _bulk_upsert(db[schema.COLL_CHUNKS], bundle.read_jsonl(chunks_file), batch)
        edges_file = d / "edges.jsonl"
        if edges_file.exists():
            _bulk_upsert(db[schema.COLL_EDGES], bundle.read_jsonl(edges_file), batch)

        state[key] = "done"
        state_path.write_text(json.dumps(state, indent=2))
        loaded += 1

    return {"loaded": loaded, "skipped": skipped}


def _bulk_upsert(coll, docs, batch: int) -> int:
    ops, n = [], 0
    for doc in docs:
        _id = doc.get("_id") or doc.get("chunk_id") or doc.get("edge_id")
        if _id is None:
            raise ValueError(f"bundle doc missing a deterministic _id: {list(doc)[:5]}")
        doc["_id"] = _id
        ops.append(UpdateOne({"_id": _id}, {"$set": doc}, upsert=True))
        if len(ops) >= batch:
            coll.bulk_write(ops, ordered=False)
            n += len(ops)
            ops = []
    if ops:
        coll.bulk_write(ops, ordered=False)
        n += len(ops)
    return n
