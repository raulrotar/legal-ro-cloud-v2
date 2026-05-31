"""On-disk bundle format — the contract between Stage A (extract+embed) and
Stage B (load to Mongo). See spec §4.0.

A bundle is a directory tree that is the durable, reviewable, content-addressable
output of processing. It can be produced on a VPS, rsync'd / shipped to S3,
inspected offline, and loaded into ANY MongoDB (Atlas Local for the pilot, paid
Atlas later) with an idempotent upsert.

    out/
    ├── manifest.jsonl                 # one line per processed PDF (loader's index)
    └── by_doc/
        └── <doc_id>/
            ├── _meta.json             # sha256, versions, file sizes + checksums
            ├── gazette.json           # 1 doc  -> COLL_GAZETTES
            ├── chunks.jsonl[.gz]      # N docs -> COLL_CHUNKS (embeddings inline)
            └── edges.jsonl            # N docs -> COLL_EDGES

This module owns ONLY the read/write plumbing + manifest schema. What goes INTO
a chunk/gazette/edge is defined by legalro_processing; this stays dependency-light
so the loader and the dashboard can read bundles without pulling ML deps.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator

MANIFEST = "manifest.jsonl"
BY_DOC = "by_doc"


@dataclass
class DocMeta:
    """_meta.json — provenance + integrity for one processed PDF."""
    doc_id: str
    sha256: str                      # content hash of the source PDF
    schema_version: str
    pipeline_version: str
    embedding_version: str
    coverage_ratio: float            # 1.0 == lossless (spec §3.13)
    files: dict[str, int] = field(default_factory=dict)        # name -> byte size
    checksums: dict[str, str] = field(default_factory=dict)    # name -> sha256
    extracted_at: str = ""


# ── writers ───────────────────────────────────────────────────────────────────
def write_json(path: Path, obj) -> int:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.stat().st_size


def write_jsonl(path: Path, items: Iterable[dict], gz: bool = False) -> int:
    opener = (lambda p: gzip.open(p, "wt", encoding="utf-8")) if gz else \
             (lambda p: open(p, "w", encoding="utf-8"))
    with opener(path) as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path.stat().st_size


def append_manifest(out_root: Path, meta: DocMeta) -> None:
    with (out_root / MANIFEST).open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── readers (used by the loader + dashboard) ──────────────────────────────────
def read_manifest(out_root: Path) -> Iterator[DocMeta]:
    mpath = out_root / MANIFEST
    if not mpath.exists():
        return
    with mpath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield DocMeta(**json.loads(line))


def read_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def doc_dir(out_root: Path, doc_id: str) -> Path:
    return out_root / BY_DOC / doc_id


def verify_checksums(d: Path, checksums: dict[str, str]) -> None:
    """Fail loud on transit corruption before loading anything."""
    for name, expected in checksums.items():
        actual = sha256_file(d / name)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {d / name}: {actual} != {expected}")
