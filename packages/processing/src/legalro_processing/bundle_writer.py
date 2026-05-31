"""Stage A sink: write a fully-processed issue (chunks WITH embeddings inline)
to an on-disk bundle. See legalro_core.bundle for the format.

This REPLACES the inline "write straight to Mongo" behaviour of the old
ingest_module. Stage A produces the durable artifact; Stage B (loader.py) pushes
it to Mongo separately and idempotently.

TODO(pilot): wire this to the extract+chunk+embed output. The skeleton below
shows the intended shape; fill in once the chunk/embed step emits dicts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from legalro_core import bundle, schema


def emit_doc(
    out_root: Path,
    doc_id: str,
    sha256: str,
    coverage_ratio: float,
    gazette_doc: dict,
    chunks: list[dict],
    edges: list[dict] | None = None,
    gzip_chunks: bool = True,
) -> bundle.DocMeta:
    """Write one issue's artifacts under out_root/by_doc/<doc_id>/ and append to manifest."""
    edges = edges or []
    d = bundle.doc_dir(out_root, doc_id)
    d.mkdir(parents=True, exist_ok=True)

    chunks_name = "chunks.jsonl.gz" if gzip_chunks else "chunks.jsonl"
    files = {
        "gazette.json": bundle.write_json(d / "gazette.json", gazette_doc),
        chunks_name: bundle.write_jsonl(d / chunks_name, chunks, gz=gzip_chunks),
        "edges.jsonl": bundle.write_jsonl(d / "edges.jsonl", edges),
    }
    checksums = {name: bundle.sha256_file(d / name) for name in files}

    meta = bundle.DocMeta(
        doc_id=doc_id,
        sha256=sha256,
        schema_version=schema.SCHEMA_VERSION,
        pipeline_version=schema.PIPELINE_VERSION,
        embedding_version=schema.EMBEDDING_VERSION,
        coverage_ratio=coverage_ratio,
        files=files,
        checksums=checksums,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )
    bundle.write_json(d / "_meta.json", {**meta.__dict__})
    bundle.append_manifest(out_root, meta)
    return meta
