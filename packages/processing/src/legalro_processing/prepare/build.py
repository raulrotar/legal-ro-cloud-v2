"""Turn an extracted GazetteDocument into bundle-ready documents.

This is the front half of the old ingest_module, MINUS the direct-to-Mongo write:
it produces the dicts (gazette + chunks) that bundle_writer serialises to disk.
Stage B (loader.py) is what actually pushes them to Mongo, separately.

Key differences from v1 ingest_module:
  * deterministic _id / chunk_id / act_id from legalro_core.schema (idempotent loads),
  * modality + embedding_version + publication_date facets stamped on every chunk,
  * `act_full_text` is NOT duplicated onto every chunk (M0 storage bloat fix),
  * real coverage_ratio computed against the raw PyMuPDF stream.
"""
from __future__ import annotations

import re
from pathlib import Path

import fitz

from legalro_core import schema
from legalro_core.normalize import normalize_for_search
from legalro_core.embeddings import embed_batch
from legalro_processing.prepare.chunk import chunk_act
from legalro_processing.audit.coverage import coverage_ratio

_ALINEAT = re.compile(r"alin_(\d+)")
_LITERA = re.compile(r"lit_([a-z])")


def _text_embedded(chunk_text: str, act, issue_id: str) -> str:
    """Prefix structured metadata before embedding (kept identical to v1 format)."""
    head = act.doc_type or "ACT"
    if act.act_number and act.act_year:
        head += f" {act.act_number}/{act.act_year}"
    elif act.act_number:
        head += f" {act.act_number}"
    parts = [head]
    if act.issuing_authority:
        parts.append(act.issuing_authority)
    if issue_id:
        parts.append(f"MO {issue_id}")
    if act.act_year:
        parts.append(str(act.act_year))
    return "[" + " | ".join(parts) + "] " + chunk_text


def _law_id(act) -> str:
    parts = [act.doc_type]
    if act.act_number:
        parts.append(str(act.act_number))
    if act.act_year:
        parts.append(str(act.act_year))
    return "_".join(parts) if len(parts) > 1 else (act.doc_type or "UNKNOWN")


def _raw_text(pdf_path: str | Path) -> str:
    try:
        with fitz.open(str(pdf_path)) as doc:
            return "".join(p.get_text("text") for p in doc)
    except Exception:
        return ""


def build_issue_docs(gazette, pdf_path: str | Path, settings, embed: bool = True):
    """Return (issue_id, sha256, gazette_doc, chunks, coverage_ratio)."""
    issue_id = schema.doc_id(gazette.issue_number, gazette.issue_year, gazette.is_bis, gazette.part)
    modality = schema.modality_for_era(gazette.era)

    gazette_doc = {
        "_id": issue_id,
        "issue_number": gazette.issue_number,
        "part": gazette.part,
        "date": gazette.issue_date,
        "year": gazette.issue_year,
        "era": gazette.era,
        "modality": modality,
        "is_bis": gazette.is_bis,
        "filename": gazette.filename,
        "sha256": gazette.sha256,
        "page_count": gazette.pdf_page_count,
        "act_count": len(gazette.acts),
        "schema_version": schema.SCHEMA_VERSION,
        "pipeline_version": schema.PIPELINE_VERSION,
        "sumar": [
            {
                "act_number": e.act_number, "doc_type": e.doc_type, "title": e.title,
                "page_start": e.page_start, "page_end": e.page_end, "category": e.category,
            }
            for e in gazette.sumar
        ],
        "extraction_warnings": gazette.extraction_warnings,
    }

    chunks: list[dict] = []
    mapped_texts: list[str] = [gazette.sumar_raw]

    for act in gazette.acts:
        if not act.full_text or len(act.full_text) < 20:
            continue
        mapped_texts.append(act.full_text)
        aid = schema.act_id(issue_id, act.act_index)
        text_chunks = chunk_act(act.full_text, act.doc_type, act.issuing_authority)
        if not text_chunks:
            continue

        embedded_texts = [_text_embedded(c.text, act, issue_id) for c in text_chunks]
        vectors = embed_batch(embedded_texts, settings) if embed else [[] for _ in text_chunks]

        for i, (c, et, vec) in enumerate(zip(text_chunks, embedded_texts, vectors)):
            cid = schema.chunk_id(aid, i)
            path = c.hierarchy_path or "unknown"
            am, lm = _ALINEAT.search(path), _LITERA.search(path)
            chunks.append({
                "_id": cid, "chunk_id": cid, "act_id": aid,
                "issue_id": issue_id, "source_issue_id": issue_id,
                "act_index_in_issue": act.act_index,
                # ── facets (vector-index filters) ──
                "document_type": act.doc_type,
                "issuing_authority": act.issuing_authority,
                "act_number": act.act_number,
                "act_year": act.act_year,
                "locality": act.locality,
                "title": act.title,
                "modality": modality,
                "publication_date": gazette.issue_date,
                "law_id": _law_id(act),
                # ── structural path ──
                "article_number": c.article_number,
                "alineat": am.group(1) if am else None,
                "litera": lm.group(1) if lm else None,
                "full_path": path,
                "chunk_type": "preamble" if path == "preamble" else "article",
                "position_in_law": i,
                "tokens": c.token_count,
                # ── payloads ──
                "text": c.text,
                "text_embedded": et,
                "text_normalized": normalize_for_search(c.text),
                # act_full_text capped at 12000 chars — enables parent-doc retrieval
                # in context.py without re-exploding chunk sizes on M0.
                "act_full_text": act.full_text[:12000],
                "embedding": vec,
                "embedding_dim": len(vec),
                "embedding_model": settings.embeddings.model,
                "embedding_version": schema.EMBEDDING_VERSION,
            })

    raw = _raw_text(pdf_path)
    coverage = min(coverage_ratio(raw, mapped_texts, []), 1.0) if raw else 1.0
    return issue_id, gazette.sha256, gazette_doc, chunks, coverage
