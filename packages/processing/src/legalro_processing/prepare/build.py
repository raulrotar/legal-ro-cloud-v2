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


def _text_embedded(chunk_text: str, act, issue_id: str, summary: str = "") -> str:
    """Prefix structured metadata (+ SAC act summary) before embedding."""
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
    prefix = "[" + " | ".join(parts) + "]"
    if summary:
        # SAC: a generic whole-act summary on every chunk halves wrong-act
        # retrieval on structurally similar legal corpora (arXiv 2510.06999)
        prefix += f" [{summary}]"
    return prefix + " " + chunk_text


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

    # ── Phase-0 pre-embedding cleanup (docs/EMBEDDINGS_PLAN.md) ───────────────
    from legalro_processing.prepare.clean import (
        clean_act_text, is_embedding_poison, near_duplicate_act_indices,
    )
    _dup_idx = near_duplicate_act_indices(gazette.acts)

    for _ai, act in enumerate(gazette.acts):
        if not act.full_text or len(act.full_text) < 20:
            continue
        if _ai in _dup_idx:
            print(f"[build] {issue_id} act[{act.act_index}] skipped: near-duplicate body", flush=True)
            continue
        _clean_text = clean_act_text(act.full_text)
        if len(_clean_text) < 20:
            continue
        if is_embedding_poison(_clean_text):
            print(f"[build] {issue_id} act[{act.act_index}] QUARANTINED: repetition-loop body", flush=True)
            continue
        act.full_text = _clean_text
        mapped_texts.append(act.full_text)
        aid = schema.act_id(issue_id, act.act_index)
        text_chunks = chunk_act(act.full_text, act.doc_type, act.issuing_authority)
        if not text_chunks:
            continue

        from legalro_processing.prepare.sac import act_summary
        _summary = act_summary(act, settings) if embed else ""
        embedded_texts = [_text_embedded(c.text, act, issue_id, _summary) for c in text_chunks]
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
                # summary appended for the contextual-BM25 effect (lexical
                # act-level terms searchable from every chunk)
                "text_normalized": normalize_for_search(
                    c.text + (" " + _summary if _summary else "")
                ),
                # act_full_text capped at 12000 chars — enables parent-doc retrieval
                # in context.py without re-exploding chunk sizes on M0.
                "act_full_text": act.full_text[:12000],
                "embedding": vec,
                "embedding_dim": len(vec),
                "embedding_model": settings.embeddings.model,
                "embedding_version": schema.EMBEDDING_VERSION,
            })

    # ── Table chunks (financial_table) ────────────────────────────────────────
    # Tables extracted by the layout triage are stored in the same db.chunks
    # collection so hybrid_search retrieves them automatically.  Each table is
    # one chunk (or split on row boundaries if oversized).
    for tbl_idx, tbl in enumerate(getattr(gazette, "tables", [])):
        if not tbl.markdown or len(tbl.markdown) < 10:
            continue
        tbl_chunks = _split_table(tbl.markdown)
        tbl_title = tbl.title or f"Tabel {tbl_idx + 1}"
        for part_idx, tbl_text in enumerate(tbl_chunks):
            et = f"[TABLE | {tbl_title} | MO {issue_id} | {gazette.issue_year}] {tbl_text}"
            vec = embed_batch([et], settings)[0] if embed else []
            tid = f"{issue_id}_tbl{tbl_idx}_p{part_idx}"
            chunks.append({
                "_id": tid, "chunk_id": tid, "act_id": None,
                "issue_id": issue_id, "source_issue_id": issue_id,
                "act_index_in_issue": None,
                # facets
                "document_type": "TABLE",
                "issuing_authority": None,
                "act_number": None,
                "act_year": gazette.issue_year,
                "locality": None,
                "title": tbl_title,
                "modality": modality,
                "publication_date": gazette.issue_date,
                "law_id": None,
                # structural
                "article_number": None,
                "alineat": None,
                "litera": None,
                "full_path": f"table/{tbl_idx}/part/{part_idx}",
                "chunk_type": "financial_table",
                "position_in_law": part_idx,
                "tokens": len(tbl_text.split()),
                # payloads
                "text": tbl_text,
                "text_embedded": et,
                "text_normalized": normalize_for_search(tbl_text),
                "act_full_text": tbl.markdown[:12000],
                "embedding": vec,
                "embedding_dim": len(vec),
                "embedding_model": settings.embeddings.model,
                "embedding_version": schema.EMBEDDING_VERSION,
                # table metadata
                "table_page": tbl.page,
                "table_n_rows": tbl.n_rows,
            })

    raw = _raw_text(pdf_path)
    coverage = min(coverage_ratio(raw, mapped_texts, []), 1.0) if raw else 1.0
    return issue_id, gazette.sha256, gazette_doc, chunks, coverage


def _split_table(table_md: str, max_tokens: int = 900) -> list[str]:
    """Split a large pipe table on row boundaries, repeating the header row."""
    lines = table_md.splitlines()
    if not lines:
        return [table_md]

    # Identify header (first row) and separator (second row, dashes).
    header = lines[0] if lines else ""
    sep = lines[1] if len(lines) > 1 and re.match(r'^\|[\s\-:|]+\|', lines[1]) else ""
    data_lines = lines[2:] if sep else lines[1:]

    parts: list[str] = []
    current: list[str] = []
    token_count = 0

    for row in data_lines:
        row_tokens = len(row.split())
        if token_count + row_tokens > max_tokens and current:
            chunk = "\n".join(filter(None, [header, sep] + current))
            parts.append(chunk)
            current = [row]
            token_count = row_tokens
        else:
            current.append(row)
            token_count += row_tokens

    if current:
        parts.append("\n".join(filter(None, [header, sep] + current)))

    return parts if parts else [table_md]
