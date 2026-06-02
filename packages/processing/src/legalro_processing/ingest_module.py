"""Standalone GazetteDocument JSON → MongoDB ingestion.

No PDF access, no OCR. Pure: JSON in, Atlas chunks out.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legalro_core.config import Settings

ALINEAT_IN_PATH = re.compile(r'alin_(\d+)')
LITERA_IN_PATH = re.compile(r'lit_([a-z])')

# ── Monthly cotizații table restructuring ────────────────────────────────────
# Party financing reports published in MO contain a 12-column monthly table
# that LlamaParse flattens into unstructured text.  We detect this pattern and
# prepend an explicit "luna X: value" block so the LLM can answer month-specific
# questions correctly.

_COTIZATII_TABLE_RE = re.compile(
    r'cuantum\w*\s+total\s+al\s+cotiza[tț]iilor',
    re.IGNORECASE | re.DOTALL,
)
_MONTHS_RO = [
    'ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie',
    'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie',
]


def _restructure_cotizatii_table(text: str) -> str:
    """Detect monthly party cotizatii table and prepend explicit month=value lines.

    LlamaParse may extract the 12-column monthly table in two formats:
      - Horizontal: "1 Centru    180 830 300 ..." (one row per line)
      - Vertical:   "1\\nCentru\\n180\\n830\\n300\\n..." (one cell per line)

    In both cases the 12 monthly values appear in January-December order.
    We extract them and prepend a structured label block so Gemini can answer
    "luna X = Y" questions correctly without needing to count columns.
    """
    if not _COTIZATII_TABLE_RE.search(text):
        return text

    # Try horizontal format first: row_num + name + 2+spaces + space-sep numbers
    horiz_re = re.compile(
        r'^\d+\s+\w[\w\s-]*?\s{2,}([\d.]+(?:\s+[\d.]+)*)',
        re.MULTILINE,
    )
    rows = horiz_re.findall(text)

    if not rows:
        # Try vertical format: row_num alone on a line, then org name, then one
        # number per line for each of 12 months.
        # Pattern: digit-only line, then a name line, then 12 number-only lines.
        vert_re = re.compile(
            r'(?:^|\n)(\d+)\n([^\n\d][^\n]*)\n'   # row_num \n org_name \n
            r'((?:[\d.]+\n){1,12})',               # 1–12 number lines
            re.MULTILINE,
        )
        vert_rows = vert_re.findall(text)
        if not vert_rows:
            return text
        # vert_rows: list of (row_num, org_name, values_block)
        rows = [vb for _, _, vb in vert_rows]  # keep just the numbers block

    total_m = re.search(r'[Cc]uantumul\s+total\s+([\d.,]+)', text)
    total = total_m.group(1) if total_m else '?'

    lines = ['STRUCTURA TABEL COTIZATII (valori lunare în lei):']
    for row_str in rows:
        nums = re.findall(r'[\d.]+', row_str)
        monthly = (nums + ['0'] * 12)[:12]
        for month, value in zip(_MONTHS_RO, monthly):
            lines.append(f'  luna {month}: {value}')
    lines.append(f'  Total anual: {total}')
    lines.append('')  # blank separator before original OCR text

    return '\n'.join(lines) + '\n' + text


def run_ingestion(json_path: str | Path, settings: "Settings") -> dict:
    """Ingest a GazetteDocument JSON file into MongoDB.

    Returns a dict with keys: gazette_id, acts_ingested, chunks_created, status.
    Skips if the gazette sha256 is already in the database.
    """
    from legalro_processing.extract.gazette_extractor import load_gazette
    from legalro_core.normalize import normalize_for_search
    from legalro_processing.prepare.chunk import chunk_act
    from legalro_core.embeddings import embed_batch
    from legalro_core.store import get_db

    gazette = load_gazette(Path(json_path))
    db = get_db(settings)

    if db.gazettes.find_one({"sha256": gazette.sha256}):
        return {"gazette_id": "", "acts_ingested": 0, "chunks_created": 0, "status": "skipped"}

    source_issue_id = f"P{gazette.part}_{gazette.issue_number}{'Bis' if gazette.is_bis else ''}_{gazette.issue_year}"

    gazette_doc = {
        "issue_number": gazette.issue_number,
        "part": gazette.part,
        "date": gazette.issue_date,
        "year": gazette.issue_year,
        "era": gazette.era,
        "filename": gazette.filename,
        "sha256": gazette.sha256,
        "page_count": gazette.pdf_page_count,
        "act_count": len(gazette.acts),
        "status": "completed",
        "sumar": [
            {
                "act_number": e.act_number,
                "doc_type": e.doc_type,
                "title": e.title,
                "page_start": e.page_start,
                "page_end": e.page_end,
                "category": e.category,
            }
            for e in gazette.sumar
        ],
        "extraction_warnings": gazette.extraction_warnings,
    }
    gazette_id = str(db.gazettes.insert_one(gazette_doc).inserted_id)

    all_chunks: list[dict] = []
    total_chunks = 0

    act_warnings: list[str] = []

    for act in gazette.acts:
        if not act.full_text or len(act.full_text) < 20:
            continue

        try:
            act_text = _restructure_cotizatii_table(act.full_text)
            chunks = chunk_act(act_text, act.doc_type, act.issuing_authority)
            if not chunks:
                continue

            text_embedded_list = [
                _build_text_embedded(chunk.text, act, source_issue_id)
                for chunk in chunks
            ]
            embeddings = embed_batch(text_embedded_list, settings)

            for position, (chunk, text_embedded, embedding) in enumerate(
                zip(chunks, text_embedded_list, embeddings)
            ):
                full_path = chunk.hierarchy_path or "unknown"
                alineat_m = ALINEAT_IN_PATH.search(full_path)
                litera_m = LITERA_IN_PATH.search(full_path)

                all_chunks.append({
                    "law_id": _make_law_id(act),
                    "source_issue_id": source_issue_id,
                    "act_index_in_issue": act.act_index,
                    "document_type": act.doc_type,
                    "issuing_authority": act.issuing_authority,
                    "act_number": act.act_number,
                    "act_year": act.act_year,
                    "locality": act.locality,
                    "title": act.title,
                    "article_number": chunk.article_number,
                    "alineat": alineat_m.group(1) if alineat_m else None,
                    "litera": litera_m.group(1) if litera_m else None,
                    "full_path": full_path,
                    "titlu": None,
                    "capitol": None,
                    "sectiune": None,
                    "chunk_type": "preamble" if full_path == "preamble" else "article",
                    "position_in_law": position,
                    "tokens": chunk.token_count,
                    "text": chunk.text,
                    "text_embedded": text_embedded,
                    "text_normalized": normalize_for_search(chunk.text),
                    "act_full_text": act.full_text,
                    "embedding": embedding,
                    "embedding_dim": len(embedding),
                    "embedding_model": settings.embeddings.model,
                    "gazette_id": gazette_id,
                })

            total_chunks += len(chunks)

        except Exception as exc:
            msg = f"act[{act.act_index}] ({act.doc_type} {act.act_number}): {exc}"
            act_warnings.append(msg)
            print(f"[ingest] WARNING skipping {msg}", flush=True)

    if all_chunks:
        db.chunks.insert_many(all_chunks)

    db.gazettes.update_one(
        {"filename": gazette.filename},
        {"$set": {"chunk_count": total_chunks, "ingest_warnings": act_warnings}},
    )

    return {
        "gazette_id": gazette_id,
        "acts_ingested": len(gazette.acts),
        "chunks_created": total_chunks,
        "status": "completed",
    }


def _make_law_id(act) -> str:
    parts = [act.doc_type]
    if act.act_number:
        parts.append(str(act.act_number))
    if act.act_year:
        parts.append(str(act.act_year))
    return "_".join(parts) if len(parts) > 1 else act.doc_type or "UNKNOWN"


def _build_text_embedded(chunk_text: str, act, source_issue_id: str) -> str:
    """Prepend structured metadata to chunk text before embedding.

    Format: [DOC_TYPE NR/AN | Autoritate | MO source_issue_id | dată] chunk_text
    Storing this alongside the raw text lets us audit what was embedded
    and re-embed from text_embedded without reconstructing the prefix.
    """
    parts = [act.doc_type or "ACT"]
    if act.act_number and act.act_year:
        parts[0] += f" {act.act_number}/{act.act_year}"
    elif act.act_number:
        parts[0] += f" {act.act_number}"

    if act.issuing_authority:
        parts.append(act.issuing_authority)
    elif "STRUCTURA TABEL COTIZATII" in chunk_text:
        # Party financing monthly tables are extracted as UNKNOWN acts with no
        # authority.  Without a party name in the prefix, BGE-M3 cannot anchor
        # the chunk to party-specific queries ("Partidul Oltenilor luna ianuarie").
        # Extract the party name from the chunk body and add it explicitly.
        party_m = re.search(r'(Partidul[ \t]+\w+(?:[ \t]+\w+)?)', chunk_text, re.IGNORECASE)
        if party_m:
            parts.append(party_m.group(1).strip())
        parts.append("cotizatii lunare 12 luni")

    if source_issue_id:
        parts.append(f"MO {source_issue_id}")
    if act.act_year:
        parts.append(str(act.act_year))

    prefix = "[" + " | ".join(parts) + "]"
    return f"{prefix} {chunk_text}"
