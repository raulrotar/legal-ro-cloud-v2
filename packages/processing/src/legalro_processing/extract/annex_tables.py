"""Ruled annex-table extraction from the PDF, bypassing TableFormer output.

Large ruled annexes (MO 294Bis/2026: 146 pages of university nomenclature)
are the worst case for TableFormer-in-Markdown: output inflates ~5× and
multi-page tables fragment per page with repeated headers.  Born-digital
ruled tables are a solved deterministic problem: PyMuPDF's find_tables()
(line strategy) reads the ruling lines directly — no ML, no new deps.

This module extracts per-page tables and STITCHES multi-page continuations:
a table whose header row repeats the previous page's header (or whose column
count matches a continuing table that has no header) is appended to it.
Output is the pipeline's existing Table dataclass (pipe-table Markdown), so
chunking/ingestion is unchanged.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _row_key(row: list) -> str:
    return re.sub(r"\s+", "", "|".join(str(c or "") for c in row)).lower()


def _to_markdown(header: list, rows: list[list]) -> str:
    def fmt(cells: list) -> str:
        return "| " + " | ".join(
            re.sub(r"\s+", " ", str(c or "")).strip() for c in cells
        ) + " |"

    lines = [fmt(header), "|" + "---|" * len(header)]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


# Heading-like line worth using as a table title (annex labels, nomenclature
# captions) — NOT signatures ("PRIM-MINISTRU ILIE-GAVRIL BOLOJAN"), page
# headers, or bare numbers.
_TITLE_GOOD = re.compile(r"ANEX[AĂE]|Nomenclator|Lista|Situa[țt]ia|Structura", re.IGNORECASE)
_TITLE_BAD = re.compile(
    r"^\d+$|MONITORUL\s+OFICIAL|PRIM-?MINISTRU|MINISTRU|PRE[ȘS]EDINTELE|^[A-ZĂÂÎȘȚ\s.\-]+$"
)


def _table_title(page, bbox) -> str:
    """Pick the best heading line above the table; prefer annex/caption lines,
    reject signatures, running headers, and bare page numbers."""
    import fitz

    above = page.get_text(
        "text", clip=fitz.Rect(0, max(0, bbox[1] - 120), page.rect.width, bbox[1])
    )
    lines = [l.strip() for l in above.splitlines() if l.strip()]
    for line in reversed(lines):
        if _TITLE_GOOD.search(line):
            return line[:120]
    for line in reversed(lines):
        if not _TITLE_BAD.match(line):
            return line[:120]
    return ""


def extract_annex_tables(pdf_path: str | Path, min_rows: int = 4) -> list:
    """Extract ruled tables from the PDF text layer, stitching multi-page runs.

    Returns a list of gazette_schema.Table.  Tables with fewer than min_rows
    data rows are ignored (boxed notes, mastheads).
    """
    import fitz

    from legalro_processing.extract.gazette_schema import Table

    doc = fitz.open(str(pdf_path))
    out: list[Table] = []
    # running stitch state
    cur_header: list | None = None
    cur_rows: list[list] = []
    cur_page0 = 0
    cur_title = ""

    def _flush() -> None:
        nonlocal cur_header, cur_rows
        if cur_header is not None and len(cur_rows) >= min_rows:
            out.append(Table(
                markdown=_to_markdown(cur_header, cur_rows),
                page=cur_page0,
                title=cur_title,
                n_rows=len(cur_rows),
            ))
        cur_header, cur_rows = None, []

    for pno, page in enumerate(doc):
        try:
            tabs = page.find_tables(strategy="lines_strict").tables
        except Exception:
            tabs = []
        if not tabs:
            _flush()
            continue
        for t in tabs:
            data = t.extract()
            if not data:
                continue
            header, rows = data[0], data[1:]
            if cur_header is not None and len(header) == len(cur_header):
                # continuation: repeated header → drop it; headerless → all rows
                if _row_key(header) == _row_key(cur_header):
                    cur_rows.extend(rows)
                    continue
                if not any(str(c or "").strip() for c in header) or len(rows) == 0:
                    cur_rows.extend([header] + rows)
                    continue
                # same width but a genuinely new header → new table
            _flush()
            cur_header, cur_rows = header, rows
            cur_page0 = pno
            cur_title = _table_title(page, t.bbox)
    _flush()
    doc.close()
    if out:
        print(f"[annex-tables] {Path(pdf_path).stem}: {len(out)} stitched table(s), "
              f"{sum(t.n_rows for t in out)} rows", file=sys.stderr, flush=True)
    return out
