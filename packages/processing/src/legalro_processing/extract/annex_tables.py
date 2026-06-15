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


def _clean_cell(c) -> str:
    """Tag-free, whitespace-collapsed view of one cell's text."""
    return re.sub(r"\s+", " ", str(c or "")).strip()


def _to_markdown(header: list, rows: list[list]) -> str:
    def fmt(cells: list) -> str:
        return "| " + " | ".join(_clean_cell(c) for c in cells) + " |"

    lines = [fmt(header), "|" + "---|" * len(header)]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


def _esc(text: str) -> str:
    """HTML-escape the three structural chars (cell text only, no tags)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _to_html(header: list, rows: list[list]) -> str:
    """Single-line flat grid <table>.  Cells are whitespace-collapsed and
    HTML-escaped.  No colspan/rowspan yet — true spans are Phase 1b/2; this
    flat grid still carries every cell in source order (QA §3.2/§3.3 xfail).
    """
    parts = ["<table>"]
    parts.append("<tr>" + "".join(
        f"<th>{_esc(_clean_cell(c))}</th>" for c in header) + "</tr>")
    for r in rows:
        parts.append("<tr>" + "".join(
            f"<td>{_esc(_clean_cell(c))}</td>" for c in r) + "</tr>")
    parts.append("</table>")
    return "".join(parts)


def _to_flat_text(header: list, rows: list[list]) -> str:
    """Tag-free, source-order view used as the search / embedding payload.
    Cells are tab-joined within a row, rows newline-joined."""
    def fmt(cells: list) -> str:
        return "\t".join(_clean_cell(c) for c in cells)

    lines = [fmt(header)] + [fmt(r) for r in rows]
    return "\n".join(lines)


def _cell_text(page, bbox) -> str:
    """Rebuild a cell's text in human reading order from dict lines.

    PyMuPDF's Table.extract() returns words in raw stream order, which scrambles
    ROTATED cells (the 294Bis Nomenclator has 90°-rotated header/specialization
    columns): "Ingineria substanțelor și protecția anorganice mediului" instead
    of "Ingineria substanțelor anorganice și protecția mediului".  Re-deriving
    from get_text('dict') lines — ordered by writing direction — recovers source
    order and kills the §1 text-bleed bigrams.
    """
    import fitz

    d = page.get_text("dict", clip=fitz.Rect(bbox))
    lines = []
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            txt = "".join(s.get("text", "") for s in l.get("spans", []))
            if not txt.strip():
                continue
            x0, y0, _x1, _y1 = l["bbox"]
            dirx, diry = l.get("dir", (1.0, 0.0))
            lines.append((dirx, diry, x0, y0, txt.strip()))
    if not lines:
        return ""
    # Vertical (rotated) text: order columns left→right (by x), then by y.
    # Horizontal text: normal top→bottom (by y), then left→right (by x).
    # NOTE: orientation is inferred from lines[0] only — fine for the current
    # corpus (a cell is uniformly horizontal or 90°-rotated), but fragile if a
    # single cell ever mixes orientations.
    rotated = abs(lines[0][1]) > abs(lines[0][0])
    if rotated:
        lines.sort(key=lambda L: (round(L[2], 1), L[3]))
    else:
        lines.sort(key=lambda L: (round(L[3], 1), L[2]))
    return re.sub(r"\s+", " ", " ".join(L[4] for L in lines)).strip()


def _extract_grid(page, table) -> list[list[str]]:
    """Extract a table as a row-major grid with reading-order cell text.

    Falls back to PyMuPDF's Table.extract() text per-cell only when a cell has
    no recoverable dict text (e.g. empty cells), so behaviour is unchanged for
    plain horizontal tables and only the rotated cells get repaired.
    """
    raw = table.extract()
    grid: list[list[str]] = []
    for ri, trow in enumerate(table.rows):
        out_row: list[str] = []
        for ci, cbbox in enumerate(trow.cells):
            txt = _cell_text(page, cbbox) if cbbox else ""
            if not txt and ri < len(raw) and ci < len(raw[ri]):
                txt = re.sub(r"\s+", " ", str(raw[ri][ci] or "")).strip()
            out_row.append(txt)
        grid.append(out_row)
    return grid if grid else [
        [re.sub(r"\s+", " ", str(c or "")).strip() for c in r] for r in raw
    ]


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


def extract_annex_tables(
    pdf_path: str | Path, min_rows: int = 4, rebuild_cells: bool = False
) -> list:
    """Extract ruled tables from the PDF text layer, stitching multi-page runs.

    Returns a list of gazette_schema.Table.  Tables with fewer than min_rows
    data rows are ignored (boxed notes, mastheads).

    ``rebuild_cells`` (Phase 1 HTML-table feature, gated by ``html_tables_annex``)
    re-derives every cell from dict lines in reading order via ``_extract_grid``
    to repair rotated-cell text-bleed.  When False (default, and the validated
    baseline), cell text comes straight from PyMuPDF's ``Table.extract()`` —
    byte-identical to the pre-commit behaviour under ``annex_tables_fitz``.
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
                html=_to_html(cur_header, cur_rows),
                text_flat=_to_flat_text(cur_header, cur_rows),
                n_cols=len(cur_header),
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
            data = _extract_grid(page, t) if rebuild_cells else t.extract()
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
