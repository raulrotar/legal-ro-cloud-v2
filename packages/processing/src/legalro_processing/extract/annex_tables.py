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


def _span_grid(bbox_grid: list[list]) -> tuple[list, list]:
    """Derive (colspan, rowspan) per cell from source bboxes (mechanism (a)).

    PyMuPDF's find_tables(lines_strict) reports the 294Bis Nomenclator on a
    uniform N-column grid, but a *merged* cell carries ONE wide bbox spanning
    several grid columns/rows while the positions it covers are ``None`` (see
    the page-3 domain band: ``Informatică`` is one cell whose x-range covers 4
    grid columns, cols to its right being ``None``).  So spans are read off the
    geometry, not by collapsing equal text.

    Builds the global x-edge / y-edge grid from every non-None bbox, then for
    each real cell counts how many grid columns its x-range covers (colspan)
    and how many grid rows its y-range covers (rowspan).  Returns two parallel
    grids; ``None`` marks a covered position to skip when emitting HTML.
    """
    xs: set = set()
    ys: set = set()
    for ri, row in enumerate(bbox_grid):
        for cb in row:
            if cb:
                xs.add(round(cb[0], 1)); xs.add(round(cb[2], 1))
                ys.add(round(cb[1], 1)); ys.add(round(cb[3], 1))
    xe = sorted(xs)
    ye = sorted(ys)

    def _idx(v: float, edges: list) -> int:
        return min(range(len(edges)), key=lambda i: abs(edges[i] - v))

    colspans: list = []
    rowspans: list = []
    for row in bbox_grid:
        cs_row: list = []
        rs_row: list = []
        for cb in row:
            if cb is None:
                cs_row.append(None); rs_row.append(None)
                continue
            cs = max(1, _idx(cb[2], xe) - _idx(cb[0], xe)) if len(xe) > 1 else 1
            rs = max(1, _idx(cb[3], ye) - _idx(cb[1], ye)) if len(ye) > 1 else 1
            cs_row.append(cs); rs_row.append(rs)
        colspans.append(cs_row)
        rowspans.append(rs_row)
    return colspans, rowspans


def _to_html(header: list, rows: list[list], bbox_grid: list[list] | None = None) -> str:
    """Single-line flat grid <table>.  Cells are whitespace-collapsed and
    HTML-escaped.

    When ``bbox_grid`` is supplied (the ``rebuild_cells`` / ``html_tables_annex``
    path), TRUE ``colspan``/``rowspan`` are derived from cell geometry via
    ``_span_grid`` and ``None`` positions (covered by a span) are skipped, so a
    merged domain band is emitted once as e.g. ``<th colspan="3">Matematică</th>``
    instead of three repeated leaf cells (QA §3.2/§3.3).  Without ``bbox_grid``
    (flags-off baseline) it falls back to the Phase-1 flat grid — byte-identical.
    """
    if bbox_grid is None:
        parts = ["<table>"]
        parts.append("<tr>" + "".join(
            f"<th>{_esc(_clean_cell(c))}</th>" for c in header) + "</tr>")
        for r in rows:
            parts.append("<tr>" + "".join(
                f"<td>{_esc(_clean_cell(c))}</td>" for c in r) + "</tr>")
        parts.append("</table>")
        return "".join(parts)

    grid = [header] + list(rows)
    colspans, rowspans = _span_grid(bbox_grid)

    def _attrs(cs: int, rs: int) -> str:
        a = ""
        if cs > 1:
            a += f' colspan="{cs}"'
        if rs > 1:
            a += f' rowspan="{rs}"'
        return a

    parts = ["<table>"]
    for ri, row in enumerate(grid):
        cells: list[str] = []
        for ci, c in enumerate(row):
            # Skip positions covered by another cell's span (bbox is None), or
            # rows/cols beyond the geometric grid (defensive).
            if ri >= len(colspans) or ci >= len(colspans[ri]):
                tag = "th" if ri == 0 else "td"
                cells.append(f"<{tag}>{_esc(_clean_cell(c))}</{tag}>")
                continue
            cs = colspans[ri][ci]
            if cs is None:
                continue
            rs = rowspans[ri][ci]
            # Header cell when it's the top header row OR a merged grouping band
            # (a cell spanning >1 leaf column is, in this transposed nomenclator,
            # a domain/header band — e.g. <th colspan="4">Informatică</th>).
            tag = "th" if (ri == 0 or cs > 1 or rs > 1) else "td"
            cells.append(f"<{tag}{_attrs(cs, rs)}>{_esc(_clean_cell(c))}</{tag}>")
        parts.append("<tr>" + "".join(cells) + "</tr>")
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


def _extract_grid(page, table) -> tuple[list[list[str]], list[list]]:
    """Extract a table as a row-major grid with reading-order cell text, plus a
    parallel grid of each cell's source bbox (``None`` where the position is
    covered by another cell's merge — see ``_span_grid``).

    Falls back to PyMuPDF's Table.extract() text per-cell only when a cell has
    no recoverable dict text (e.g. empty cells), so behaviour is unchanged for
    plain horizontal tables and only the rotated cells get repaired.
    """
    raw = table.extract()
    grid: list[list[str]] = []
    bboxes: list[list] = []
    for ri, trow in enumerate(table.rows):
        out_row: list[str] = []
        bb_row: list = []
        for ci, cbbox in enumerate(trow.cells):
            txt = _cell_text(page, cbbox) if cbbox else ""
            if not txt and ri < len(raw) and ci < len(raw[ri]):
                txt = re.sub(r"\s+", " ", str(raw[ri][ci] or "")).strip()
            out_row.append(txt)
            bb_row.append(tuple(cbbox) if cbbox else None)
        grid.append(out_row)
        bboxes.append(bb_row)
    if grid:
        return grid, bboxes
    flat = [[re.sub(r"\s+", " ", str(c or "")).strip() for c in r] for r in raw]
    return flat, [[None] * len(r) for r in flat]


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
    # parallel bbox grid (rebuild_cells path only) for true-span HTML.  Stays
    # aligned with [cur_header] + cur_rows; None entries mean span-covered.
    cur_bb_header: list | None = None
    cur_bb_rows: list[list] = []

    def _flush() -> None:
        nonlocal cur_header, cur_rows, cur_bb_header, cur_bb_rows
        if cur_header is not None and len(cur_rows) >= min_rows:
            bbox_grid = None
            if cur_bb_header is not None:
                bbox_grid = [cur_bb_header] + cur_bb_rows
            out.append(Table(
                markdown=_to_markdown(cur_header, cur_rows),
                html=_to_html(cur_header, cur_rows, bbox_grid),
                text_flat=_to_flat_text(cur_header, cur_rows),
                n_cols=len(cur_header),
                page=cur_page0,
                title=cur_title,
                n_rows=len(cur_rows),
            ))
        cur_header, cur_rows = None, []
        cur_bb_header, cur_bb_rows = None, []

    for pno, page in enumerate(doc):
        try:
            tabs = page.find_tables(strategy="lines_strict").tables
        except Exception:
            tabs = []
        if not tabs:
            _flush()
            continue
        for t in tabs:
            if rebuild_cells:
                data, bb = _extract_grid(page, t)
            else:
                data, bb = t.extract(), None
            if not data:
                continue
            header, rows = data[0], data[1:]
            bb_header = bb[0] if bb else None
            bb_rows = bb[1:] if bb else []
            if cur_header is not None and len(header) == len(cur_header):
                # continuation: repeated header → drop it; headerless → all rows
                if _row_key(header) == _row_key(cur_header):
                    cur_rows.extend(rows)
                    cur_bb_rows.extend(bb_rows)
                    continue
                if not any(str(c or "").strip() for c in header) or len(rows) == 0:
                    cur_rows.extend([header] + rows)
                    cur_bb_rows.extend([bb_header] + bb_rows if bb else [])
                    continue
                # same width but a genuinely new header → new table
            _flush()
            cur_header, cur_rows = header, rows
            cur_bb_header, cur_bb_rows = bb_header, bb_rows
            cur_page0 = pno
            cur_title = _table_title(page, t.bbox)
    _flush()
    doc.close()
    if out:
        print(f"[annex-tables] {Path(pdf_path).stem}: {len(out)} stitched table(s), "
              f"{sum(t.n_rows for t in out)} rows", file=sys.stderr, flush=True)
    return out
