"""Layout triage: detect and extract table-dense regions from gazette markdown.

Table-heavy pages (e.g. AEP party-financing reports, MO_PI_311) contain large
pipe tables that cause the act segmenter to mint phantom acts out of table rows.
This module:
  1. Scans normalized markdown for contiguous table regions.
  2. Returns them as Table objects (verbatim markdown + metadata).
  3. Returns the original markdown with those regions masked so that
     md_segmenter only sees act-dense content.

All heuristics are pure regex — no models loaded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from legalro_processing.extract.gazette_schema import Table

# Reuse the same table-line patterns as md_segmenter to stay consistent.
_MD_TABLE_SEP = re.compile(r'^\|[\s\-:|]+\|\s*$')
_MD_TABLE_ROW = re.compile(r'^\|.+\|\s*$')
_PAGE_BREAK = re.compile(r'<!--\s*legalro:page-break\s*-->')
_HEADING = re.compile(r'^#{1,4}\s+(.+)', re.MULTILINE)

# A region is "table-dense" when it has at least this many consecutive table rows.
_MIN_TABLE_ROWS = 4

# Placeholder inserted in the masked markdown so the segmenter keeps correct
# line count / page hints but ignores the table content.
_TABLE_PLACEHOLDER = "[TABELUL EXTRAS — conținut în câmpul tables]"


def find_table_regions(markdown: str) -> tuple[list[Table], str]:
    """Detect table-dense regions in *markdown*.

    Returns:
        tables: extracted Table objects (verbatim markdown, page hint, title).
        masked_md: the original markdown with table regions replaced by a short
                   placeholder so md_segmenter cannot mint phantom acts from them.
    """
    lines = markdown.splitlines(keepends=True)
    n = len(lines)

    # Track current page (0-based) as we scan page-break markers.
    current_page = 0
    # Track the last non-table heading seen, for use as table title.
    last_heading = ""

    # Mark which lines belong to a table region.
    in_table: list[bool] = [False] * n

    i = 0
    while i < n:
        line = lines[i].rstrip("\n").rstrip()

        # Update page counter on page-break markers.
        if _PAGE_BREAK.search(line):
            current_page += 1
            i += 1
            continue

        # Update last heading.
        hm = _HEADING.match(line)
        if hm:
            last_heading = hm.group(1).strip()
            i += 1
            continue

        # Detect start of a table block.
        if _MD_TABLE_ROW.match(line) or _MD_TABLE_SEP.match(line):
            # Scan forward to find the extent of this contiguous table block.
            j = i
            while j < n and (_MD_TABLE_ROW.match(lines[j].rstrip()) or
                              _MD_TABLE_SEP.match(lines[j].rstrip()) or
                              lines[j].strip() == ""):
                j += 1
            row_count = sum(
                1 for k in range(i, j)
                if _MD_TABLE_ROW.match(lines[k].rstrip())
                and not _MD_TABLE_SEP.match(lines[k].rstrip())
            )
            if row_count >= _MIN_TABLE_ROWS:
                for k in range(i, j):
                    in_table[k] = True
            i = j
            continue

        i += 1

    # Build Table objects and construct masked markdown.
    tables: list[Table] = []
    masked_lines: list[str] = []

    i = 0
    current_page = 0
    last_heading = ""

    while i < n:
        line = lines[i].rstrip("\n").rstrip()

        if _PAGE_BREAK.search(line):
            current_page += 1
            masked_lines.append(lines[i])
            i += 1
            continue

        hm = _HEADING.match(line)
        if hm:
            last_heading = hm.group(1).strip()
            masked_lines.append(lines[i])
            i += 1
            continue

        if in_table[i]:
            # Collect the whole contiguous marked region.
            j = i
            while j < n and in_table[j]:
                j += 1
            table_lines = lines[i:j]
            table_md = "".join(table_lines).strip()
            n_rows = sum(
                1 for l in table_lines
                if _MD_TABLE_ROW.match(l.rstrip())
                and not _MD_TABLE_SEP.match(l.rstrip())
            )
            tables.append(Table(
                markdown=table_md,
                page=current_page,
                title=last_heading,
                n_rows=n_rows,
            ))
            masked_lines.append(_TABLE_PLACEHOLDER + "\n")
            i = j
            continue

        masked_lines.append(lines[i])
        i += 1

    return tables, "".join(masked_lines)
