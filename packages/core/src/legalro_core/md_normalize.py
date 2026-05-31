"""Normalize LlamaParse/Mistral markdown output to plain-text per-page list.

Cloud OCR providers return a single markdown blob for the whole document.
This module splits it back into per-page plain text that matches the contract
expected by normalize_pages(), strip_structural(), and segment_acts().

Markdown constructs handled:
  - Headings (#, ##, ###…)       → plain text, marker stripped
  - Tables (| col | col |)        → rows flattened to tab-separated plain text
  - Table separator rows (|---|)  → removed
  - Images (![alt](url))          → removed (no useful legal text)
  - Code fences (``` ... ```)     → content preserved, fences stripped
  - Inline code (`text`)          → content preserved, backticks stripped
  - Bold/italic (**text**, *text*)→ plain text, markers stripped
  - Horizontal rules (---, ***)  → used as page-break signal, then removed
  - Form feeds (\\f)              → page-break signal
"""
from __future__ import annotations

import re

# ── Page-break signals ────────────────────────────────────────────────────────
# LlamaParse uses \f (form feed) between pages.
# Some providers use a run of dashes on its own line.
_PAGE_BREAK = re.compile(r'\f|^-{3,}\s*$|^\*{3,}\s*$', re.MULTILINE)

# ── Markdown stripping patterns ───────────────────────────────────────────────
_MD_HEADING      = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_IMAGE        = re.compile(r'!\[[^\]]*\]\([^)]*\)')
_MD_CODE_FENCE   = re.compile(r'```[^\n]*\n(.*?)```', re.DOTALL)
_MD_INLINE_CODE  = re.compile(r'`([^`\n]+)`')
_MD_BOLD_ITALIC  = re.compile(r'\*{1,3}([^*\n]+)\*{1,3}')
_MD_TABLE_SEP    = re.compile(r'^\|[\s\-:|]+\|\s*$', re.MULTILINE)
_MD_TABLE_ROW    = re.compile(r'^\|(.+)\|\s*$', re.MULTILINE)

# Residual markdown artifacts that survive partial normalization
_MD_RESIDUAL     = re.compile(r'^#{1,6}\s+|!\[[^\]]*\]\([^)]*\)', re.MULTILINE)


def normalize_llamaparse_markdown(markdown: str) -> list[str]:
    """Split a whole-document markdown string into per-page plain text.

    Returns a list[str] with one entry per page, matching the contract of
    extract_text() → normalize_pages() → strip_structural() → segment_acts().
    """
    raw_pages = _PAGE_BREAK.split(markdown)
    result = []
    for page in raw_pages:
        cleaned = _strip_markdown(page)
        if cleaned.strip():
            result.append(cleaned)
    return result if result else [_strip_markdown(markdown)]


def strip_markdown_artifacts(text: str) -> str:
    """Strip residual markdown syntax from text extracted by any provider.

    Safe to call on PyMuPDF output too — the patterns are narrow enough
    not to touch normal legal text.
    """
    return _MD_RESIDUAL.sub('', text)


# ── Internal ──────────────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    # Remove images entirely (no useful text content)
    text = _MD_IMAGE.sub('', text)
    # Code fences: keep content, drop fences
    text = _MD_CODE_FENCE.sub(r'\1', text)
    # Inline code: keep content, drop backticks
    text = _MD_INLINE_CODE.sub(r'\1', text)
    # Bold/italic: keep content, drop markers
    text = _MD_BOLD_ITALIC.sub(r'\1', text)
    # Headings: strip marker, keep text
    text = _MD_HEADING.sub('', text)
    # Tables: flatten to plain text
    text = _flatten_tables(text)
    return text


def _flatten_tables(text: str) -> str:
    """Convert markdown tables to plain indented text.

    Separator rows (| --- | --- |) are removed.
    Data rows (| cell | cell |) become tab-separated lines, preserving
    all cell content so table data (e.g. annexe schedules) remains searchable.
    """
    # Remove separator rows first
    text = _MD_TABLE_SEP.sub('', text)

    # Flatten data rows: | col1 | col2 | → "col1\tcol2"
    def _flatten_row(m: re.Match) -> str:
        cells = [c.strip() for c in m.group(1).split('|')]
        cells = [c for c in cells if c]  # drop empty border cells
        return '\t'.join(cells)

    text = _MD_TABLE_ROW.sub(_flatten_row, text)
    return text
