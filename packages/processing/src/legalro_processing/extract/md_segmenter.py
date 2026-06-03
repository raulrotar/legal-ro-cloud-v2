"""Segment a full-gazette Markdown document into per-act Markdown blocks.

Strategy (in priority order):

1. **Heading-based** (primary): Docling emits `## ORDIN`, `## HOTĂRÂRE`, etc.
   as H2 headings at act boundaries.  Split on these.

2. **Closing-block anchor** (secondary): The pattern
   "București, DD LUNA YYYY.\n\nNr. NNN."
   always marks the end of an act.  Used to further split heading-merged blocks
   when two short acts share a heading.

3. **Sumar cross-check** (validation): expected act count from sumar vs produced.
   If |produced - expected| > max(2, expected*0.5), fall back to a coarser split.

Each returned `MdActBlock` carries:
  - `markdown`   — the raw Markdown text of this act (preserving tables, headings)
  - `plain_text` — the Markdown stripped to plain text (for regex fallbacks)
  - `title_hint` — the heading text, if any (often the act type + title)
  - `page_hints` — page numbers mentioned in the MD (from `---` page-break markers)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Heading patterns ──────────────────────────────────────────────────────────

# Docling emits act-type headers as H2 or H3.
# Match: ## ORDIN, ## HOTĂRÂRE, ## DECIZIE, ## DECRET, ## LEGE, etc.
_ACT_HEADING = re.compile(
    r'^#{1,3}\s+'
    r'(?:ORDIN(?:UL)?|HOT[ĂA]R[ÂA]RE[A]?|DECRET(?:-LEGE)?|DECIZ(?:IE|IA?)'
    r'|LEGE[A]?|OUG|ORDONAN[TȚ][ĂA]|COMUNICAT|RAPORT|ANUN[TȚ]|RECTIFICARE'
    r'|CURTEA\s+CONSTITU[TȚ]IONAL[ĂA]|GUVERNUL\s+ROM[ÂA]NIEI'
    r'|PRE[ȘS]EDINTELE\s+ROM[ÂA]NIEI|PARLAMENTUL\s+ROM[ÂA]NIEI'
    r'|MINISTERUL|AGEN[TȚ]IA|BANCA\s+NA[TȚ]IONAL[ĂA])',
    re.IGNORECASE | re.MULTILINE,
)

# Closing signature: "București, DD LUNA YYYY." followed (possibly) by "Nr. NNN."
_CLOSING_BLOCK = re.compile(
    r'Bucure[șs]ti,\s+\d{1,2}\s+\w+\s+\d{4}\.\s*\n[\s\S]{0,120}?Nr\.\s*[\d.]+\.',
    re.MULTILINE,
)

# Page-break markers from LlamaParse (\f) or Docling (horizontal rule)
_PAGE_BREAK = re.compile(r'\f|^---\s*$', re.MULTILINE)

# Strip markdown to plain text for regex fallbacks
_MD_HEADING_MARKER = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_TABLE_SEP = re.compile(r'^\|[\s\-:|]+\|\s*$', re.MULTILINE)
_MD_TABLE_ROW = re.compile(r'^\|(.+)\|\s*$', re.MULTILINE)
_MD_BOLD_ITALIC = re.compile(r'\*{1,3}([^*\n]+)\*{1,3}')
_MD_INLINE_CODE = re.compile(r'`([^`\n]+)`')


@dataclass
class MdActBlock:
    markdown: str
    plain_text: str
    title_hint: str = ""
    page_hints: list[int] = field(default_factory=list)


def segment_gazette_md(
    full_markdown: str,
    expected_act_count: int = 0,
) -> list[MdActBlock]:
    """Split a full-gazette Markdown into per-act blocks.

    Parameters
    ----------
    full_markdown:
        The complete Markdown output from md_extractor (one string, full doc).
    expected_act_count:
        Number of acts expected from the sumar.  0 = unknown (no validation).

    Returns
    -------
    list[MdActBlock]
        One entry per act, in document order.
    """
    blocks = _split_by_headings(full_markdown)

    if not blocks:
        # No headings found — fall back to closing-block splitting
        blocks = _split_by_closing(full_markdown)

    if not blocks:
        # Nothing found — whole document is one act
        blocks = [_make_block(full_markdown)]

    # Further split any block that contains multiple closing signatures
    blocks = _split_multi_closing(blocks)

    # Validation: if produced count is wildly off, fall back to closing-only
    if expected_act_count >= 2:
        ratio = len(blocks) / expected_act_count
        if ratio < 0.3 or ratio > 4.0:
            fallback = _split_by_closing(full_markdown)
            if fallback and abs(len(fallback) - expected_act_count) < abs(len(blocks) - expected_act_count):
                blocks = fallback

    return [b for b in blocks if b.plain_text.strip()]


# ── Internal ──────────────────────────────────────────────────────────────────

def _split_by_headings(markdown: str) -> list[MdActBlock]:
    """Split on act-type H2/H3 headings."""
    boundaries = [m.start() for m in _ACT_HEADING.finditer(markdown)]
    if not boundaries:
        return []

    blocks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(markdown)
        chunk = markdown[start:end]
        # Extract heading text as title hint
        first_line = chunk.splitlines()[0] if chunk.splitlines() else ""
        title_hint = _MD_HEADING_MARKER.sub("", first_line).strip()
        blocks.append(_make_block(chunk, title_hint=title_hint))

    return blocks


def _split_by_closing(markdown: str) -> list[MdActBlock]:
    """Split by closing signature blocks."""
    closings = list(_CLOSING_BLOCK.finditer(markdown))
    if not closings:
        return []

    blocks = []
    prev = 0
    for m in closings:
        chunk = markdown[prev:m.end()]
        if chunk.strip():
            blocks.append(_make_block(chunk))
        prev = m.end()
    # remainder after last closing
    remainder = markdown[prev:].strip()
    if remainder:
        blocks.append(_make_block(remainder))

    return blocks


def _split_multi_closing(blocks: list[MdActBlock]) -> list[MdActBlock]:
    """Further split any block that contains multiple closing signatures."""
    result = []
    for block in blocks:
        closings = list(_CLOSING_BLOCK.finditer(block.markdown))
        if len(closings) <= 1:
            result.append(block)
            continue
        prev = 0
        for i, m in enumerate(closings):
            chunk = block.markdown[prev:m.end()]
            if chunk.strip():
                result.append(_make_block(chunk, title_hint=block.title_hint if i == 0 else ""))
            prev = m.end()
        remainder = block.markdown[prev:].strip()
        if remainder:
            result.append(_make_block(remainder))

    return result


def _make_block(markdown: str, title_hint: str = "") -> MdActBlock:
    plain = _md_to_plain(markdown)
    # Extract page numbers from page-break markers or running headers
    page_nums = [
        int(m.group(1))
        for m in re.finditer(r'(?:^|\n)\s*(\d{1,3})\s*(?:\n|$)', plain)
        if 1 <= int(m.group(1)) <= 999
    ]
    return MdActBlock(
        markdown=markdown.strip(),
        plain_text=plain.strip(),
        title_hint=title_hint,
        page_hints=sorted(set(page_nums)),
    )


def _md_to_plain(text: str) -> str:
    """Strip Markdown syntax to plain text for regex fallbacks."""
    text = _MD_BOLD_ITALIC.sub(r'\1', text)
    text = _MD_INLINE_CODE.sub(r'\1', text)
    text = _MD_HEADING_MARKER.sub('', text)
    text = _MD_TABLE_SEP.sub('', text)
    text = _MD_TABLE_ROW.sub(
        lambda m: '  '.join(c.strip() for c in m.group(1).split('|') if c.strip()),
        text,
    )
    text = _PAGE_BREAK.sub('', text)
    return text
