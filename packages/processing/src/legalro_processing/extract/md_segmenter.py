"""Segment a full-gazette Markdown document into per-act Markdown blocks.

Strategy:

1. **Normalize letterspacing** in headings: Docling emits letterspaced headers
   from born-digital PDFs — "D E C I Z I A   Nr. 576" → "DECIZIA Nr. 576",
   "R E C T I F I C Ă R I" → "RECTIFICĂRI".

2. **Heading-based segmentation** on the normalized MD:
   - Act-type headings (DECIZIE, HOTĂRÂRE, ORDIN, …) → act boundaries
   - Category / body / signature headings → skipped (not boundaries)

3. **Closing-block secondary split**: "București, DD LUNA YYYY. Nr. NNN."
   further splits any block that contains multiple acts (e.g. two short PM
   decisions merged under one category header).

4. **Sumar cross-check**: if produced count is wildly off from expected,
   fall back to closing-block-only segmentation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Heading patterns (applied AFTER letterspacing normalization) ──────────────

# Act-type headings that always mark the start of a new act.
# Applied after _normalize_heading_letterspacing() so no letter-spaces to handle.
_ACT_HEADING = re.compile(
    r'^#{1,3}\s+(?:'
    r'DECIZIA\b'               # DECIZIA Nr. 576 (DCC or other numbered decision)
    r'|DECIZIE\b'              # standalone DECIZIE (PM / agency decisions)
    r'|HOT[ĂA]R[ÂA]RE[A]?\b'  # HOTĂRÂRE / HOTĂRÂREA (HG)
    r'|DECRET(?:-LEGE)?\b'     # DECRET / DECRET-LEGE
    r'|ORDIN\b'                # ORDIN (not ORDINUL used in references)
    r'|LEGE[A]?\b'             # LEGE / LEGEA
    r'|OUG\b'
    r'|ORDONAN[TȚ][ĂA]\b'
    r'|COMUNICAT\b'
    r'|RAPORT\b'
    r'|RECTIFIC[ĂA]RI\b'
    r'|ANUN[TȚ]\b'
    r')',
    # Note: institution names (GUVERNUL ROMÂNIEI, CURTEA CONSTITUȚIONALĂ,
    # PREȘEDINTELE ROMÂNIEI, etc.) are NOT act boundaries — they are context
    # headers that appear before the act-type heading.  Act-type headings
    # (DECIZIE, HOTĂRÂRE, ORDIN…) provide the correct boundaries.
    re.IGNORECASE | re.MULTILINE,
)

# Headings that are NOT act boundaries and should be skipped.
# Applied after normalization.
_SKIP_HEADING = re.compile(
    r'^#{1,3}\s+(?:'
    # Table-of-contents / gazette structure
    r'S\s*U\s*M\s*A\s*R\b'
    r'|PARTEA\s+[IVX]'
    # Category / section headers (contain "ALE" or "ȘI")
    r'|.+\bALE\b'                      # DECIZII ALE, HOTĂRÂRI ALE, ACTE ALE
    r'|LEGI,\s+DECRETE'
    # Body headings within DCC decisions
    r'|CURTEA(?:[,\s]|$)'              # ## CURTEA, or ## CURTEA CONSTITUȚIONALĂ (body)
    r'|ÎN\s+NUMELE'                    # ## În numele legii
    r'|DECIDE\b'                       # ## DECIDE: (decision body)
    # Signature headings within acts
    r'|PRIM-?MINISTR'                  # ## PRIM-MINISTRU ILIE-GAVRIL BOLOJAN
    r'|MINISTR(?:UL|UL\s)'             # ## MINISTRUL FINANȚELOR (signature)
    # Footer / publisher
    r'|EDITOR'
    r'|MONITORUL\s+OFICIAL'
    r')',
    re.IGNORECASE | re.MULTILINE,
)

# Closing signature block
_CLOSING_BLOCK = re.compile(
    r'Bucure[șs]ti,\s+\d{1,2}\s+\w+\s+\d{4}\.\s*[\s\S]{0,150}?Nr\.\s*[\d.]+\.',
    re.MULTILINE,
)

# Page-break markers
_PAGE_BREAK = re.compile(r'\f|^---\s*$', re.MULTILINE)

# Markdown stripping helpers for plain_text output
_MD_HEADING_MARKER = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_TABLE_SEP      = re.compile(r'^\|[\s\-:|]+\|\s*$', re.MULTILINE)
_MD_TABLE_ROW      = re.compile(r'^\|(.+)\|\s*$', re.MULTILINE)
_MD_BOLD_ITALIC    = re.compile(r'\*{1,3}([^*\n]+)\*{1,3}')
_MD_INLINE_CODE    = re.compile(r'`([^`\n]+)`')


@dataclass
class MdActBlock:
    markdown: str
    plain_text: str
    title_hint: str = ""
    page_hints: list[int] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def segment_gazette_md(
    full_markdown: str,
    expected_act_count: int = 0,
    era=None,
) -> list[MdActBlock]:
    """Split a full-gazette Markdown into per-act blocks.

    Args:
        full_markdown: The full gazette Markdown text.
        expected_act_count: Number of acts from the sumar (0 = unknown).
        era: Optional Era enum value.  When provided and the era indicates
             scanned/OCR content (era.value contains "scanned"), the orphan-merge
             token threshold is lowered to 20 (from 50) to avoid merging real
             short acts that appear in noisy 1989-era text.
    """
    # Step 1: normalize letterspacing in headings
    normalized = _normalize_md_headings(full_markdown)

    # Step 2: heading-based primary segmentation
    blocks = _split_by_headings(normalized)

    # Step 3: closing-block secondary split (handles merged short acts)
    blocks = _split_multi_closing(blocks)

    # Step 4: fallback if heading segmentation found nothing
    if not blocks:
        blocks = _split_by_closing(normalized)
    if not blocks:
        blocks = [_make_block(normalized)]

    # Step 5: sumar cross-check — tightened upper bound (2.0 instead of 4.0) so
    # 2-4× over-splits also trigger the closing-block fallback comparison.
    if expected_act_count >= 2:
        ratio = len(blocks) / expected_act_count
        if ratio < 0.3 or ratio > 2.0:
            fallback = _split_by_closing(normalized)
            if fallback and abs(len(fallback) - expected_act_count) < abs(len(blocks) - expected_act_count):
                blocks = fallback

    # Step 6: drop artefact blocks — footnote fragments and bare category headers
    blocks = [b for b in blocks if b.plain_text.strip() and not _is_artefact(b)]

    # Step 7: merge orphan blocks — blocks with fewer than N tokens that have no
    # act-type header are sentence fragments from over-segmentation (e.g. a numbered
    # list item split at a heading boundary). Absorb them into the preceding block.
    #
    # Era-aware threshold: scanned (1989) content is noisier; lower the threshold
    # so short real acts are not accidentally merged into a neighbour.
    #
    # Guard fix: check the raw MARKDOWN first line (which still has '#' heading
    # markers) instead of plain_text (which has markers stripped, making the
    # `^#{1,3}\s+`-anchored _ACT_HEADING pattern unmatchable against it).
    _era_val = getattr(era, "value", "") or ""
    _orphan_threshold = 20 if "scanned" in _era_val.lower() else 50
    if len(blocks) > 1:
        merged: list[MdActBlock] = [blocks[0]]
        for blk in blocks[1:]:
            tokens = blk.plain_text.split()
            _first_md_line = blk.markdown.splitlines()[0] if blk.markdown else ""
            _has_act_heading = bool(_ACT_HEADING.match(_first_md_line))
            if len(tokens) < _orphan_threshold and not _has_act_heading:
                prev = merged[-1]
                merged[-1] = _make_block(
                    prev.markdown + "\n\n" + blk.markdown,
                    title_hint=prev.title_hint,
                )
            else:
                merged.append(blk)
        blocks = merged

    return blocks


# ── Letterspacing normalization ───────────────────────────────────────────────

def _normalize_md_headings(markdown: str) -> str:
    """Collapse letterspaced headings in the full Markdown.

    "## D E C I Z I A   Nr. 576" → "## DECIZIA Nr. 576"
    "## R E C T I F I C Ă R I"  → "## RECTIFICĂRI"

    Only touches heading lines (starting with #).
    Body text is unchanged.  Normal multi-character words like
    "GUVERNUL ROMÂNIEI" or "PRIM-MINISTRU ILIE-GAVRIL" are NOT collapsed.
    """
    lines = markdown.splitlines()
    result = []
    for line in lines:
        if line.startswith('#'):
            result.append(_normalize_one_heading(line))
        else:
            result.append(line)
    return '\n'.join(result)


def _normalize_one_heading(line: str) -> str:
    """Collapse letter-spacing in a heading line if the content is letterspaced.

    Detection heuristic: a heading is letterspaced when ≥ 70% of its alphabetic
    tokens are single characters (e.g. "D E C R E T" → each token is 1 char).
    Normal words like "GUVERNUL" or "PRIM-MINISTRU" have multi-char tokens so
    they are left untouched.
    """
    m = re.match(r'^(#{1,6}\s+)(.*)', line)
    if not m:
        return line
    prefix, text = m.group(1), m.group(2)

    # Classify tokens
    tokens = text.split(' ')
    alpha_tokens = [t for t in tokens if t and re.match(r'^[A-ZĂÂÎȘȚa-zăâîșț]+$', t)]
    single_char   = sum(1 for t in alpha_tokens if len(t) == 1)

    if not alpha_tokens or single_char / len(alpha_tokens) < 0.70:
        # Not letterspaced — leave as-is (avoids collapsing "GUVERNUL ROMÂNIEI")
        return line

    # It IS letterspaced — collapse single spaces between consecutive uppercase
    # letters; double/triple spaces (word separators) are preserved then normalized.
    collapsed = re.sub(r'([A-ZĂÂÎȘȚ]) (?=[A-ZĂÂÎȘȚ])', r'\1', text)
    collapsed = re.sub(r'  +', ' ', collapsed).strip()
    return prefix + collapsed


# ── Segmentation strategies ───────────────────────────────────────────────────

def _split_by_headings(normalized_markdown: str) -> list[MdActBlock]:
    """Split on act-type H1/H2/H3 headings, skipping category/body headings."""
    boundaries: list[int] = []

    for m in _ACT_HEADING.finditer(normalized_markdown):
        line_start = m.start()
        line_text = normalized_markdown[line_start:normalized_markdown.find('\n', line_start)]

        # Skip headings that are category headers or body sections
        if _SKIP_HEADING.match(line_text):
            continue

        boundaries.append(line_start)

    if not boundaries:
        return []

    blocks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(normalized_markdown)
        chunk = normalized_markdown[start:end]
        first_line = chunk.splitlines()[0] if chunk.splitlines() else ""
        title_hint = _MD_HEADING_MARKER.sub('', first_line).strip()
        blocks.append(_make_block(chunk, title_hint=title_hint))

    return blocks


def _split_by_closing(markdown: str) -> list[MdActBlock]:
    """Split by closing signature blocks (fallback)."""
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
        for j, m in enumerate(closings):
            chunk = block.markdown[prev:m.end()]
            if chunk.strip():
                result.append(_make_block(
                    chunk,
                    title_hint=block.title_hint if j == 0 else "",
                ))
            prev = m.end()
        remainder = block.markdown[prev:].strip()
        if remainder:
            result.append(_make_block(remainder))
    return result


# ── Artefact detection ────────────────────────────────────────────────────────

# Blocks starting with a footnote marker (*) are page-footer fragments that
# Docling placed on the wrong side of an act boundary — not standalone acts.
_FOOTNOTE_START = re.compile(r'^\s*\*\)', re.MULTILINE)

# Category-header-only blocks: letterspaced or plain ALL-CAPS category lines
# (e.g. "A C T E  A L E  P A R T I D E L O R  P O L I T I C E") with no
# recognisable act body beneath them.
_ACT_BODY_SIGNAL = re.compile(
    r'Art\.\s*\d+|Articol|având\s+în\s+vedere|în\s+temeiul|'
    r'hotărăşte|dispune|emite|se\s+abrog|se\s+aprobă',
    re.IGNORECASE,
)

# Publisher/printer masthead footer — unique tokens that only appear in the
# Monitorul Oficial R.A. footer block, NOT in real act bodies.
# Do NOT key on bare "Monitorul Oficial" — it appears in act bodies ("se publică în MO").
_PUBLISHER_FOOTER = re.compile(
    r'EDITOR:\s*GUVERNUL\s+ROMÂNIEI'
    r"|'Monitorul\s+Oficial'\s+R\.A\."  # the legal entity name with quotes
    r"|www\.monitoruloficial\.ro"
    r'|C\.I\.F\.\s+RO\d+'              # fiscal code
    r'|IBAN:\s*RO\d+'                  # bank account
    r'|Tiparul:',
    re.IGNORECASE,
)


def _is_artefact(block: MdActBlock) -> bool:
    """Return True for blocks that are segmentation artefacts, not real acts."""
    text = block.plain_text.strip()

    # Very short blocks with no act-body signal
    if len(text) < 150 and not _ACT_BODY_SIGNAL.search(text):
        return True

    # Blocks that start with a footnote marker (*)
    if _FOOTNOTE_START.match(text):
        return True

    # Publisher/printer masthead footer (EDITOR, IBAN, CIF, Tiparul, URL)
    # Require absence of act-body signal as a second gate so a real act that
    # happens to mention the publisher is not accidentally dropped.
    if _PUBLISHER_FOOTER.search(text) and not _ACT_BODY_SIGNAL.search(text):
        return True

    return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_block(markdown: str, title_hint: str = "") -> MdActBlock:
    plain = _md_to_plain(markdown)
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
    """Strip Markdown syntax to plain text."""
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
