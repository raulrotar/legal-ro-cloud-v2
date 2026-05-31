"""Parse table of contents from page 1 of gazette."""
import re
from dataclasses import dataclass


@dataclass
class SumarBoundary:
    title: str
    page_number: int


SUMAR_MARKER = re.compile(r'S\s*U\s*M\s*A\s*R', re.IGNORECASE)

# Raw entry regex — matched AFTER normalising wrapped lines.
_ENTRY_RE = re.compile(r'^(.+?)\s*\.{2,}\s*(\d+)', re.MULTILINE)


def parse_sumar(first_page_text: str) -> list[SumarBoundary]:
    match = SUMAR_MARKER.search(first_page_text)
    if not match:
        return []

    sumar_text = first_page_text[match.end():]

    # Step 1: collapse standalone page-number-only lines (e.g. "2–15" with no dot
    # leader) into " ... N" appended to the preceding line, so _ENTRY_RE can match.
    # Must run before the dot-collapse step so we don't double-process.
    normalized = re.sub(r'\n(\d+)(?:[–\-]\d+)?(?=\n|$)', r' ... \1', sumar_text)
    # Step 2: dots at end of line, page number on the next line → inline.
    normalized = re.sub(r'(\.{2,})\s*\n\s*(\d+(?:[–\-]\d+)?)', r'\1 \2', normalized)

    entries = []
    for m in _ENTRY_RE.finditer(normalized):
        title = m.group(1).strip()
        page_num = int(m.group(2))
        if title and page_num > 0:
            entries.append(SumarBoundary(title=title, page_number=page_num))

    return entries
