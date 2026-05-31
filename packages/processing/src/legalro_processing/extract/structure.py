"""Remove running headers, footers, page numbers, colophon."""
import re

RUNNING_HEADER = re.compile(
    r'MONITORUL\s+OFICIAL\s+AL\s+ROM[ÂA]NIEI,?\s*PARTEA\s+I,?\s*Nr\.\s*\d+.*?\n',
    re.IGNORECASE
)
PAGE_NUMBER = re.compile(r'^\s*\d{1,3}\s*$', re.MULTILINE)
COLOPHON_START = re.compile(
    r'(EDITOR:\s*PARLAMENTUL|Monitorul Oficial R\.A\.|Pre[tț]ul:|ABONAMENTE)',
    re.IGNORECASE
)

# OCR/footnote markers safe to remove. NOTE: we deliberately do NOT strip "(n)"
# because in Romanian legal text those are alineat numbers (legally significant
# and used by the ARTICLE chunking strategy), not footnote references.
OCR_MARKERS = re.compile(r'<[^>]{1,12}>|\[\d{1,3}\]|(?<!\S)\*(?!\S)|\bcite\b')


def clean_markers(text: str) -> str:
    from legalro_core.md_normalize import strip_markdown_artifacts
    text = OCR_MARKERS.sub('', text)
    return strip_markdown_artifacts(text)


def strip_structural(pages: list[str]) -> list[str]:
    result = []
    for i, page in enumerate(pages):
        if i > 0:
            page = RUNNING_HEADER.sub('', page)
        page = PAGE_NUMBER.sub('', page)
        match = COLOPHON_START.search(page)
        if match:
            page = page[:match.start()]
        page = clean_markers(page)
        result.append(page.strip())
    return result
