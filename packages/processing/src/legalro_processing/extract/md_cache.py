"""Markdown cache — save/load gazette .md files alongside the JSON cache.

The MD file is the intermediate product of PDF→Docling/OCR extraction.
Caching it means:
  1. Re-running the LLM structuring step never re-invokes Docling or cloud OCR.
  2. The Markdown is human-inspectable and verifiable against the original PDF.
  3. The sha256 of the source PDF is embedded in the file header so cache
     invalidation is automatic when the PDF changes.

File layout (md_cache/{year}/{month}/{day}/{gazette_id}.md):
  <!--legalro:sha256={sha256}-->
  <!--legalro:era={era}-->
  <!--legalro:source={filename}-->
  <full Markdown content>
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


_HEADER_SHA = re.compile(r'<!--legalro:sha256=([0-9a-f]{64})-->')
_HEADER_ERA = re.compile(r'<!--legalro:era=(\w+)-->')

DEFAULT_MD_CACHE_DIR = "md_cache"


def cache_path(pdf_path: str | Path, md_cache_dir: str | Path = DEFAULT_MD_CACHE_DIR) -> Path:
    """Derive the .md cache path from the PDF filename."""
    pdf = Path(pdf_path)
    # Mirror laws/{year}/{month}/{day} structure
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', pdf.stem)
    if m:
        sub = Path(m.group(1)) / m.group(2) / m.group(3)
    else:
        sub = Path("unknown")
    return Path(md_cache_dir) / sub / f"{pdf.stem}.md"


def load(pdf_path: str | Path, md_cache_dir: str | Path = DEFAULT_MD_CACHE_DIR) -> str | None:
    """Load cached Markdown for this PDF if it exists and sha256 matches.

    Returns the Markdown string, or None if the cache is missing/stale.
    """
    path = cache_path(pdf_path, md_cache_dir)
    if not path.exists():
        return None

    content = path.read_text(encoding="utf-8")
    m = _HEADER_SHA.search(content)
    if not m:
        return None

    current_sha = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    if m.group(1) != current_sha:
        return None  # PDF changed — cache stale

    # Strip the header lines to return clean Markdown
    lines = content.splitlines()
    body_start = next((i for i, line in enumerate(lines) if not line.startswith("<!--legalro:")), 0)
    return "\n".join(lines[body_start:])


def save(
    pdf_path: str | Path,
    markdown: str,
    era: str,
    md_cache_dir: str | Path = DEFAULT_MD_CACHE_DIR,
) -> Path:
    """Save Markdown to the cache with a sha256-keyed header."""
    sha256 = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    path = cache_path(pdf_path, md_cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"<!--legalro:sha256={sha256}-->\n"
        f"<!--legalro:era={era}-->\n"
        f"<!--legalro:source={Path(pdf_path).name}-->\n"
    )
    path.write_text(header + markdown, encoding="utf-8")
    return path
