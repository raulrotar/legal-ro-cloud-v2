"""Standalone PDF → GazetteDocument JSON extraction.

No database access, no embeddings. Pure: PDF in, JSON file out.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legalro_core.config import Settings

DEFAULT_EXTRACTED_DIR = "extracted"

# Must match gazette_extractor.FILENAME_PATTERN
_FILENAME_RE = re.compile(
    r'MO_P([IV]+)_(\d+(?:Bis)?)_(\d{4})-(\d{2})-(\d{2})\.pdf', re.IGNORECASE
)


def run_extraction(
    pdf_path: str | Path,
    settings: "Settings",
    extracted_dir: str | Path | None = None,
) -> Path:
    """Extract a PDF to a GazetteDocument JSON file.

    Skips re-extraction when a JSON already exists whose sha256 matches the PDF.
    Returns the path to the written JSON file.
    """
    from legalro_processing.extract.gazette_extractor import extract_gazette, save_gazette, load_gazette

    path = Path(pdf_path).resolve()
    out_dir = Path(extracted_dir) if extracted_dir else Path(DEFAULT_EXTRACTED_DIR)

    # Cheap sha256 check before running full OCR/extraction
    current_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = _expected_json_path(path, out_dir)
    if expected and expected.exists():
        try:
            cached = load_gazette(expected)
            if cached.sha256 == current_sha:
                return expected
        except Exception:
            pass  # corrupted cache — fall through to re-extract

    gazette = extract_gazette(path, settings)
    return save_gazette(gazette, out_dir)


def _expected_json_path(pdf_path: Path, out_dir: Path) -> Path | None:
    """Derive the expected JSON output path from the PDF filename without extracting."""
    m = _FILENAME_RE.match(pdf_path.name)
    if not m:
        return None
    year, month, day = m.group(3), m.group(4), m.group(5)
    stem = pdf_path.stem
    return out_dir / year / month / day / f"{stem}.json"
