"""Standalone PDF → GazetteDocument JSON extraction.

No database access, no embeddings. Pure: PDF in, JSON file out.

After every extraction the resulting JSON is validated automatically.
If ERROR-level issues are found (wrong act_number, UNKNOWN doc_type,
LLM failures, etc.) extraction is retried up to MAX_RETRIES times.
Remaining issues after all retries are preserved as extraction_warnings
in the JSON so they're visible downstream.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legalro_core.config import Settings

DEFAULT_EXTRACTED_DIR = "extracted"
MAX_RETRIES = 2   # maximum re-extraction attempts after first try

# Must match gazette_extractor.FILENAME_PATTERN
_FILENAME_RE = re.compile(
    r'MO_P([IV]+)_(\d+(?:Bis)?)_(\d{4})-(\d{2})-(\d{2})\.pdf', re.IGNORECASE
)


def run_extraction(
    pdf_path: str | Path,
    settings: "Settings",
    extracted_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Extract a PDF to a GazetteDocument JSON file.

    Pipeline:
      1. Check sha256 cache — skip if JSON already matches PDF (unless force=True).
      2. Extract PDF → GazetteDocument JSON.
      3. Validate the JSON for ERROR-level issues.
      4. Retry extraction (up to MAX_RETRIES) if fixable errors remain.
      5. Append any unfixable issues to extraction_warnings in the JSON.

    Returns the path to the written JSON file.
    """
    from legalro_processing.extract.gazette_extractor import extract_gazette, save_gazette, load_gazette
    from legalro_processing.extraction_validator import validate_file, needs_reextraction, Severity

    path = Path(pdf_path).resolve()
    out_dir = Path(extracted_dir) if extracted_dir else Path(DEFAULT_EXTRACTED_DIR)

    # ── Cache check ──────────────────────────────────────────────────────────
    current_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = _expected_json_path(path, out_dir)
    if not force and expected and expected.exists():
        try:
            cached = load_gazette(expected)
            if cached.sha256 == current_sha:
                return expected
        except Exception:
            pass  # corrupted cache — fall through to re-extract

    # ── Extract + validate loop ───────────────────────────────────────────────
    json_path: Path | None = None
    last_issues: list = []

    for attempt in range(1 + MAX_RETRIES):
        gazette = extract_gazette(path, settings)
        json_path = save_gazette(gazette, out_dir)

        issues = validate_file(json_path)
        errors = [i for i in issues if i.severity == Severity.ERROR]

        if not errors:
            # Clean extraction — done
            return json_path

        last_issues = errors
        error_summary = "; ".join(
            f"act[{i.act_index}] {i.check}: {i.message}" for i in errors
        )

        if attempt < MAX_RETRIES:
            # Delete and retry
            json_path.unlink(missing_ok=True)
            _log(f"[validate] {path.name} attempt {attempt + 1} — {len(errors)} error(s), "
                 f"retrying: {error_summary}")
        else:
            # Exhausted retries — annotate the JSON and return it
            _log(f"[validate] {path.name} — {len(errors)} error(s) remain after "
                 f"{1 + MAX_RETRIES} attempt(s); annotating JSON: {error_summary}")
            _annotate_remaining_issues(json_path, errors)

    return json_path  # type: ignore[return-value]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _expected_json_path(pdf_path: Path, out_dir: Path) -> Path | None:
    """Derive the expected JSON output path from the PDF filename."""
    m = _FILENAME_RE.match(pdf_path.name)
    if not m:
        return None
    year, month, day = m.group(3), m.group(4), m.group(5)
    return out_dir / year / month / day / f"{pdf_path.stem}.json"


def _annotate_remaining_issues(json_path: Path, errors: list) -> None:
    """Append validation errors to the gazette's extraction_warnings."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        existing = data.get("extraction_warnings", [])
        for issue in errors:
            tag = f"[VALIDATE:{issue.check}]"
            act = f"act[{issue.act_index}]" if issue.act_index is not None else "gazette"
            msg = f"{tag} {act}: {issue.message}"
            if msg not in existing:
                existing.append(msg)
        data["extraction_warnings"] = existing
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass  # annotation is best-effort


def _log(msg: str) -> None:
    import sys
    print(msg, file=sys.stderr)
