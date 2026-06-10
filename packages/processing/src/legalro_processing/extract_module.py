"""Standalone PDF → GazetteDocument JSON extraction.

No database access, no embeddings. Pure: PDF in, JSON file out.

Extraction is single-pass: the PDF is processed once, the result is
validated, and any remaining ERROR-level issues are annotated in the
JSON as extraction_warnings for downstream consumers.  Per-act retries
happen inside pipeline.py (each individual failing act is retried up to
extraction_llm.max_retries times before the gazette finishes).
End-of-batch recovery for still-flagged JSONs is handled by the
fallback-merge pass in cli.py / fallback_merge.py.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legalro_core.config import Settings

DEFAULT_EXTRACTED_DIR = "db/extracted"

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
      2. Extract PDF → GazetteDocument JSON (per-act retries happen inside pipeline.py).
      3. Validate the JSON for ERROR-level issues.
      4. Append any remaining issues to extraction_warnings in the JSON.
         End-of-batch fallback merge handles them from the source PDF.

    Returns the path to the written JSON file.
    """
    from legalro_processing.extract.gazette_extractor import extract_gazette, save_gazette, load_gazette
    from legalro_processing.extraction_validator import validate_file, Severity

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

    # ── Extract + validate (single pass — per-act retries happen inside pipeline) ─
    # The Option C pipeline (pipeline.py) retries each individual act up to
    # extraction_llm.max_retries times before returning.  Re-running the entire
    # gazette here would repeat all N acts just to fix a subset, which is expensive
    # and ineffective.  We extract once, validate, annotate any remaining errors,
    # and let the end-of-batch fallback merge handle them from the source PDF.
    gazette = extract_gazette(path, settings)
    json_path = save_gazette(gazette, out_dir)

    issues = validate_file(json_path)
    errors = [i for i in issues if i.severity == Severity.ERROR]

    if errors:
        error_summary = "; ".join(
            f"act[{i.act_index}] {i.check}: {i.message}" for i in errors
        )
        _log(f"[validate] {path.name} — {len(errors)} error(s) remain after "
             f"single-pass extraction; annotating JSON: {error_summary}")
        _annotate_remaining_issues(json_path, errors)
    else:
        _log(f"[validate] {path.name} — clean")

    return json_path


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
