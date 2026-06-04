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

    # ── Extract + validate loop ───────────────────────────────────────────────
    json_path: Path | None = None
    active_settings = settings

    for attempt in range(1 + MAX_RETRIES):
        gazette = extract_gazette(path, active_settings)
        json_path = save_gazette(gazette, out_dir)

        issues = validate_file(json_path)
        errors = [i for i in issues if i.severity == Severity.ERROR]

        if not errors:
            if attempt > 0:
                _log(f"[validate] {path.name} — clean after {attempt + 1} attempt(s)")
            return json_path

        error_summary = "; ".join(
            f"act[{i.act_index}] {i.check}: {i.message}" for i in errors
        )

        if attempt < MAX_RETRIES:
            json_path.unlink(missing_ok=True)
            # Switch to fallback model on the next attempt if one is configured
            fallback = _make_fallback_settings(settings, attempt)
            if fallback is not active_settings:
                ecfg = getattr(fallback, "extraction_llm", None)
                model = getattr(ecfg, "model", "") or getattr(getattr(fallback, "llm", None), "model", "")
                _log(f"[validate] {path.name} attempt {attempt + 1} — "
                     f"{len(errors)} error(s), retrying with fallback model {model!r}: "
                     f"{error_summary}")
            else:
                _log(f"[validate] {path.name} attempt {attempt + 1} — "
                     f"{len(errors)} error(s), retrying: {error_summary}")
            active_settings = fallback
        else:
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


def _make_fallback_settings(settings: "Settings", attempt: int) -> "Settings":
    """Return a settings object that uses the fallback model for retries.

    On attempt 0 the primary model is used; attempt 1+ switches to the
    fallback model if one is configured.  If no fallback is set, returns
    the same settings object (same model retried, which still helps when
    temperature > 0 or the LLM output is non-deterministic).
    """
    if attempt == 0:
        return settings

    ecfg = getattr(settings, "extraction_llm", None)
    if not ecfg:
        return settings

    fallback_model   = getattr(ecfg, "fallback_model", "")
    fallback_base    = getattr(ecfg, "fallback_base_url", "")
    fallback_key     = getattr(ecfg, "fallback_api_key", "")
    fallback_tokens  = getattr(ecfg, "fallback_max_tokens", 0)

    if not fallback_model:
        # No fallback configured — retry with same model
        return settings

    # Build a shallow copy of settings with the extraction_llm fields patched.
    # Pydantic v2: use model_copy; v1: copy() or manual dict round-trip.
    try:
        new_ecfg = ecfg.model_copy(update={
            "model":      fallback_model,
            "base_url":   fallback_base  or ecfg.base_url,
            "api_key":    fallback_key   or ecfg.api_key,
            "max_tokens": fallback_tokens or ecfg.max_tokens,
        })
        return settings.model_copy(update={"extraction_llm": new_ecfg})
    except AttributeError:
        # Pydantic v1 fallback
        try:
            new_ecfg = ecfg.copy(update={
                "model":      fallback_model,
                "base_url":   fallback_base  or ecfg.base_url,
                "api_key":    fallback_key   or ecfg.api_key,
                "max_tokens": fallback_tokens or ecfg.max_tokens,
            })
            return settings.copy(update={"extraction_llm": new_ecfg})
        except Exception:
            return settings


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
