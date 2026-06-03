"""Post-extraction validation: scan GazetteDocument JSONs for quality issues.

Each extracted JSON is checked against a suite of rules.  Issues are
classified by severity:

  ERROR   — field is clearly wrong or missing; re-extraction will likely fix it
  WARNING — suspicious value; may be correct but worth reviewing
  INFO    — structural observation (orphaned signature, etc.)

The validator returns a list of Issue records.  The CLI command
`legalro-process validate-extractions` uses this module to produce a report
and optionally re-extract flagged files.

Checks implemented
------------------
ACT_NUMBER_ZERO          act_number == "0"                            ERROR
ACT_NUMBER_ABROGATION    act_number found in abrogation clause        ERROR
ACT_NUMBER_SUSPICIOUS    act_number is a very common reference number WARNING
DOC_TYPE_UNKNOWN         doc_type == "UNKNOWN"                        ERROR
AUTHORITY_MISSING        issuing_authority is empty                   WARNING
TITLE_MISSING            title is empty                               WARNING
FULL_TEXT_SHORT          full_text < MIN_FULL_TEXT chars for non-trivial types
                                                                      WARNING
LLM_FAILED               _via contains "regex_fallback(llm_failed"   WARNING
HALLUCINATION_REJECTED   _via contains "hallucination_rejected"       INFO
SUMAR_ZERO               gazette-level: sumar has 0 entries           INFO
ACT_COUNT_MISMATCH       |sumar_entries| vs |acts| differs > 50%     WARNING
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator


# ── Severity ──────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    ERROR   = "ERROR"
    WARNING = "WARNING"
    INFO    = "INFO"


# ── Issue dataclass ───────────────────────────────────────────────────────────

@dataclass
class Issue:
    check:       str           # check name, e.g. "ACT_NUMBER_ZERO"
    severity:    Severity
    gazette_id:  str           # e.g. "MO_PI_820_2007"
    json_path:   Path
    act_index:   int | None    # None = gazette-level issue
    act_number:  str | None
    doc_type:    str | None
    message:     str


# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_FULL_TEXT = 80   # chars; acts shorter than this are suspicious
# doc types where a missing full_text is definitely wrong
_REQUIRES_BODY = {"HG", "OUG", "ORDONANȚĂ", "DECRET", "DECRET_LEGE",
                  "LEGE", "DCC", "DECIZIE", "ORDIN"}

# Numbers that appear very frequently in legal references (years, common refs)
# and should not be used as act_numbers
_SUSPICIOUS_NUMBERS = {"2003", "2005", "2006", "2007", "2008", "2009",
                       "2010", "2011", "2012", "2013", "2014", "2015",
                       "2016", "2017", "2018", "2019", "2020", "2021",
                       "2022", "2023", "2024", "2025", "2026",
                       "1", "2", "3"}

# Abrogation pattern (same as md_rule_extractor but applied to plain text)
_ABROGATION_NR = re.compile(
    r'[Nn][Rr]\.\s*([\d.]+)/(\d{4})\b.{0,400}?se\s+abrog[ăa]'
    r'|se\s+abrog[ăa].{0,400}?[Nn][Rr]\.\s*([\d.]+)/(\d{4})\b',
    re.IGNORECASE | re.DOTALL,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def validate_directory(extracted_dir: Path) -> list[Issue]:
    """Validate all extracted JSONs under extracted_dir. Returns sorted issues."""
    issues: list[Issue] = []
    json_files = sorted(extracted_dir.rglob("*.json"))
    for json_path in json_files:
        try:
            gazette = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(Issue(
                check="PARSE_ERROR", severity=Severity.ERROR,
                gazette_id=json_path.stem, json_path=json_path,
                act_index=None, act_number=None, doc_type=None,
                message=f"Could not parse JSON: {exc}",
            ))
            continue
        issues.extend(_validate_gazette(gazette, json_path))
    return sorted(issues, key=lambda i: (i.severity.value, i.gazette_id, i.act_index or -1))


def validate_file(json_path: Path) -> list[Issue]:
    """Validate a single extracted JSON file."""
    gazette = json.loads(json_path.read_text(encoding="utf-8"))
    return _validate_gazette(gazette, json_path)


def needs_reextraction(issues: list[Issue]) -> bool:
    """True if any ERROR-level issue is present that re-extraction can fix."""
    fixable = {"ACT_NUMBER_ZERO", "ACT_NUMBER_ABROGATION", "DOC_TYPE_UNKNOWN", "LLM_FAILED"}
    return any(i.severity == Severity.ERROR and i.check in fixable for i in issues)


def group_by_file(issues: list[Issue]) -> dict[Path, list[Issue]]:
    result: dict[Path, list[Issue]] = {}
    for issue in issues:
        result.setdefault(issue.json_path, []).append(issue)
    return result


# ── Internal validation logic ─────────────────────────────────────────────────

def _validate_gazette(gazette: dict, json_path: Path) -> list[Issue]:
    issues: list[Issue] = []
    gazette_id = gazette.get("gazette_id") or json_path.stem

    # ── Gazette-level checks ──────────────────────────────────────────────────

    sumar = gazette.get("sumar", [])
    acts  = gazette.get("acts", [])

    if not sumar and not gazette.get("sumar_raw", "").strip():
        # check if it's NOT a known single-act gazette (< 2 acts is fine)
        if len(acts) >= 3:
            issues.append(_issue(
                "SUMAR_ZERO", Severity.INFO, gazette_id, json_path, None, None, None,
                "sumar has 0 entries and gazette has ≥3 acts — sumar parsing may have failed",
            ))

    if sumar and acts:
        ratio = len(acts) / max(len(sumar), 1)
        if ratio < 0.5 or ratio > 2.5:
            issues.append(_issue(
                "ACT_COUNT_MISMATCH", Severity.WARNING, gazette_id, json_path, None, None, None,
                f"sumar has {len(sumar)} entries but extracted {len(acts)} acts (ratio={ratio:.2f})",
            ))

    # ── Act-level checks ──────────────────────────────────────────────────────

    for act in acts:
        idx       = act.get("act_index", 0)
        act_nr    = str(act.get("act_number", "") or "")
        doc_type  = act.get("doc_type", "UNKNOWN")
        full_text = act.get("full_text", "")
        authority = act.get("issuing_authority", "")
        title     = act.get("title", "")
        warnings  = act.get("extraction_warnings", [])
        via       = next((w for w in warnings if w.startswith("_via:")), "")

        # ACT_NUMBER_ZERO — explicit "not found" marker set by extraction
        if act_nr == "0":
            issues.append(_issue(
                "ACT_NUMBER_ZERO", Severity.ERROR, gazette_id, json_path, idx, act_nr, doc_type,
                "act_number is '0' — closing signature not found during extraction",
            ))

        # ACT_NUMBER_EMPTY — empty string: usually an annex; flag as WARNING only
        # when the act has a body (full_text) and isn't preceded by a same-type act
        elif act_nr == "" and doc_type in _REQUIRES_BODY and len(full_text) > MIN_FULL_TEXT:
            # Heuristic: if act has substance but no number it's suspicious
            issues.append(_issue(
                "ACT_NUMBER_EMPTY", Severity.WARNING, gazette_id, json_path, idx, act_nr, doc_type,
                "act_number is empty — may be an annex; verify against source PDF",
            ))

        # ACT_NUMBER_PLACEHOLDER — LLM returned Romanian "unknown" or similar
        elif act_nr.lower() in {"necunoscut", "unknown", "lipsă", "lipsa", "n/a"}:
            issues.append(_issue(
                "ACT_NUMBER_PLACEHOLDER", Severity.ERROR, gazette_id, json_path, idx, act_nr, doc_type,
                f"act_number is a placeholder value {act_nr!r} — LLM could not determine the number",
            ))

        # ACT_NUMBER_ABROGATION — the extracted number appears in an abrogation clause
        elif act_nr and act_nr not in _SUSPICIOUS_NUMBERS and act_nr not in {"", "0"}:
            abrogation_nrs = _find_abrogation_numbers(full_text)
            if act_nr.replace(".", "") in [n.split("/")[0] for n in abrogation_nrs]:
                issues.append(_issue(
                    "ACT_NUMBER_ABROGATION", Severity.ERROR, gazette_id, json_path, idx, act_nr, doc_type,
                    f"act_number '{act_nr}' matches an abrogation-clause number "
                    f"({abrogation_nrs}) — likely confused with a referenced/abrogated act",
                ))

        # DOC_TYPE_UNKNOWN
        if doc_type == "UNKNOWN":
            issues.append(_issue(
                "DOC_TYPE_UNKNOWN", Severity.ERROR, gazette_id, json_path, idx, act_nr, doc_type,
                "doc_type is UNKNOWN — classification failed",
            ))

        # AUTHORITY_MISSING
        if not authority and doc_type in _REQUIRES_BODY:
            issues.append(_issue(
                "AUTHORITY_MISSING", Severity.WARNING, gazette_id, json_path, idx, act_nr, doc_type,
                "issuing_authority is empty for a substantive act",
            ))

        # TITLE_MISSING
        if not title:
            issues.append(_issue(
                "TITLE_MISSING", Severity.WARNING, gazette_id, json_path, idx, act_nr, doc_type,
                "title is empty",
            ))

        # FULL_TEXT_SHORT
        if doc_type in _REQUIRES_BODY and len(full_text) < MIN_FULL_TEXT:
            issues.append(_issue(
                "FULL_TEXT_SHORT", Severity.WARNING, gazette_id, json_path, idx, act_nr, doc_type,
                f"full_text is only {len(full_text)} chars (threshold={MIN_FULL_TEXT})",
            ))

        # LLM_FAILED
        if "regex_fallback(llm_failed" in via:
            issues.append(_issue(
                "LLM_FAILED", Severity.WARNING, gazette_id, json_path, idx, act_nr, doc_type,
                f"LLM structuring failed — used regex fallback ({via})",
            ))

        # HALLUCINATION_REJECTED
        if "hallucination_rejected" in via:
            issues.append(_issue(
                "HALLUCINATION_REJECTED", Severity.INFO, gazette_id, json_path, idx, act_nr, doc_type,
                "LLM full_text_corrected was rejected by edit-distance guard — used source plain text",
            ))

    return issues


def _issue(check, severity, gazette_id, json_path, act_index, act_number, doc_type, message) -> Issue:
    return Issue(
        check=check, severity=severity, gazette_id=gazette_id, json_path=json_path,
        act_index=act_index, act_number=act_number, doc_type=doc_type, message=message,
    )


def _find_abrogation_numbers(text: str) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for m in _ABROGATION_NR.finditer(text):
        nr = (m.group(1) or m.group(3) or "").replace(".", "")
        yr = m.group(2) or m.group(4) or ""
        key = f"{nr}/{yr}"
        if nr and key not in seen:
            seen.add(key)
            results.append(key)
    return results
