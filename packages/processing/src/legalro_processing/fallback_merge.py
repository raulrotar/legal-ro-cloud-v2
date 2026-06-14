"""End-of-batch fallback re-extraction and field-level merge.

After the main extraction loop finishes, any JSON that still carries
VALIDATE:* error tags is re-extracted from the original PDF using the
regex pipeline (extraction_llm disabled).  The two results are then
merged field-by-field at the act level: whichever side has a better
value for each field wins.  Provenance is recorded in extraction_warnings.

Merge rules (per act field):
  act_number      — prefer non-"0" and non-suspicious; primary wins on tie
  doc_type        — prefer non-"UNKNOWN"; primary wins on tie
  issuing_authority — prefer non-empty; primary wins on tie
  title           — prefer non-empty and no dot-leader "......"; primary wins on tie
  full_text       — prefer the longer text (LLM-corrected > raw regex); primary
                    text kept unless it is very short (<120 chars) and fallback
                    is substantially longer

Act matching between primary and fallback uses act_index (positional).
When the fallback has more acts, extra acts are appended with a provenance tag.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legalro_core.config import Settings

# Numbers that are clearly wrong as act_numbers
_BAD_NUMBERS = {"0", "1", "2", "3"}
_SUSPICIOUS_YEARS = {str(y) for y in range(2000, 2030)}
_DOT_LEADER = "......"


# ── Public API ────────────────────────────────────────────────────────────────

def collect_flagged(extracted_dir: Path) -> list[Path]:
    """Return paths of JSONs that carry VALIDATE:* error tags."""
    flagged = []
    for json_path in sorted(extracted_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        warnings = data.get("extraction_warnings", [])
        if any(w.startswith("[VALIDATE:") for w in warnings):
            flagged.append(json_path)
    return flagged


def run_fallback_merge(
    json_path: Path,
    pdf_path: Path,
    settings: "Settings",
) -> Path:
    """Re-extract pdf_path with the regex pipeline and merge into json_path.

    Overwrites json_path with the merged result and returns it.
    """
    from legalro_processing.extract.gazette_extractor import extract_gazette, save_gazette
    import dataclasses

    _log(f"[fallback] {pdf_path.name} — re-extracting with regex pipeline")

    # Run regex pipeline (LLM disabled)
    import copy
    fallback_settings = copy.deepcopy(settings)
    if hasattr(fallback_settings, "extraction_llm"):
        object.__setattr__(fallback_settings.extraction_llm, "enabled", False)

    try:
        fallback_gazette = extract_gazette(pdf_path, fallback_settings)
    except Exception as exc:
        _log(f"[fallback] {pdf_path.name} — regex pipeline failed: {exc}; keeping primary")
        return json_path

    fallback_data = dataclasses.asdict(fallback_gazette)

    # Load primary JSON
    primary_data = json.loads(json_path.read_text(encoding="utf-8"))

    merged = _merge_gazette(primary_data, fallback_data, pdf_path.name)

    # Reconciliation ran before this merge phase, so acts appended here can
    # resolve MISSING-sumar warnings retroactively (e.g. a closing-block
    # split act recovered by the regex pipeline).
    for note in _resolve_missing_after_merge(merged):
        _log(f"[fallback] {pdf_path.name} — {note}")

    json_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"[fallback] {pdf_path.name} — merge written to {json_path}")
    return json_path


_MISSING_WARNING = re.compile(
    r"sumar_reconcile: MISSING act — sumar\[(\d+)\] "
    r"(\S+) nr='([^']*)' p\.(\S+) '(.*)' has no matching"
)


def _resolve_missing_after_merge(merged: dict) -> list[str]:
    """Resolve MISSING-sumar warnings satisfied by acts the merge appended.

    Matches by folded act number (wildcard-tolerant on doc_type), prefers the
    longest body among duplicates, and backfills doc_type/title from the sumar
    entry recorded in the warning text.
    """
    from legalro_core.act_number import fold_act_number

    warnings = merged.get("extraction_warnings") or []
    kept: list[str] = []
    notes: list[str] = []
    for w in warnings:
        m = _MISSING_WARNING.search(w)
        if not m:
            kept.append(w)
            continue
        sj, stype, snr, _page, stitle = m.groups()
        folded = fold_act_number(snr)
        cands = [
            a for a in merged.get("acts", [])
            if folded and fold_act_number(a.get("act_number")) == folded
            and (a.get("doc_type") in ("UNKNOWN", "", None) or a.get("doc_type") == stype)
        ]
        if not cands:
            kept.append(w)
            continue
        best = max(cands, key=lambda a: len(a.get("full_text") or ""))
        if best.get("doc_type") in ("UNKNOWN", "", None) and stype not in ("?", "ACT", ""):
            best["doc_type"] = stype
        if not best.get("title"):
            best["title"] = stitle
        notes.append(
            f"fallback_merge: resolved MISSING sumar[{sj}] {stype} nr={snr!r} "
            f"→ merged act nr={best.get('act_number')!r}"
        )
    if notes:
        merged["extraction_warnings"] = kept + notes
    return notes


def find_source_pdf(json_path: Path, laws_dir: Path) -> Path | None:
    """Map db/extracted/YYYY/MM/DD/stem.json → laws/YYYY/MM/DD/stem.pdf."""
    # Try structural mirror first
    for part in json_path.parts:
        if part.isdigit() and len(part) == 4:  # year segment
            try:
                idx = list(json_path.parts).index(part)
                rel = Path(*json_path.parts[idx:]).with_suffix(".pdf")
                candidate = laws_dir / rel
                if candidate.exists():
                    return candidate
            except Exception:
                pass
    # Fallback: search by filename
    matches = list(laws_dir.rglob(f"{json_path.stem}.pdf"))
    return matches[0] if matches else None


# ── Merge logic ───────────────────────────────────────────────────────────────

def _merge_gazette(primary: dict, fallback: dict, filename: str) -> dict:
    merged = dict(primary)
    p_acts = primary.get("acts", [])
    f_acts = fallback.get("acts", [])

    # Sumar numbers — the authority for validating fallback number adoption.
    # The fallback aligns acts POSITIONALLY by index, so its number can belong
    # to a neighbouring act when the two pipelines segmented differently;
    # adopting it unchecked plants chimeras after all in-pipeline safeguards.
    # Scanned era excluded: its OCR sumar is partial, so absence from it is
    # not evidence — an empty set disables the guard below.
    sumar_nrs: set[str] = set()
    if merged.get("era") != "scanned":
        sumar_nrs = {
            re.sub(r"\D", "", str(e.get("act_number") or "").split("/")[0])
            for e in merged.get("sumar", [])
        } - {""}

    merged_acts = []
    for i, p_act in enumerate(p_acts):
        f_act = f_acts[i] if i < len(f_acts) else None
        merged_acts.append(_merge_act(p_act, f_act, i, sumar_nrs))

    # Append extra acts from fallback that primary missed entirely.
    # Skip extras whose act number the primary already has — those are the
    # regex pipeline's over-segmentation duplicates, not missing acts; the
    # primary's sumar reconciliation is the authority on completeness.
    def _nr_key(a: dict) -> str:
        return re.sub(r"\D", "", str(a.get("act_number") or ""))
    primary_nrs = {_nr_key(a) for a in p_acts if _nr_key(a)}
    for i in range(len(p_acts), len(f_acts)):
        if _nr_key(f_acts[i]) and _nr_key(f_acts[i]) in primary_nrs:
            continue
        extra = dict(f_acts[i])
        extra_warns = list(extra.get("extraction_warnings", []))
        extra_warns.append(
            "_source:fallback-only — act present in regex pipeline but missing from primary"
        )
        extra_nr = str(extra.get("act_number") or "0")
        if _is_bad_number(extra_nr):
            extra_warns.append(
                f"[VALIDATE:ACT_NUMBER_ZERO] act_number is {extra_nr!r} — "
                "could not recover number for fallback-only act"
            )
        extra["extraction_warnings"] = extra_warns
        merged_acts.append(extra)
        _log(f"[fallback] {filename} — appended extra act[{i}] from fallback")

    merged["acts"] = merged_acts

    # Record merge provenance on the gazette level
    warnings = merged.get("extraction_warnings", [])
    warnings.append(
        f"[MERGE] fallback regex pipeline run; "
        f"primary={len(p_acts)} acts, fallback={len(f_acts)} acts, "
        f"merged={len(merged_acts)} acts"
    )
    merged["extraction_warnings"] = warnings
    return merged


def _merge_act(primary: dict, fallback: dict | None, idx: int,
               sumar_nrs: set[str] | None = None) -> dict:
    """Merge one act dict, preferring the better value per field."""
    if fallback is None:
        return primary

    result = dict(primary)
    changes = []

    # act_number — adopt the fallback's positional number only when the
    # gazette's sumar confirms it exists (or no sumar is available to check);
    # an unconfirmed number is recorded as a warning instead of adopted.
    p_nr = primary.get("act_number", "0") or "0"
    f_nr = fallback.get("act_number", "0") or "0"
    if _is_bad_number(p_nr) and not _is_bad_number(f_nr):
        f_key = re.sub(r"\D", "", str(f_nr).split("/")[0])
        if not sumar_nrs or f_key in sumar_nrs:
            result["act_number"] = f_nr
            # Sync act_year from fallback when we take its number
            if fallback.get("act_year"):
                result["act_year"] = fallback["act_year"]
            changes.append(f"act_number: {p_nr!r}→{f_nr!r}")
        else:
            warns = list(result.get("extraction_warnings", []))
            warns.append(
                f"[MERGE] fallback act_number {f_nr!r} NOT adopted — "
                f"absent from sumar (positional misalignment guard)"
            )
            result["extraction_warnings"] = warns

    # doc_type
    p_dt = primary.get("doc_type", "UNKNOWN")
    f_dt = fallback.get("doc_type", "UNKNOWN")
    if p_dt == "UNKNOWN" and f_dt != "UNKNOWN":
        result["doc_type"] = f_dt
        changes.append(f"doc_type: UNKNOWN→{f_dt!r}")

    # issuing_authority
    p_auth = (primary.get("issuing_authority") or "").strip()
    f_auth = (fallback.get("issuing_authority") or "").strip()
    if not p_auth and f_auth:
        result["issuing_authority"] = f_auth
        changes.append(f"issuing_authority: empty→{f_auth!r}")

    # title
    p_title = (primary.get("title") or "").strip()
    f_title = (fallback.get("title") or "").strip()
    if _is_bad_title(p_title) and not _is_bad_title(f_title):
        result["title"] = f_title
        changes.append(f"title: bad→{f_title[:60]!r}")

    # full_text — only replace if primary is very short and fallback substantially longer
    p_text = primary.get("full_text") or ""
    f_text = fallback.get("full_text") or ""
    if len(p_text) < 120 and len(f_text) > len(p_text) * 2:
        result["full_text"] = f_text
        changes.append(f"full_text: {len(p_text)}→{len(f_text)} chars")

    if changes:
        warnings = list(result.get("extraction_warnings", []))
        warnings.append(f"[MERGE] act[{idx}] fields taken from fallback: {', '.join(changes)}")
        result["extraction_warnings"] = warnings

    return result


def _is_bad_number(nr: str) -> bool:
    nr = nr.strip().lstrip("0") or "0"
    return nr in _BAD_NUMBERS or nr in _SUSPICIOUS_YEARS


_RUNON_TITLE = re.compile(
    r"În temeiul|Av[âî]nd în vedere|decret[ăe]a?z[ăa]|Consiliul Frontului|Art\.\s*\d",
    re.IGNORECASE,
)


def _is_bad_title(title: str) -> bool:
    # run-on titles leak body text (regex segmenter artifact) — never adopt them
    return not title or _DOT_LEADER in title or len(title) > 220 or bool(_RUNON_TITLE.search(title))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
