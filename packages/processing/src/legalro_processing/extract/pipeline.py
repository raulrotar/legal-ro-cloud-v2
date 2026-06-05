"""Option C extraction pipeline: PDF → Docling MD → LLM → GazetteDocument.

Orchestrates the full Option C flow:
  1. md_extractor.extract_markdown()  — PDF → full Markdown (Docling/LlamaParse/Mistral)
  2. md_cache.save/load()             — sha256-keyed on-disk Markdown cache
  3. md_segmenter.segment_gazette_md()— full MD → per-act MdActBlock list
  4. llm_structurer.structure_act()   — per MdActBlock → metadata + corrected full_text
  5. Assemble into GazetteDocument    — same schema as the regex pipeline

Falls back to the standard regex pipeline (gazette_extractor.extract_gazette with
extraction_llm.enabled=False) on any fatal error, so ingestion is never blocked.

Entry point: run(pdf_path, settings, gazette_context) → GazetteDocument
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from legalro_core.act_number import NO_NUMBER_DOC_TYPES

if TYPE_CHECKING:
    from legalro_core.config import Settings
    from legalro_processing.extract.gazette_schema import GazetteDocument


def _log(gazette_name: str, step: str, msg: str = "", t0: float | None = None) -> float:
    """Emit a timestamped progress line to stderr. Returns current time."""
    now = time.time()
    elapsed = f" +{now - t0:.1f}s" if t0 is not None else ""
    prefix = f"[pipeline] {gazette_name} | {step}{elapsed}"
    print(f"{prefix}{': ' + msg if msg else ''}", file=sys.stderr, flush=True)
    return now


def run(
    pdf_path: str | Path,
    settings: "Settings",
    *,
    gazette_year: int,
    issue_number: int,
    gazette_id: str,
    era,
    sumar_entries: list,
    pdf_page_count: int,
    sha256: str,
    issue_date: str,
    part: str,
    is_bis: bool,
    year_label,
    weekday,
    sumar_raw: str,
    warnings: list[str],
) -> "GazetteDocument":
    """Run the Option C pipeline and return a GazetteDocument.

    All parameters mirror what gazette_extractor.extract_gazette() has already
    computed (identity, sumar, era) so this function is a drop-in replacement
    for the act-extraction segment of that function.
    """
    from legalro_processing.extract import md_cache, md_extractor, md_segmenter, llm_structurer, md_rule_extractor
    from legalro_processing.extract.gazette_schema import GazetteDocument, LegalAct
    from legalro_processing.extract.gazette_extractor import _build_act
    from legalro_processing.extract.segment import RawAct
    from datetime import datetime, timezone

    ecfg = settings.extraction_llm
    md_dir = ecfg.md_cache_dir if ecfg else "db/md_cache"
    edit_thr = ecfg.edit_distance_threshold if ecfg else 0.15
    _gname = Path(pdf_path).stem  # e.g. MO_PI_820_2007-12-03

    _t = _log(_gname, "start")

    # ── Step 1+2: get Markdown (from cache or fresh extraction) ───────────────
    cached_md = md_cache.load(pdf_path, md_dir)
    if cached_md is not None:
        full_md = cached_md
        _t = _log(_gname, "md_cache", "hit — skipped Docling", _t)
        warnings.append("md_cache: hit — skipped Docling/OCR")
    else:
        _log(_gname, "md_cache", "miss — starting Docling")
        try:
            full_md = md_extractor.extract_markdown(str(pdf_path), era, settings)
            md_cache.save(pdf_path, full_md, era.value, md_dir)
            _t = _log(_gname, "md_cache", f"Docling done, {len(full_md):,} chars", _t)
            warnings.append("md_cache: miss — extracted fresh Markdown")
        except Exception as exc:
            _log(_gname, "md_cache", f"FAILED: {exc}")
            warnings.append(f"md_extractor failed: {exc}; falling back to regex pipeline")
            return _regex_fallback(
                pdf_path, settings, gazette_year=gazette_year, issue_number=issue_number,
                gazette_id=gazette_id, era=era, sumar_entries=sumar_entries,
                pdf_page_count=pdf_page_count, sha256=sha256, issue_date=issue_date,
                part=part, is_bis=is_bis, year_label=year_label, weekday=weekday,
                sumar_raw=sumar_raw, warnings=warnings,
            )

    # ── Step 2.5: normalize Markdown before segmentation ─────────────────────
    full_md = _normalize_gazette_md(full_md, era)
    _t = _log(_gname, "normalize", "done", _t)

    # ── Step 2.6: enrich Markdown with fitz-recovered closing blocks ──────────
    from legalro_processing.extract.secondary_analyzer import FitzAnalyzer, enrich_markdown_with_fitz
    _log(_gname, "fitz_enrich", "scanning PDF text layer")
    try:
        _enrich_sigs = FitzAnalyzer().recover_closing_numbers(pdf_path)
        _t = _log(_gname, "fitz_enrich", f"{len(_enrich_sigs)} sigs recovered", _t)
        if _enrich_sigs:
            full_md, _n_injected = enrich_markdown_with_fitz(full_md, _enrich_sigs)
            _t = _log(_gname, "fitz_enrich", f"injected {_n_injected} closing block(s)", _t)
            if _n_injected:
                warnings.append(
                    f"md_enrichment: injected {_n_injected} missing closing block(s) "
                    f"from PDF text layer before segmentation"
                )
    except Exception as _exc:
        _log(_gname, "fitz_enrich", f"SKIPPED: {_exc}")
        warnings.append(f"md_enrichment: skipped ({_exc})")

    # ── Step 3: segment ────────────────────────────────────────────────────────
    _log(_gname, "segment", "starting")
    blocks = md_segmenter.segment_gazette_md(
        full_md,
        expected_act_count=len(sumar_entries),
        era=era,
    )
    _t = _log(_gname, "segment", f"{len(blocks)} blocks", _t)

    if not blocks:
        warnings.append("md_segmenter: no blocks found; falling back to regex pipeline")
        return _regex_fallback(
            pdf_path, settings, gazette_year=gazette_year, issue_number=issue_number,
            gazette_id=gazette_id, era=era, sumar_entries=sumar_entries,
            pdf_page_count=pdf_page_count, sha256=sha256, issue_date=issue_date,
            part=part, is_bis=is_bis, year_label=year_label, weekday=weekday,
            sumar_raw=sumar_raw, warnings=warnings,
        )

    # Reconciliation warning
    produced_n = len(blocks)
    expected_n = len(sumar_entries)
    if expected_n >= 2:
        if produced_n < expected_n // 2:
            warnings.append(f"option-c under-segmentation: sumar={expected_n}, produced={produced_n}")
        elif produced_n > expected_n * 3:
            warnings.append(f"option-c over-segmentation: sumar={expected_n}, produced={produced_n}")

    # ── Step 3.5: secondary analyzer — recover closing signatures from PDF ────
    from legalro_processing.extract.secondary_analyzer import FitzAnalyzer, match_recovered_number, find_candidates
    _recovered_sigs = []
    try:
        _recovered_sigs = FitzAnalyzer().recover_closing_numbers(pdf_path)
        if _recovered_sigs:
            warnings.append(
                f"secondary_analyzer: recovered {len(_recovered_sigs)} closing sig(s) from PDF text layer"
            )
    except Exception as _exc:
        warnings.append(f"secondary_analyzer: skipped ({_exc})")

    # Per-gazette signatory counter for the positional tiebreaker.
    # Key = normalised signatory hint string; value = how many times we have
    # already assigned a sig to an act with that signatory in THIS gazette run.
    # Resets for every gazette (local variable) — no cross-gazette contamination.
    _signatory_assign_count: dict[str, int] = {}

    # ── Step 4: LLM structuring per act ───────────────────────────────────────
    _log(_gname, "llm_loop", f"starting {len(blocks)} acts")
    _act_max_retries = _act_retry_limit(settings)
    acts: list[LegalAct] = []
    for act_idx, block in enumerate(blocks):
        _t_act = time.time()
        _log(_gname, f"act[{act_idx}/{len(blocks)-1}]", "rule_draft")
        # Carry a sumar title hint if available
        if act_idx < len(sumar_entries):
            block.title_hint = block.title_hint or getattr(sumar_entries[act_idx], "title", "")

        # Stage 1: deterministic rule-based draft (full-block scan)
        rule_draft = md_rule_extractor.extract_rule_draft(block, gazette_year)

        # Stage 1.5: backfill act_number from secondary analyzer when draft is bad.
        # Only fires when: number is 0/low-confidence OR matches an abrogation number.
        # Write-once-into-empty: high-confidence Docling numbers are never overridden.
        _needs_recovery = (
            rule_draft.act_number == "0"
            or rule_draft.act_number_confidence == "low"
            or (
                rule_draft.act_number
                and rule_draft.abrogation_numbers
                and rule_draft.act_number.replace(".", "") in
                    {n.split("/")[0] for n in rule_draft.abrogation_numbers}
            )
        )
        if _needs_recovery and _recovered_sigs:
            _sig_hint = _extract_signatory_hint(block.plain_text)
            _recovered_nr = match_recovered_number(
                _recovered_sigs,
                signatory_hint=_sig_hint,
                page_hints=block.page_hints,
                abrogation_numbers=rule_draft.abrogation_numbers,
            )

            # Positional tiebreaker: when multiple sigs share the same signatory
            # (e.g. two consecutive ANCEX ordins both signed by Munteanu),
            # match_recovered_number returns None to avoid guessing.  Here we
            # fall back to positional assignment: "Nth act with this signatory
            # gets the Nth sig sorted by page."
            # The counter (_signatory_assign_count) is LOCAL to this gazette run
            # and resets for every call to run() — no cross-gazette side effects.
            if _recovered_nr is None and _sig_hint:
                _candidates = find_candidates(
                    _recovered_sigs,
                    signatory_hint=_sig_hint,
                    page_hints=block.page_hints,
                    abrogation_numbers=rule_draft.abrogation_numbers,
                )
                if len(_candidates) > 1:
                    _nth = _signatory_assign_count.get(_sig_hint, 0)
                    if _nth < len(_candidates):
                        _recovered_nr = _candidates[_nth].number
                        _signatory_assign_count[_sig_hint] = _nth + 1

            if _recovered_nr:
                _prev_nr = rule_draft.act_number
                rule_draft.act_number = _recovered_nr
                # HIGH confidence: fitz recovered this from the PDF text layer with a
                # confirmed all-tokens signatory match. The override guard in
                # llm_structurer.py:388 must fire to prevent the LLM from substituting
                # an abrogation-clause number (e.g. 275→356).
                rule_draft.act_number_confidence = "high"
                rule_draft.warnings.append(
                    f"act_number recovered from PDF text layer by secondary_analyzer "
                    f"(was: {_prev_nr!r} before recovery, signatory: {_sig_hint!r})"
                )

        # Stage 1.6: sumar-table positional fallback — only fires when act_number
        # is still "0" after Stage 1.5 recovery (e.g. mojibake-corrupted closing
        # block that fitz also could not recover).  Uses positional alignment
        # (act_idx ↔ sumar_entries[act_idx]) with a doc_type sanity check.
        # Write-once-into-empty: never overrides a real number.
        if rule_draft.act_number == "0" and act_idx < len(sumar_entries):
            _sumar = sumar_entries[act_idx]
            _sumar_nr = str(getattr(_sumar, "act_number", "") or "").strip()
            _sumar_dt = str(getattr(_sumar, "doc_type", "") or "").strip().upper()
            # Accept when: sumar has a non-zero number AND doc_type matches or
            # either side is UNKNOWN (be lenient when draft doc_type is still unresolved).
            _dt_ok = (
                not _sumar_dt
                or not rule_draft.doc_type
                or rule_draft.doc_type == "UNKNOWN"
                or _sumar_dt == rule_draft.doc_type
            )
            if _sumar_nr and _sumar_nr not in ("0", "") and _dt_ok:
                rule_draft.act_number = _sumar_nr
                rule_draft.act_number_confidence = "high"
                rule_draft.warnings.append(
                    f"act_number filled from sumar table (positional fallback, "
                    f"sumar[{act_idx}]: {_sumar_nr!r}, sumar_doc_type={_sumar_dt!r})"
                )

        # Stage 2: LLM verifies and corrects the draft (with per-act retry on bad result)
        _act_settings = settings
        meta = full_text = _via = _llm_warnings = None  # type: ignore[assignment]
        for _attempt in range(1 + _act_max_retries):
            _log(_gname, f"act[{act_idx}/{len(blocks)-1}]",
                 f"LLM call attempt {_attempt + 1} (draft nr={rule_draft.act_number!r} type={rule_draft.doc_type})")
            meta = llm_structurer.structure_act(
                block,
                gazette_year=gazette_year,
                settings=_act_settings,
                edit_distance_threshold=edit_thr,
                rule_draft=rule_draft,
            )
            full_text = meta.pop("full_text_corrected", None) or block.plain_text
            _via = meta.pop("_via", "unknown")
            _llm_warnings = meta.pop("extraction_warnings", [])
            _llm_warnings.insert(0, f"_via:{_via}")

            _act_nr = str(meta.get("act_number") or "0")
            _act_dt = str(meta.get("doc_type") or "UNKNOWN")
            # Skip "missing number" retry for doc types that legitimately have
            # no own number (COMUNICAT, RECTIFICARE, ANUNT, ANUNȚ).  Still
            # retry when doc_type itself is UNKNOWN.
            _no_number_ok = _act_dt in NO_NUMBER_DOC_TYPES
            _act_is_bad = (_act_nr in ("0", "") and not _no_number_ok) or _act_dt == "UNKNOWN"
            if not _act_is_bad or _attempt >= _act_max_retries:
                if _attempt > 0 and not _act_is_bad:
                    _llm_warnings.append(f"act_number recovered on retry {_attempt + 1}")
                break
            _log(_gname, f"act[{act_idx}/{len(blocks)-1}]",
                 f"bad result (nr={_act_nr!r} type={_act_dt!r}), retrying with fallback model")
            _act_settings = _make_act_fallback_settings(settings, _attempt)

        meta["_gazette_issue_number"] = issue_number

        _log(_gname, f"act[{act_idx}/{len(blocks)-1}]",
             f"done nr={meta.get('act_number')!r} via={_via.split('+')[-1]} "
             f"+{time.time()-_t_act:.1f}s")

        # Build LegalAct using the existing _build_act helper
        # (handles article/annex parsing, signatories, etc.)
        raw_for_build = RawAct(
            text=full_text,
            title=meta.get("title", ""),
            page_range=block.page_hints[:2] if block.page_hints else [],
            position_in_gazette=act_idx,
        )
        act = _build_act(act_idx, raw_for_build.text, meta, raw_for_build.page_range, gazette_year)
        act.extraction_warnings.extend(_llm_warnings)
        acts.append(act)

    # ── Phantom dedup: drop empty segmentation artefacts that survived LLM ──────
    # A phantom is an act with no identity (number=0/unknown, type=UNKNOWN),
    # no authority, no articles, and a very short body — it is a segmentation
    # fragment, not a real act.  Must run BEFORE Annex propagation so a phantom
    # cannot serve as the "previous act" inheritance source.
    _before_dedup = len(acts)
    acts = [a for a in acts if not _is_phantom_act(a)]
    if len(acts) < _before_dedup:
        _log(_gname, "phantom_dedup",
             f"dropped {_before_dedup - len(acts)} phantom act(s)")

    # ── Annex propagation (same as regex pipeline) ────────────────────────────
    for i in range(1, len(acts)):
        a = acts[i]
        if a.doc_type == "UNKNOWN" and a.full_text.strip()[:10].upper().startswith("ANEX"):
            parent = acts[i - 1]
            if parent.doc_type != "UNKNOWN":
                a.doc_type = parent.doc_type
            if not a.issuing_authority:
                a.issuing_authority = parent.issuing_authority
            if (not a.act_number or a.act_number == "0") and parent.act_number:
                a.act_number = parent.act_number
                a.act_year = parent.act_year

    return GazetteDocument(
        filename=Path(pdf_path).name,
        gazette_id=gazette_id,
        part=part,
        issue_number=issue_number,
        is_bis=is_bis,
        issue_year=gazette_year,
        issue_date=issue_date,
        era=era.value,
        year_label=year_label,
        weekday=weekday,
        pdf_page_count=pdf_page_count,
        sha256=sha256,
        sumar=sumar_entries,
        sumar_raw=sumar_raw,
        acts=acts,
        extraction_version="2.0.0-option-c",
        extracted_at=datetime.now(timezone.utc).isoformat(),
        extraction_warnings=warnings,
    )


def _is_phantom_act(act) -> bool:
    """Return True for acts that are segmentation artefacts, not real acts.

    All five conditions must hold to avoid dropping legitimate unnumbered acts:
    - act_number is "0", empty, or absent (no identity)
    - doc_type is UNKNOWN (no recognised act type)
    - issuing_authority is empty (no identified issuer)
    - articles list is empty (no article structure parsed)
    - full_text is very short (< 200 whitespace-tokens)

    A COMUNICAT that is legitimately unnumbered still has an authority or a real
    body, so it will NOT be flagged as phantom.
    """
    act_nr = getattr(act, "act_number", None) or ""
    return (
        (act_nr in ("0", "") or act_nr is None)
        and getattr(act, "doc_type", "UNKNOWN") == "UNKNOWN"
        and not getattr(act, "issuing_authority", "")
        and not getattr(act, "articles", [])
        and len((getattr(act, "full_text", "") or "").split()) < 200
    )


def _act_retry_limit(settings) -> int:
    """Return the per-act LLM retry limit from extraction_llm config (default 2)."""
    ecfg = getattr(settings, "extraction_llm", None)
    return getattr(ecfg, "max_retries", 2) if ecfg else 2


def _make_act_fallback_settings(settings, attempt: int):
    """Return settings with the fallback LLM model for per-act retries."""
    ecfg = getattr(settings, "extraction_llm", None)
    if not ecfg:
        return settings
    fallback_model = getattr(ecfg, "fallback_model", "") or ""
    if not fallback_model:
        return settings
    try:
        import copy
        s = copy.deepcopy(settings)
        object.__setattr__(s.extraction_llm, "model", fallback_model)
        fb_url = getattr(ecfg, "fallback_base_url", "") or ""
        if fb_url:
            object.__setattr__(s.extraction_llm, "base_url", fb_url)
        fb_tokens = getattr(ecfg, "fallback_max_tokens", 0) or 0
        if fb_tokens:
            object.__setattr__(s.extraction_llm, "max_tokens", fb_tokens)
        return s
    except Exception:
        return settings


def _normalize_gazette_md(md: str, era=None) -> str:
    """Clean up Docling Markdown before segmentation and LLM extraction.

    Applied after md_cache load so the cache stores the raw Docling output
    (human-inspectable) while the LLM sees the cleaned version.

    Normalizations:
    1. Strip legalro cache header comments (<!--legalro:...-->)
    2. Era-aware mojibake repair: applies the full normalize_text table for
       broken_2007/broken_2002 eras (handles „→ă ∫→ș ˛→ț ‚→â Ó→î etc.).
       Falls back to the universal ş/ţ→ș/ț fixes for modern eras.
    3. Collapse multiple internal spaces in body lines (PDF word-spacing artifacts)
    4. Recover split closing blocks — when Docling places a lone "Nr. N." line
       BEFORE the institution heading that starts the next act, it belongs to the
       act ENDING there, not the one starting.  We leave it in place (the segmenter
       keeps it in the preceding block) but ensure no double-counting occurs.
    5. Annotate orphaned signatures — minister signatures without a preceding
       date+Nr. block get a comment tag so the rule extractor can warn the LLM.
    """
    import re

    # 1. Strip cache header comments
    md = re.sub(r'<!--legalro:[^>]+-->\n?', '', md)

    # 2. Era-aware mojibake repair.
    # For broken_2007/broken_2002 eras the full normalization table covers all
    # Mac-Roman/Quark artifacts (e.g. „→ă, ∫→ș, ˛→ț, ‚→â, Ó→î, Œ→Î, √→Ă …).
    # For modern eras only the universal ş/ţ cedilla fixes apply.
    if era is not None:
        from legalro_core.normalize import normalize_text as _normalize_text
        md = _normalize_text(md, era)
    else:
        # Legacy call-site without era: apply universal fixes only
        md = md.replace('ş', 'ș').replace('Ş', 'Ș')
        md = md.replace('ţ', 'ț').replace('Ţ', 'Ț')
    # Extra \x8x byte repairs (seen in some LlamaParse outputs)
    md = md.replace('\x82', 'ă').replace('\x92', 'ș').replace('\x93', 'ț')

    # 3. Collapse multiple internal spaces in body lines (not in headings/tables)
    lines = md.splitlines()
    result = []
    for line in lines:
        if line.startswith('#') or line.startswith('|') or line.startswith('  ') or not line.strip():
            result.append(line)
        else:
            result.append(re.sub(r'(?<=\S)  +(?=\S)', ' ', line))
    md = '\n'.join(result)

    # 4+5. Detect orphaned minister signatures:
    # Pattern: a signature attribution line ("Ministrul X, Prenume Nume") that is
    # NOT preceded by a "București, DATE." within the previous 300 chars.
    # This happens when Docling misses the date+Nr. block for long acts.
    # We insert an HTML comment that the rule extractor can detect via the warning.
    _MINISTER_SIG = re.compile(
        r'(?m)^((?:Ministrul|Ministr(?:ul|a)|Președintele|Directorul\s+general'
        r'|Guvernatorul|p\.\s+Ministrul|Secretarul\s+de\s+stat)[^\n]{0,120},)\s*$'
    )
    _BUCURESTI_DATE = re.compile(r'Bucure[șs]ti,\s+\d{1,2}\s+\w+\s+\d{4}')

    def _annotate_orphaned(m: re.Match) -> str:
        # Check whether a București date appears in the 300 chars before this signature
        before = md[:m.start()][-300:]
        if not _BUCURESTI_DATE.search(before):
            return m.group(0) + "\n<!-- legalro:orphaned-signature -->"
        return m.group(0)

    md = _MINISTER_SIG.sub(_annotate_orphaned, md)

    return md


def _extract_signatory_hint(plain_text: str) -> str:
    """Extract a signatory name fragment for secondary-analyzer context matching.

    Looks for a signature attribution line near the end of the act text and
    returns the name portion (e.g. "Cristian David" from "Ministrul internelor,
    Cristian David"). The name may appear on the immediately next line or after
    one blank line (both patterns appear in Docling output for long acts).
    Falls back to empty string if none found.
    """
    import re
    # Allow 0 or 1 blank lines between the title line and the person's name
    _SIG_ATTR = re.compile(
        r'(?:Ministrul|Ministr(?:ul|a)|Președintele|Directorul(?:\s+general)?'
        r'|Guvernatorul|p\.\s+Ministrul|Secretarul\s+de\s+stat)[^\n]{0,120},\s*\n'
        r'(?:\s*\n)?'          # optional blank line
        r'([A-ZĂÂÎȘȚ][^\n]{2,60})',  # name starts with uppercase
        re.MULTILINE,
    )
    # Strip the orphaned-signature HTML comment injected by _normalize_gazette_md —
    # it sits between the attribution line and the person's name and breaks the pattern.
    plain_text = plain_text.replace("<!-- legalro:orphaned-signature -->", "")

    # Search the full plain_text — for acts with long annexes, the signatory
    # line appears BEFORE the annex body, so tail truncation would miss it.
    # Use the FIRST match (act signatory precedes any annex content).
    matches = list(_SIG_ATTR.finditer(plain_text))
    if matches:
        name = matches[0].group(1).strip().rstrip(",;.")
        # Cap at 3 words — the captured line may include trailing section headers
        # like "ANEXĂ" that don't appear in the PDF sig context window.
        words = name.split()
        # Require at least 2 words (first + last name) — a single word or a
        # word ending with punctuation is a malformed capture, not a real name.
        if len(words) < 2:
            return ""
        return " ".join(words[:3])
    return ""


def _regex_fallback(pdf_path, settings, **kwargs) -> "GazetteDocument":
    """Fall back to the standard regex-based extraction pipeline."""
    from legalro_processing.extract.gazette_extractor import extract_gazette

    # Temporarily disable extraction_llm to avoid recursion
    orig_enabled = settings.extraction_llm.enabled if hasattr(settings, "extraction_llm") else False
    if hasattr(settings, "extraction_llm"):
        object.__setattr__(settings.extraction_llm, "enabled", False)
    try:
        result = extract_gazette(pdf_path, settings)
    finally:
        if hasattr(settings, "extraction_llm"):
            object.__setattr__(settings.extraction_llm, "enabled", orig_enabled)
    return result
