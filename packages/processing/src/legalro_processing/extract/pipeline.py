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

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legalro_core.config import Settings
    from legalro_processing.extract.gazette_schema import GazetteDocument


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
    from legalro_processing.extract import md_cache, md_extractor, md_segmenter, llm_structurer
    from legalro_processing.extract.gazette_schema import GazetteDocument, LegalAct
    from legalro_processing.extract.gazette_extractor import _build_act
    from legalro_processing.extract.metadata import extract_metadata
    from legalro_processing.extract.segment import RawAct
    from datetime import datetime, timezone

    ecfg = settings.extraction_llm
    md_dir = ecfg.md_cache_dir if ecfg else "md_cache"
    edit_thr = ecfg.edit_distance_threshold if ecfg else 0.15

    # ── Step 1+2: get Markdown (from cache or fresh extraction) ───────────────
    cached_md = md_cache.load(pdf_path, md_dir)
    if cached_md is not None:
        full_md = cached_md
        warnings.append("md_cache: hit — skipped Docling/OCR")
    else:
        try:
            full_md = md_extractor.extract_markdown(str(pdf_path), era, settings)
            md_cache.save(pdf_path, full_md, era.value, md_dir)
            warnings.append("md_cache: miss — extracted fresh Markdown")
        except Exception as exc:
            warnings.append(f"md_extractor failed: {exc}; falling back to regex pipeline")
            return _regex_fallback(
                pdf_path, settings, gazette_year=gazette_year, issue_number=issue_number,
                gazette_id=gazette_id, era=era, sumar_entries=sumar_entries,
                pdf_page_count=pdf_page_count, sha256=sha256, issue_date=issue_date,
                part=part, is_bis=is_bis, year_label=year_label, weekday=weekday,
                sumar_raw=sumar_raw, warnings=warnings,
            )

    # ── Step 2.5: normalize Markdown before segmentation ─────────────────────
    full_md = _normalize_gazette_md(full_md)

    # ── Step 3: segment ────────────────────────────────────────────────────────
    blocks = md_segmenter.segment_gazette_md(
        full_md,
        expected_act_count=len(sumar_entries),
    )

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

    # ── Step 4: LLM structuring per act ───────────────────────────────────────
    acts: list[LegalAct] = []
    for act_idx, block in enumerate(blocks):
        # Carry a sumar title hint if available
        if act_idx < len(sumar_entries):
            block.title_hint = block.title_hint or getattr(sumar_entries[act_idx], "title", "")

        meta = llm_structurer.structure_act(
            block,
            gazette_year=gazette_year,
            settings=settings,
            edit_distance_threshold=edit_thr,
        )

        # full_text: use LLM-corrected text if available, else plain text
        full_text = meta.pop("full_text_corrected", None) or block.plain_text
        _via = meta.pop("_via", "unknown")
        _llm_warnings = meta.pop("extraction_warnings", [])
        _llm_warnings.insert(0, f"_via:{_via}")

        meta["_gazette_issue_number"] = issue_number

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


def _normalize_gazette_md(md: str) -> str:
    """Clean up Docling Markdown before segmentation and LLM extraction.

    Applied after md_cache load so the cache stores the raw Docling output
    (human-inspectable) while the LLM sees the cleaned version.

    Normalizations:
    1. Strip legalro cache header comments (<!--legalro:...-->)
    2. Collapse multiple internal spaces in body lines (PDF word-spacing artifacts)
    3. Fix common broken diacritics: ş→ș, ţ→ț (broken_2007/2002 encoding)
    4. Strip SUMAR table block (large TOC that wastes tokens; segmenter skips it anyway)
    """
    import re

    # 1. Strip cache header comments
    md = re.sub(r'<!--legalro:[^>]+-->\n?', '', md)

    # 2. Fix broken diacritics (cedilla variants → comma-below, correct Romanian)
    md = md.replace('ş', 'ș').replace('Ş', 'Ș')
    md = md.replace('ţ', 'ț').replace('Ţ', 'Ț')
    # Common OCR mojibake for ă
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

    return md


def _regex_fallback(pdf_path, settings, **kwargs) -> "GazetteDocument":
    """Fall back to the standard regex-based extraction pipeline."""
    from legalro_processing.extract.gazette_extractor import extract_gazette
    from legalro_core.config import Settings as _S

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
