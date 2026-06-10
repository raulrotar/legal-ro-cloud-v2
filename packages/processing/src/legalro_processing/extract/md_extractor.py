"""Docling / GLM-OCR → full Markdown extractor (Option C).

Unlike docling_extractor.py which strips Markdown back to plain text,
this module preserves the full structured Markdown output from Docling:
  - ## headings  → act/section boundaries
  - | tables |   → structured table data (cotizatii, nomenclator, etc.)
  - Correct reading order (two-column layouts, scanned pages)

The Markdown is saved to md_cache/ and used as input to md_segmenter.py
and the repair pass. It is also human-inspectable and verifiable against
the original PDF.

Provider routing:
  - SCANNED         → GLM-OCR (vision, page-by-page) when ocr.scanned_provider="glm-ocr"
                       (correct diacritics, ~13 s/page, no 3 GB OCR model in RAM)
  - MODERN/HYBRID   → Docling (no OCR, TableFormer on)
  - BROKEN_*        → Docling with OCR disabled (embedded text has recoverable broken-CMap mojibake)
  - LlamaParse/Mistral passthrough for cloud OCR (already Markdown — skip md_normalize)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from legalro_core.models import Era

if TYPE_CHECKING:
    from legalro_core.config import Settings


def extract_markdown(pdf_path: str | Path, era: Era, settings: "Settings | None" = None) -> str:
    """Extract a gazette PDF to a single Markdown string.

    Returns the full document as one Markdown string (not split by page).
    Headings, tables, and reading order are preserved.
    """
    path = str(pdf_path)
    provider = _provider(settings)

    # ── Cloud OCR providers (LlamaParse / Mistral) ────────────────────────────
    # These already return Markdown — skip md_normalize to preserve structure.
    if provider in ("llamaparse", "mistral") and era in (
        Era.SCANNED, Era.BROKEN_2007, Era.BROKEN_2002
    ):
        return _extract_cloud_md(path, era, settings)

    # ── GLM-OCR for SCANNED era ───────────────────────────────────────────────
    # Side-by-side testing showed GLM-OCR extracts correct Romanian diacritics
    # (ț/ș/ă/î), 14-23% more content, and runs ~14x faster than Docling RapidOCR
    # on 1989 scanned gazettes.  Docling's OCR models also compete for RAM with
    # the vision repair LLM; GLM-OCR (2.2 GB) stays well clear of that budget.
    scanned_provider = getattr(getattr(settings, "ocr", None), "scanned_provider", "glm-ocr") if settings else "glm-ocr"
    if era == Era.SCANNED and scanned_provider == "glm-ocr":
        return _extract_glm_ocr_md(path, era, settings)

    # ── Docling (local, all other eras) ──────────────────────────────────────
    md = _extract_docling_md(path, era)

    # ── Hybrid router: escalate Docling failures to GLM-OCR ─────────────────
    # Docling's layout stage drops/merges column content on multi-act pages
    # (architectural — persists with layout-heron).  When the completeness
    # check finds >=2 acts missing or merged, re-extract the whole document
    # through the verified GLM-OCR tiled path (OmniDocBench v1.5 #1; reads
    # every detected region, so closing blocks and column twins survive).
    # Escalation threshold: small defect counts stay on the (free) text-layer
    # recovery path — md_act_recovery + fitz_enrich already fix isolated
    # missing closings/bodies.  Whole-doc GLM-OCR (~9 s/page) is only worth it
    # when Docling broke a substantial share of the document.
    _n_fail = _docling_failure_count(md, path, era)
    if _n_fail >= 6:
        print(f"[hybrid-router] {Path(path).stem}: Docling output incomplete — "
              f"escalating to GLM-OCR structured pass", file=sys.stderr, flush=True)
        try:
            glm_md = _extract_glm_ocr_md(path, era, settings, structured=True)
            # Adoption metric: missing CLOSING NUMBERS only.  The unique-tail
            # check compares verbatim against the PDF text layer, which only
            # a text-layer copier (Docling) can match exactly — OCR-derived
            # text would always "lose" on it even when more complete.
            glm_fail = _docling_failure_count(glm_md, path, era, count_tails=False)
            doc_fail = _docling_failure_count(md, path, era, count_tails=False)
            # Bloat guard: repetition residue can triple the output volume
            # (MO_PI_2_2007: 431K chars vs a 141K text layer).  A bloated MD
            # mints phantom acts downstream — completeness is not worth that.
            _bloated = len(glm_md) > 1.5 * max(len(md), 10_000)
            if glm_fail < doc_fail and not _bloated:
                print(f"[hybrid-router] {Path(path).stem}: GLM-OCR adopted "
                      f"(missing closings {doc_fail} → {glm_fail})", file=sys.stderr, flush=True)
                # marker tells text-layer-verbatim checks (recovery tails)
                # that this MD is OCR-derived and won't match exactly
                return "<!-- legalro:md-source=glm-ocr -->\n" + glm_md
            print(f"[hybrid-router] {Path(path).stem}: GLM-OCR not better "
                  f"({glm_fail} vs {doc_fail} missing closings) — keeping Docling",
                  file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[hybrid-router] {Path(path).stem}: GLM-OCR escalation failed ({exc})",
                  file=sys.stderr, flush=True)
    return md


def _docling_failure_count(md: str, pdf_path: str, era: Era, count_tails: bool = True) -> int:
    """Count acts whose closing number or unique body tail is absent from the
    MD — the page-router's escalation signal (same logic the recovery pass
    uses, but counting instead of patching)."""
    try:
        import fitz
        from legalro_core.normalize import normalize_text
        from legalro_processing.extract.md_act_recovery import (
            _closing_numbers, _fold_ws,
        )
        import re as _re

        doc = fitz.open(str(pdf_path))
        pdf_text = "\n".join(p.get_text() for p in doc)
        doc.close()
        if len(pdf_text.strip()) < 500:
            return 0
        pdf_text = normalize_text(pdf_text, era)
        closings = _closing_numbers(pdf_text)
        md_digits = {d for d, _, _ in _closing_numbers(md)}
        md_folded = _fold_ws(md)
        ends = [e for _, _, e in closings]
        failures = 0
        for d, s, e in closings:
            prev_end = max((pe for pe in ends if pe < s), default=0)
            span = pdf_text[prev_end:e]
            if not (120 <= len(span) <= 25_000):
                continue
            if d not in md_digits:
                failures += 1
                continue
            if not count_tails:
                continue
            parts = _re.split(
                r"PRE[ȘS]EDINTELE\s+ROM[ÂA]NIEI|Bucure[șs]ti,\s*\d{1,2}\s+\w+\s+\d{4}",
                span,
            )
            tail = _fold_ws(max(parts, key=len))[-120:]
            if len(tail) >= 60 and tail not in md_folded:
                failures += 1
        return failures
    except Exception:
        return 0


def _provider(settings: "Settings | None") -> str:
    if settings is None:
        return "docling"
    return getattr(settings.ocr, "provider", "docling")


_OCR_PROMPT = (
    "Transcrie exact textul din această imagine dintr-un document oficial "
    "românesc. Păstrează diacriticile corecte (ț, ș, ă, î, â). Transcrie "
    "fiecare rând o singură dată, nu repeta nimic. "
    "Returnează doar textul, fără explicații."
)

# Structured mode (hybrid-router escalation for born-digital pages): asks for
# Markdown so headings survive for the act segmenter. Left column first —
# gazette text flows down columns.
_OCR_PROMPT_STRUCTURED = (
    "Convertește această pagină dintr-un document oficial românesc în Markdown. "
    "Citește coloana din stânga complet, apoi coloana din dreapta. "
    "Folosește ## pentru titlurile de acte (DECRET, LEGE, HOTĂRÂRE, ORDIN, DECIZIE) "
    "și pentru instituții (PREȘEDINTELE ROMÂNIEI, GUVERNUL ROMÂNIEI). "
    "Păstrează diacriticile corecte (ț, ș, ă, î, â). Transcrie fiecare rând o "
    "singură dată, nu repeta nimic. Returnează doar Markdown, fără explicații."
)

# Retry temperatures: 0 first (deterministic), one escalation to break
# repetition loops (olmOCR's recipe).  A third attempt almost never converges
# and cost up to ~10 min/page; the duplicate-block collapse + dedup handle
# the residue instead.
_RETRY_TEMPS = (0.0, 0.4)


def _extract_glm_ocr_md(pdf_path: str, era: Era, settings: "Settings | None" = None,
                        structured: bool = False) -> str:
    """OCR a scanned PDF with a local vision LLM, tile-by-tile with verification.

    The Ollama glm-ocr port clips images >2048px (ollama#14114) and aborts
    generation on token-repeat (ollama#14117) — both cause SILENT page loss.
    Defenses, in order:
      1. tile each page below the size limit along whitespace cuts
         (page_tiles.tile_page), preserving column reading order;
      2. per-tile repetition detection + retry with escalating temperature;
      3. per-page Tesseract oracle gate (ocr_verify.verify_page): a page whose
         output covers too little of what Tesseract saw is re-OCR'd once and
         flagged in a .verify.json sidecar if it still fails.
    """
    import base64
    import fitz  # pymupdf

    try:
        import ollama as _ollama
    except ImportError as exc:
        raise ImportError("ollama Python SDK not installed. Run: uv add ollama") from exc

    from legalro_processing.extract import ocr_verify
    from legalro_processing.extract.page_tiles import tile_page, pixmap_to_gray, crop_png

    _name = Path(pdf_path).stem
    _cfg = getattr(settings, "ocr", None)
    _model = getattr(_cfg, "glm_model", "glm-ocr:latest") if _cfg else "glm-ocr:latest"
    _dpi   = int(getattr(_cfg, "glm_dpi", 200)) if _cfg else 200

    print(f"[glm-ocr] {_name} | start (era={era.value}, model={_model}, dpi={_dpi}, tiled)", file=sys.stderr, flush=True)
    _t0 = time.time()

    client = _ollama.Client()

    _prompt = _OCR_PROMPT_STRUCTURED if structured else _OCR_PROMPT

    def _ocr_image(png_bytes: bytes, temperature: float) -> str:
        resp = client.chat(
            model=_model,
            messages=[{
                "role": "user",
                "content": _prompt,
                "images": [base64.b64encode(png_bytes).decode()],
            }],
            options={"temperature": temperature, "num_ctx": 16384},
        )
        return (resp.message.content or "").strip()

    def _ocr_tile_with_retry(png_bytes: bytes) -> tuple[str, bool]:
        """OCR one tile; retry on repetition loop. Returns (text, clean)."""
        best = ""
        for temp in _RETRY_TEMPS:
            text = _ocr_image(png_bytes, temp)
            if ocr_verify.repeated_shingle(text) is None:
                return text, True
            if len(text) > len(best):
                best = text
        return best, False

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(_dpi / 72, _dpi / 72)
    pages_md: list[str] = []
    verifications: list[dict] = []

    for i, page in enumerate(doc):
        _pt = time.time()
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        gray = pixmap_to_gray(pix)
        full_png = pix.tobytes("png")

        def _ocr_whole_page(split_columns: bool = False) -> str:
            tiles = tile_page(gray, split_columns=split_columns)
            parts: list[str] = []
            for t in tiles:
                text, clean = _ocr_tile_with_retry(crop_png(gray, t))
                if not clean:
                    print(f"[glm-ocr] {_name} | page {i+1} tile {t} — repetition persisted",
                          file=sys.stderr, flush=True)
                if text:
                    parts.append(text)
            # collapse adjacent duplicate line blocks (loop residue survives
            # retries when copies differ in case/typos)
            return ocr_verify.collapse_repeated_line_blocks("\n\n".join(parts))

        page_text = _ocr_whole_page()

        # ── Oracle gate: Tesseract sees the whole page; we must match it ──
        try:
            oracle_text = ocr_verify.tesseract_page_text(full_png)
        except (FileNotFoundError, RuntimeError) as exc:
            oracle_text = ""
            print(f"[glm-ocr] {_name} | page {i+1} — oracle unavailable ({exc}); gate skipped",
                  file=sys.stderr, flush=True)

        if oracle_text:
            v = ocr_verify.verify_page(page_text, oracle_text, page_index=i)

            # Escalation 1: re-OCR with column-split tiles — but only for
            # UNDER-delivery (content loss).  Over-emission/repetition pages
            # don't gain from re-OCR (the loop reproduces); collapse + dedup
            # handle them, so skip the expensive second pass.
            if not v.passed and v.word_ratio < 0.75:
                print(f"[glm-ocr] {_name} | page {i+1} FAILED gate ({'; '.join(v.reasons)}) "
                      f"— escalating to column-split OCR", file=sys.stderr, flush=True)
                retry_text = _ocr_whole_page(split_columns=True)
                rv = ocr_verify.verify_page(retry_text, oracle_text, page_index=i)
                if rv.passed or rv.coverage > v.coverage:
                    page_text, v = retry_text, rv

            # Escalation 2: the Tesseract oracle text itself.  Its diacritics
            # are weaker but it never drops half a page; when the VLM is still
            # under-delivering, completeness wins.
            if not v.passed and v.word_ratio < 0.75:
                t_words = len(oracle_text.split())
                if t_words > len(page_text.split()):
                    page_text = (
                        "<!-- legalro:ocr-fallback=tesseract -->\n" + oracle_text.strip()
                    )
                    v = ocr_verify.verify_page(page_text, oracle_text, page_index=i)
                    v.reasons.append("fell back to tesseract oracle text")
                    print(f"[glm-ocr] {_name} | page {i+1} — VLM under-delivered; "
                          f"using tesseract text ({t_words} words)", file=sys.stderr, flush=True)

            verifications.append(v.as_dict())
            if not v.passed:
                print(f"[glm-ocr] {_name} | page {i+1} STILL FAILING gate: {'; '.join(v.reasons)}",
                      file=sys.stderr, flush=True)

        pages_md.append(page_text)
        print(
            f"[glm-ocr] {_name} | page {i+1}/{len(doc)} — {len(page_text)} chars +{time.time()-_pt:.1f}s",
            file=sys.stderr, flush=True,
        )

    doc.close()

    # Sidecar verification report next to the cached MD
    if verifications:
        import json
        from legalro_processing.extract import md_cache as _md_cache
        ecfg = getattr(settings, "extraction_llm", None)
        md_dir = getattr(ecfg, "md_cache_dir", "db/md_cache") if ecfg else "db/md_cache"
        sidecar = _md_cache.cache_path(pdf_path, md_dir).with_suffix(".verify.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(verifications, ensure_ascii=False, indent=1), encoding="utf-8")
        n_fail = sum(1 for v in verifications if not v["passed"])
        print(f"[glm-ocr] {_name} | verification: {len(verifications)-n_fail}/{len(verifications)} pages passed → {sidecar}",
              file=sys.stderr, flush=True)

    markdown = "\n\n<!-- legalro:page-break -->\n\n".join(pages_md)
    print(f"[glm-ocr] {_name} | done {len(markdown):,} chars +{time.time()-_t0:.1f}s", file=sys.stderr, flush=True)
    return markdown


def _extract_docling_md(pdf_path: str, era: Era) -> str:
    """Run Docling and return the full document Markdown without any stripping."""
    _t0 = time.time()
    _name = Path(pdf_path).stem
    print(f"[docling] {_name} | start (era={era.value})", file=sys.stderr, flush=True)

    # BROKEN eras: repair the ToUnicode CMaps first so Docling reads correct
    # diacritics directly.  Docling folds the mojibake quote glyphs („ ‚) to
    # ASCII '/, BEFORE the text-level normalization table can map them to ă/â,
    # which is unrecoverable; fixing the font mapping at the source is exact.
    if era in (Era.BROKEN_2007, Era.BROKEN_2002):
        from legalro_processing.extract.cmap_fix import fix_tounicode
        pdf_path = str(fix_tounicode(pdf_path, era))

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
    except ImportError as exc:
        raise ImportError("docling is not installed. Run: uv pip install 'docling>=2.0.0'") from exc

    opts = PdfPipelineOptions()

    # Force CPU: MPS (Apple Silicon) doesn't support float64 required by the
    # RT-DETRv2 layout model, causing "Stage layout failed" errors. On Linux
    # VPS with CUDA the CUDA path works fine — use AUTO there only if needed.
    from docling.datamodel.pipeline_options import AcceleratorOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice
    opts.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU,
    )

    # ── Resource discipline ───────────────────────────────────────────────────
    opts.do_picture_classification = False
    opts.do_picture_description = False
    opts.do_chart_extraction = False
    opts.do_code_enrichment = False
    opts.do_formula_enrichment = False
    opts.generate_page_images = False
    opts.generate_picture_images = False
    opts.generate_table_images = False

    # ── Era-specific config ───────────────────────────────────────────────────
    if era == Era.MODERN:
        # Born-digital: no OCR, TableFormer for table reconstruction.
        opts.do_ocr = False
        opts.do_table_structure = True
    elif era == Era.HYBRID:
        # Mixed: OCR for facsimile pages, TableFormer off (unreliable on mixed).
        opts.do_ocr = True
        opts.do_table_structure = False
        opts.ocr_options = _auto_ocr_options()
    elif era in (Era.BROKEN_2007, Era.BROKEN_2002):
        # BROKEN: font CMap is corrupt but the embedded text layer is still readable
        # as recoverable mojibake (e.g. '∫' for 'ș', '„' for 'ă').  Docling's text
        # backend returns these glyph codes verbatim; the BROKEN_2007/BROKEN_2002
        # normalization table in normalize.py converts them to correct diacritics.
        # force_full_page_ocr=True discards this lossless source in favour of Tesseract
        # which strips diacritics to ASCII — that regression was confirmed experimentally.
        opts.do_ocr = False
        opts.do_table_structure = False
    else:
        # SCANNED: full OCR, DocLayNet layout for reading order.
        opts.do_ocr = True
        opts.do_table_structure = False
        opts.ocr_options = _auto_ocr_options()

    # ── Timeout safety net ───────────────────────────────────────────────────
    # document_timeout caps runaway TableFormer/OCR on large annex tables.
    # The real cure for truncation is the status-check + recovery pass below;
    # the timeout only prevents an indefinite hang.
    opts.document_timeout = 3600

    print(f"[docling] {_name} | converter init done +{time.time()-_t0:.1f}s", file=sys.stderr, flush=True)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    print(f"[docling] {_name} | convert() starting", file=sys.stderr, flush=True)
    result = converter.convert(pdf_path)
    print(f"[docling] {_name} | convert() done +{time.time()-_t0:.1f}s", file=sys.stderr, flush=True)

    # ── Truncation guard ─────────────────────────────────────────────────────
    # Docling can silently drop page body content on heavy TableFormer/OCR pages,
    # returning PARTIAL_SUCCESS.  Page objects persist in result.document.pages
    # even when body is missing, so we cannot rely on a page-count comparison —
    # status is the reliable signal (Docling #2610, #2857, #3020).
    from docling.datamodel.base_models import ConversionStatus
    if result.status != ConversionStatus.SUCCESS:
        print(
            f"[docling] {_name} | WARNING: status={result.status.name}; "
            f"errors={result.errors} — attempting table-downgrade recovery",
            file=sys.stderr, flush=True,
        )
        result = _recover_truncated(pdf_path, opts, _name, _t0)

    # Export before releasing the converter — keeps full_text in a plain string.
    markdown = result.document.export_to_markdown(
        escape_html=False,
        escape_underscores=False,
        image_placeholder="",
    )

    # Content-completeness backstop: if the last substantive line is a bare
    # section header (e.g. "ANEXA Nr. 3" with nothing below), handle two cases:
    #   1. The same label appears earlier (Docling reading-order defect #1203/#2245:
    #      header serialized *after* its own body) → strip the dangling duplicate.
    #   2. The label appears only once (possible genuine truncation) → warn.
    import re as _re
    last_line = markdown.rstrip().splitlines()[-1] if markdown.strip() else ""
    if _re.match(r'^#{0,3}\s*ANEX[AĂ]\s+[Nn]r', last_line):
        # Normalize label for comparison (strip heading hashes and whitespace).
        bare_label = _re.sub(r'^#+\s*', '', last_line).strip()
        # Count occurrences of this label in the full document (case-insensitive).
        prior_count = len(_re.findall(_re.escape(bare_label), markdown.rstrip()[:-len(last_line)], _re.IGNORECASE))
        if prior_count >= 1:
            # Body is already present earlier — strip the trailing orphan header.
            markdown = markdown.rstrip()[:-len(last_line)].rstrip() + "\n"
            print(
                f"[docling] {_name} | INFO: stripped dangling trailing header "
                f"'{last_line[:60]}' (body already present, Docling #1203)",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                f"[docling] {_name} | WARNING: output ends on bare annex header "
                f"'{last_line[:60]}' — possible truncation",
                file=sys.stderr, flush=True,
            )

    # Explicitly free Docling's MPS/GPU memory before the LLM stage runs.
    # Without this, RT-DETRv2 + TableFormer stay resident alongside the LLM,
    # which can exhaust 24 GB unified memory on Apple Silicon.
    del result
    del converter
    import gc
    gc.collect()
    print(f"[docling] {_name} | gc done, {len(markdown):,} chars +{time.time()-_t0:.1f}s", file=sys.stderr, flush=True)

    return markdown


def _recover_truncated(pdf_path: str, base_opts: "PdfPipelineOptions", name: str, t0: float) -> "ConversionResult":
    """Attempt to recover a PARTIAL_SUCCESS conversion by re-running with
    table structure disabled.  TableFormer on large annex pages is the primary
    culprit; disabling it lets Docling capture the body as raw text cells
    instead of timing out/failing silently.

    Returns the recovery result (may still be PARTIAL_SUCCESS if the file is
    genuinely broken, in which case the caller raises for _regex_fallback).
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        import copy
        recovery_opts = copy.deepcopy(base_opts)
        recovery_opts.do_table_structure = False   # TableFormer off — body as text
        recovery_opts.document_timeout = 3600
        print(
            f"[docling] {name} | recovery: table_structure=False, re-converting "
            f"+{time.time()-t0:.1f}s", file=sys.stderr, flush=True,
        )
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=recovery_opts)}
        )
        result = conv.convert(pdf_path)
        print(
            f"[docling] {name} | recovery done: status={result.status.name} "
            f"+{time.time()-t0:.1f}s", file=sys.stderr, flush=True,
        )
        from docling.datamodel.base_models import ConversionStatus
        if result.status != ConversionStatus.SUCCESS:
            raise RuntimeError(
                f"Docling truncation recovery failed: status={result.status.name}, "
                f"errors={result.errors}"
            )
        return result
    except Exception as exc:
        raise RuntimeError(f"Docling truncation unrecoverable for {name}: {exc}") from exc


def _auto_ocr_options(lang: list[str] | None = None, force_full_page_ocr: bool = False):
    """Portable OCR backend — works on both Linux VPS and macOS without ocrmac.

    Args:
        lang: ISO-639-3 language codes for Tesseract (e.g. ["ron"] for Romanian).
              Empty list lets Tesseract auto-detect.
        force_full_page_ocr: When True, ignore any embedded text layer and re-OCR
              every page from pixels.  Use for PDFs with broken font CMap encoding.
    """
    from docling.datamodel.pipeline_options import OcrAutoOptions
    return OcrAutoOptions(lang=lang or [], force_full_page_ocr=force_full_page_ocr)


def _extract_cloud_md(pdf_path: str, era: Era, settings: "Settings") -> str:
    """Use LlamaParse or Mistral to get Markdown, bypassing md_normalize stripping.

    The raw Markdown from cloud OCR providers is returned as-is so heading
    structure, table syntax, and page-break signals are preserved.
    """
    from legalro_processing.extract.ocr import ocr_pdf

    # ocr_pdf returns list[str] after md_normalize splits and strips.
    # We re-run the raw API calls here to get unprocessed Markdown.
    provider = settings.ocr.provider

    if provider == "llamaparse":
        return _raw_llamaparse(pdf_path, settings)
    elif provider == "mistral":
        return _raw_mistral(pdf_path, settings)

    # Fallback: stitch the normalized pages back (loses structure but safe)
    pages = ocr_pdf(pdf_path, settings)
    return "\n\n---\n\n".join(pages)


def _raw_llamaparse(pdf_path: str, settings) -> str:
    """Fetch raw LlamaParse Markdown without normalization."""
    import time
    import httpx

    api_key = settings.ocr.llama_cloud_api_key
    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = "https://api.cloud.llamaindex.ai/api/parsing"

    with open(pdf_path, "rb") as f:
        resp = httpx.post(
            f"{base_url}/upload",
            headers=headers,
            files={"file": (pdf_path, f, "application/pdf")},
            data={"result_type": "markdown", "language": "ro"},
            timeout=120,
        )
    resp.raise_for_status()
    job_id = resp.json()["id"]

    for _ in range(120):
        time.sleep(2)
        st = httpx.get(f"{base_url}/job/{job_id}", headers=headers, timeout=30)
        st.raise_for_status()
        if st.json().get("status") == "SUCCESS":
            break

    result = httpx.get(f"{base_url}/job/{job_id}/result/markdown", headers=headers, timeout=60)
    result.raise_for_status()
    time.sleep(3)  # throttle
    return result.json().get("markdown", "")


def _raw_mistral(pdf_path: str, settings) -> str:
    """Fetch raw Mistral OCR Markdown without normalization."""
    import base64
    import httpx

    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()

    resp = httpx.post(
        "https://api.mistral.ai/v1/ocr",
        headers={
            "Authorization": f"Bearer {settings.ocr.mistral_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "mistral-ocr-latest",
            "document": {"type": "document_url",
                         "document_url": f"data:application/pdf;base64,{pdf_b64}"},
        },
        timeout=300,
    )
    resp.raise_for_status()
    import time
    time.sleep(30)  # throttle ≤2 req/min

    data = resp.json()
    pages = data.get("pages", [])
    if pages:
        return "\n\n---\n\n".join(p.get("markdown", "") for p in pages)
    return data.get("text", "")
