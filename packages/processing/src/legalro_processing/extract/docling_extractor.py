"""
Docling-based PDF extraction provider.

Output cleaning note
--------------------
Docling exports content as Markdown (## headings, GFM pipe tables). The
existing sumar parser, article splitter, and metadata extractor were all
written for plain text from PyMuPDF/ocrmac — they break on `##` prefixes
and `|` table separators. We therefore strip Markdown formatting from every
page before returning, so the rest of the pipeline is unaffected.
Tables become space-joined rows, which is still vastly better than the
interleaved-column output PyMuPDF produces for two-column scanned layouts.

Slot-in replacement for the PyMuPDF/ocrmac extraction in extract.py.
Maintains the same list[str] per-page interface so nothing downstream changes.

Resource discipline:
  - DocumentConverter instances are singletons keyed by era config.
    Models load once on first call and stay resident for the ingestion session.
  - All non-essential docling features (picture description, chart extraction,
    code/formula enrichment) are disabled to minimise RAM and CPU.
  - TableFormer (table structure) is enabled only for MODERN era where PDFs
    contain born-digital tables worth reconstructing (e.g. 294Bis Nomenclator).
  - The LLM server and docling never run at the same time: docling is used
    during Phase 1 (PDF→JSON cache); the LLM runs during Phase 2 (query).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from legalro_core.models import Era

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

# Singleton cache: one converter per (era, ocr_provider) pair.
# Keys are strings like "modern:ocrmac" or "scanned:ocrmac".
_converter_cache: dict[str, "DocumentConverter"] = {}


def extract_pages_docling(
    pdf_path: str,
    era: Era,
    ocr_provider: str = "ocrmac",
) -> list[str]:
    """
    Extract text per page from a PDF using docling.

    Returns list[str] with one entry per PDF page (1-indexed pages mapped to
    0-indexed list), preserving the same interface as extract.py so all
    downstream code (normalize, sumar, segment) is unchanged.

    Tables are exported as Markdown pipe tables, which downstream chunking
    handles gracefully as plain text.
    """
    converter = _get_converter(era, ocr_provider)

    result = converter.convert(str(pdf_path))
    doc = result.document

    # docling page numbers are 1-indexed; build a 0-indexed list
    n_pages = doc.num_pages()
    if n_pages == 0:
        return [""]

    pages: list[str] = []
    for page_no in range(1, n_pages + 1):
        # export_to_markdown preserves reading order from DocLayNet layout
        # detection (critical for two-column 1989 layouts and table structure).
        # We then strip markdown formatting so downstream plain-text parsers
        # (sumar, segmenter, metadata) are unaffected.
        raw = doc.export_to_markdown(
            page_no=page_no,
            escape_html=False,
            escape_underscores=False,
            image_placeholder="",
        )
        pages.append(_strip_markdown(raw))

    return pages


def _strip_markdown(text: str) -> str:
    """
    Remove Markdown formatting added by docling, yielding plain text.

    Rules:
    - Heading markers (## / #) are stripped; the heading text is kept.
    - GFM table separator rows (|---|---) are dropped.
    - GFM table data rows (| cell | cell |) are joined with two spaces.
    - All other lines pass through unchanged.
    """
    out: list[str] = []
    _SEP_ROW = re.compile(r'^\s*\|[\s|:\-]+\|\s*$')
    _DATA_ROW = re.compile(r'^\s*\|')

    for line in text.splitlines():
        # Strip heading markers
        clean = re.sub(r'^#{1,6}\s+', '', line)

        if _SEP_ROW.match(clean):
            continue  # drop separator rows entirely

        if _DATA_ROW.match(clean):
            # Extract non-empty cell contents
            cells = [c.strip() for c in clean.split('|') if c.strip()]
            out.append('  '.join(cells))
        else:
            out.append(clean)

    return '\n'.join(out)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cache_key(era: Era, ocr_provider: str) -> str:
    return f"{era.value}:{ocr_provider}"


def _get_converter(era: Era, ocr_provider: str) -> "DocumentConverter":
    """Return cached converter, building it on first call for this era."""
    key = _cache_key(era, ocr_provider)
    if key not in _converter_cache:
        _converter_cache[key] = _build_converter(era, ocr_provider)
    return _converter_cache[key]


def _build_converter(era: Era, ocr_provider: str) -> "DocumentConverter":
    """Build and return a configured DocumentConverter for this era."""
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
        from docling.datamodel.accelerator_options import AcceleratorDevice
        from docling.datamodel.base_models import InputFormat
    except ImportError as exc:
        raise ImportError(
            "docling is not installed. Run: uv pip install 'docling>=2.0.0'"
        ) from exc

    opts = PdfPipelineOptions()

    # Force CPU: MPS (Apple Silicon GPU) doesn't support float64 which the
    # layout model (RT-DETRv2) requires. CPU inference is fast enough for
    # legal PDFs and avoids competing with the MLX LLM for GPU memory.
    opts.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU,
    )

    # ── Disable all non-essential features to save RAM / CPU ─────────────
    opts.do_picture_classification = False
    opts.do_picture_description = False
    opts.do_chart_extraction = False
    opts.do_code_enrichment = False
    opts.do_formula_enrichment = False
    opts.generate_page_images = False
    opts.generate_picture_images = False
    opts.generate_table_images = False

    # ── Era-specific OCR and table config ─────────────────────────────────
    if era == Era.MODERN:
        # Born-digital PDFs: no OCR needed, but enable TableFormer for
        # structured table reconstruction (e.g. 146-page Nomenclator).
        opts.do_ocr = False
        opts.do_table_structure = True
    elif era in (Era.SCANNED, Era.HYBRID, Era.BROKEN_2002, Era.BROKEN_2007):
        # Scanned / broken-font pages need OCR.
        opts.do_ocr = True
        opts.do_table_structure = False  # table structure unreliable on OCR'd pages
        opts.ocr_options = _ocr_options(ocr_provider)
    else:
        # Safe default for unknown eras
        opts.do_ocr = True
        opts.do_table_structure = False
        opts.ocr_options = _ocr_options(ocr_provider)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
        }
    )


def _ocr_options(ocr_provider: str):
    """Return the correct OcrOptions object for the configured provider."""
    # OcrAutoOptions lets docling choose the best portable backend available
    # (EasyOCR or RapidOCR on Linux, Tesseract as fallback).
    # OcrMacOptions (Apple Vision) is intentionally not supported here — it is a
    # macOS-only dependency that must not run in the cloud container.
    from docling.datamodel.pipeline_options import OcrAutoOptions
    return OcrAutoOptions()
