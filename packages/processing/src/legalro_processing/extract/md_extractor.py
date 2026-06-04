"""Docling → full Markdown extractor (Option C).

Unlike docling_extractor.py which strips Markdown back to plain text,
this module preserves the full structured Markdown output from Docling:
  - ## headings  → act/section boundaries
  - | tables |   → structured table data (cotizatii, nomenclator, etc.)
  - Correct reading order (two-column layouts, scanned pages)

The Markdown is saved to md_cache/ and used as input to md_segmenter.py
and llm_structurer.py. It is also human-inspectable and verifiable against
the original PDF.

Provider routing:
  - MODERN/HYBRID   → Docling (no OCR, TableFormer on)
  - SCANNED/BROKEN  → Docling with OCR enabled (EasyOCR/RapidOCR on Linux,
                       TesseractOCR fallback), OR LlamaParse/Mistral MD passthrough
                       (their output is already Markdown — just skip md_normalize)
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

    # ── Docling (local, all eras) ─────────────────────────────────────────────
    return _extract_docling_md(path, era)


def _provider(settings: "Settings | None") -> str:
    if settings is None:
        return "docling"
    return getattr(settings.ocr, "provider", "docling")


def _extract_docling_md(pdf_path: str, era: Era) -> str:
    """Run Docling and return the full document Markdown without any stripping."""
    _t0 = time.time()
    _name = Path(pdf_path).stem
    print(f"[docling] {_name} | start (era={era.value})", file=sys.stderr, flush=True)

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
    else:
        # SCANNED / BROKEN: full OCR, DocLayNet layout for reading order.
        opts.do_ocr = True
        opts.do_table_structure = False
        opts.ocr_options = _auto_ocr_options()

    print(f"[docling] {_name} | converter init done +{time.time()-_t0:.1f}s", file=sys.stderr, flush=True)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    print(f"[docling] {_name} | convert() starting", file=sys.stderr, flush=True)
    result = converter.convert(pdf_path)
    print(f"[docling] {_name} | convert() done +{time.time()-_t0:.1f}s", file=sys.stderr, flush=True)
    # Export before releasing the converter — keeps full_text in a plain string.
    markdown = result.document.export_to_markdown(
        escape_html=False,
        escape_underscores=False,
        image_placeholder="",
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


def _auto_ocr_options():
    """Portable OCR backend — works on both Linux VPS and macOS without ocrmac."""
    from docling.datamodel.pipeline_options import OcrAutoOptions
    return OcrAutoOptions()


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
