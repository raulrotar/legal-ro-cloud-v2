"""Phase 2 — fill Table.html from Docling's native TableItem.export_to_html.

The MODERN/born-digital path runs Docling with TableFormer, whose internal
``TableData`` carries per-cell ``row_span``/``col_span``.  ``export_to_markdown``
(in md_extractor) flattens that structure to pipe tables; ``export_to_html``
preserves it as ``<table>`` with ``colspan``/``rowspan``.

Rather than change md_extractor's ``-> str`` contract (the highest-risk module),
this module re-runs Docling once (MODERN only, opt-in via
``extraction.html_tables_docling``) and matches each native HTML table onto the
already-triaged ``Table`` objects **by content**, filling ``.html`` /
``.text_flat`` / ``.n_cols``.  Search/embedding/coverage keep using the flat
view; only ``chunk.act_full_text`` gets the HTML (same contract as Phase 1).

Note: HTML faithfully reflects TableFormer's structure — well-ruled body tables
get correct colspan/rowspan, but unruled SUMAR/TOC tables inherit TableFormer's
mis-structuring (not fixed here; that needs a different mechanism).
"""
from __future__ import annotations

import html as _htmllib
import re
import sys


def _html_to_flat(html: str) -> str:
    """Strip an HTML table to clean, tab/newline-joined, source-order cell text."""
    s = re.sub(r"</t[dh]>", "\t", html)
    s = re.sub(r"</tr>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = _htmllib.unescape(s)
    lines = []
    for line in s.split("\n"):
        cells = [c.strip() for c in line.split("\t")]
        cells = [c for c in cells if c]
        if cells:
            lines.append("\t".join(cells))
    return "\n".join(lines)


_TOK = re.compile(r"[0-9A-Za-zĂÂÎȘȚăâîșț]{2,}")


def _fingerprint(text: str) -> set[str]:
    return set(_TOK.findall(text.lower()))


def _harvest(pdf_path: str) -> list[dict]:
    """Run Docling (MODERN config) and return per-table {html, flat, n_rows, fp}."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.base_models import InputFormat

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    for k in ("do_picture_classification", "do_picture_description", "do_chart_extraction",
              "do_code_enrichment", "do_formula_enrichment", "generate_page_images",
              "generate_picture_images", "generate_table_images"):
        setattr(opts, k, False)
    opts.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    opts.document_timeout = 3600

    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    doc = conv.convert(pdf_path).document
    out: list[dict] = []
    for tb in doc.tables:
        try:
            html = tb.export_to_html(doc=doc)
        except Exception:
            continue
        flat = _html_to_flat(html)
        n_rows = getattr(getattr(tb, "data", None), "num_rows", 0) or (flat.count("\n") + 1)
        out.append({"html": html, "flat": flat, "n_rows": int(n_rows), "fp": _fingerprint(flat)})
    return out


def fill_table_html(gazette_tables: list, pdf_path: str, *, min_overlap: float = 0.4) -> int:
    """Fill .html/.text_flat/.n_cols on triaged Table objects from Docling HTML.

    Matches each Table (by its .markdown fingerprint) to the best unused Docling
    table (Jaccard ≥ min_overlap).  Skips tables that already carry .html (e.g.
    the fitz-annex path).  Returns the number of tables filled.
    """
    targets = [(i, t) for i, t in enumerate(gazette_tables) if not getattr(t, "html", "")]
    if not targets:
        return 0
    try:
        dtables = _harvest(pdf_path)
    except Exception as exc:  # noqa: BLE001 — never let table HTML break extraction
        print(f"[docling-html] {pdf_path}: harvest failed ({exc}); keeping markdown", file=sys.stderr)
        return 0
    if not dtables:
        return 0

    used: set[int] = set()
    filled = 0
    for _, t in targets:
        tfp = _fingerprint(getattr(t, "markdown", "") or "")
        if not tfp:
            continue
        best_j, best_k = 0.0, -1
        for k, d in enumerate(dtables):
            if k in used or not d["fp"]:
                continue
            inter = len(tfp & d["fp"])
            j = inter / len(tfp | d["fp"])
            if j > best_j:
                best_j, best_k = j, k
        if best_k >= 0 and best_j >= min_overlap:
            d = dtables[best_k]
            used.add(best_k)
            t.html = d["html"]
            t.text_flat = d["flat"]
            ncols = max((row.count("\t") + 1 for row in d["flat"].split("\n") if row), default=0)
            t.n_cols = ncols
            filled += 1
    return filled
