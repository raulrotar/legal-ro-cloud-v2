"""Era-routed text extraction using PyMuPDF, docling, or cloud OCR (Mistral/LlamaParse)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import fitz

from legalro_core.models import Era

if TYPE_CHECKING:
    from legalro_core.config import Settings


def extract_text(pdf_path: str, era: Era, settings: "Settings | None" = None) -> list[str]:
    """
    Extract text per page. Returns list of page texts.

    Provider routing (settings.ocr.provider):
    - "llamaparse" / "mistral" → cloud OCR for SCANNED and BROKEN_2007 eras;
                                  PyMuPDF for MODERN/HYBRID (no benefit from cloud OCR
                                  on born-digital text)
    - "docling"               → layout-aware local extraction for SCANNED/BROKEN eras
    - "ocrmac" / "pymupdf"    → local PyMuPDF + Apple Vision (dev/local only)

    Page layout detection (PyMuPDF path):
    - prose        → get_text("text")
    - two_column   → column-aware reader (left col then right col, top-to-bottom each)
    - table        → word-position reconstruction (preserves cell values)
    """
    provider = _provider(settings)
    _is_cloud = provider in ("mistral", "llamaparse")

    # ── Cloud OCR: SCANNED + BROKEN_2007 always need visual-layer reading ──
    # MODERN/HYBRID have clean text; cloud OCR wastes credits with no benefit.
    if _is_cloud and era in (Era.SCANNED, Era.BROKEN_2007, Era.BROKEN_2002):
        from legalro_processing.extract.ocr import ocr_pdf
        return ocr_pdf(pdf_path, settings)

    # ── Docling: layout-aware local fallback for broken eras ──────────────
    if provider == "docling" and era in (Era.SCANNED, Era.BROKEN_2002, Era.BROKEN_2007):
        from legalro_processing.extract.docling_extractor import extract_pages_docling
        return extract_pages_docling(pdf_path, era, ocr_provider="auto")

    # ── SCANNED with local provider (ocrmac / Apple Vision) ───────────────
    if era == Era.SCANNED:
        return _ocr_all_pages_local(pdf_path)

    # ── HYBRID: cloud or local OCR for facsimile pages ────────────────────
    cloud_pages: list[str] | None = None
    if _is_cloud and era == Era.HYBRID:
        from legalro_processing.extract.ocr import ocr_pdf
        cloud_pages = ocr_pdf(pdf_path, settings)

    # ── PyMuPDF path (MODERN, HYBRID digital pages, BROKEN fallback) ──────
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        if era == Era.HYBRID and _is_facsimile_page(page):
            if cloud_pages is not None and page.number < len(cloud_pages):
                pages.append(cloud_pages[page.number])
            else:
                pages.append(_ocr_single_page_local(pdf_path, page.number))
        else:
            layout = _detect_page_layout(page)
            if layout == "two_column":
                pages.append(_extract_two_column(page))
            elif layout == "table":
                pages.append(_extract_page_words(page))
            else:
                pages.append(page.get_text("text"))
    doc.close()
    return pages


# ── Internal helpers ──────────────────────────────────────────────────────────

def _provider(settings: "Settings | None") -> str:
    if settings is None:
        return "pymupdf"
    return getattr(settings.ocr, "provider", "pymupdf")


def _detect_page_layout(page) -> str:
    """Classify page layout as 'prose', 'two_column', or 'table'.

    Two-column pages have a bimodal X distribution — two clusters of block
    centres separated by a wide gap.  True tables have irregular block sizes
    and may span the full width.  Single-column prose pages have few blocks
    that don't cluster bimodally.
    """
    blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
    if len(blocks) < 8:
        return "prose"

    x_centers = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in blocks]
    page_mid = page.rect.width / 2

    left = sum(1 for x in x_centers if x < page_mid)
    right = sum(1 for x in x_centers if x >= page_mid)

    # Bimodal: both sides have meaningful block counts and are roughly balanced
    if left >= 4 and right >= 4:
        balance = min(left, right) / max(left, right)
        if balance >= 0.35:
            # Sanity-check: both columns should start at roughly the same Y position.
            # If the right column starts significantly higher than the left, the layout
            # is "header-right / body-left" (like presidential decree pages where the
            # act header is top-right and the body text is below-left) — `get_text`
            # handles that better than _extract_two_column.
            left_blocks  = [b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < page_mid]
            right_blocks = [b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 >= page_mid]
            min_y_left  = min(b["bbox"][1] for b in left_blocks)
            min_y_right = min(b["bbox"][1] for b in right_blocks)
            if min_y_right < min_y_left - 60:
                # Right column leads significantly — not a standard parallel two-column layout
                return "prose"
            return "two_column"

    # Many blocks but not bimodal → genuine table layout
    # Few blocks that failed bimodal → still prose
    return "table" if len(blocks) >= 20 else "prose"


def _extract_two_column(page) -> str:
    """Read a two-column page as: full left column top-to-bottom, then full right column.

    This avoids the row-interleaving artefact of the word-position approach,
    which produces garbled text when columns contain unrelated acts.
    """
    blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
    mid = page.rect.width / 2

    left_blocks  = sorted([b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < mid],
                          key=lambda b: b["bbox"][1])
    right_blocks = sorted([b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 >= mid],
                          key=lambda b: b["bbox"][1])

    def _block_text(b: dict) -> str:
        return "\n".join(
            " ".join(span["text"] for span in line["spans"])
            for line in b["lines"]
        )

    left_text  = "\n".join(_block_text(b) for b in left_blocks)
    right_text = "\n".join(_block_text(b) for b in right_blocks)
    return left_text + "\n" + right_text


def _extract_page_words(page) -> str:
    """Reconstruct page text row-by-row from word positions.

    Groups words that share the same vertical position into lines, sorted
    left-to-right. Preserves table cell values lost by get_text("text") on
    multi-column table layouts.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_idx)
    if not words:
        return ""

    Y_BUCKET = 6  # points — words within 6pt vertically are on the same row
    rows: dict[int, list[tuple[float, str]]] = {}
    for w in words:
        x0, y0, word = w[0], w[1], w[4]
        key = round(y0 / Y_BUCKET) * Y_BUCKET
        rows.setdefault(key, []).append((x0, word))

    lines = []
    for y in sorted(rows):
        row_words = sorted(rows[y], key=lambda t: t[0])
        lines.append("  ".join(w for _, w in row_words))
    return "\n".join(lines)


def _ocr_all_pages_local(pdf_path: str) -> list[str]:
    """OCR entire PDF using Apple Vision via ocrmac (local/macOS only)."""
    doc = fitz.open(pdf_path)
    results = [_ocr_page_image_local(page) for page in doc]
    doc.close()
    return results


def _ocr_single_page_local(pdf_path: str, page_number: int) -> str:
    doc = fitz.open(pdf_path)
    text = _ocr_page_image_local(doc[page_number])
    doc.close()
    return text


def _ocr_page_image_local(page) -> str:
    """Render a fitz page to a PIL image and OCR it with Apple Vision (macOS only)."""
    import io
    from PIL import Image as PILImage
    from ocrmac import ocrmac  # macOS-only; not imported on cloud path
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img = PILImage.open(io.BytesIO(pix.tobytes("png")))
    annotations = ocrmac.OCR(img, recognition_level="accurate").recognize()
    return "\n".join(text for text, _, _ in annotations)


def _is_facsimile_page(page) -> bool:
    """Detect subset-font pages that need OCR (HYBRID era)."""
    import re
    fonts = set()
    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    fonts.add(span["font"])
    if not fonts:
        return False
    subset = [f for f in fonts if re.match(r'^[A-Z]{6}\+', f)]
    return len(subset) > 0 and len(subset) >= len(fonts) - len(subset)
