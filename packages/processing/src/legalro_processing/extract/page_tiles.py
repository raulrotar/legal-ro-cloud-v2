"""Page tiling for VLM OCR (scanned era).

Ollama's glm-ocr port clips images larger than 2048×2048 (ollama#14114) and a
rendered gazette page is taller than that — so whole-page OCR silently loses
the bottom of every page.

Tiles must be as LARGE as possible: GLM-OCR is a document model that handles
two-column reading order natively, while small context-free crops make it
ramble into repetition loops (and multiply per-call latency).  So a page is
cut only horizontally — full-width segments under the size limit, cut at the
blankest row so no text line is ever split.  At 200 DPI an A4-ish page is
~1650×2340 px → exactly two tiles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INK_THRESHOLD = 160        # gray value below which a pixel counts as ink


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def _trim_box(ink: np.ndarray) -> Tile | None:
    """Bounding box of all ink on the page, padded; None for a blank page."""
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    pad = 8
    h, w = ink.shape
    return Tile(
        x0=max(0, int(cols[0]) - pad),
        y0=max(0, int(rows[0]) - pad),
        x1=min(w, int(cols[-1]) + 1 + pad),
        y1=min(h, int(rows[-1]) + 1 + pad),
    )


def _split_box_by_height(ink: np.ndarray, box: Tile, max_px: int,
                         min_tile_height: int) -> list[Tile]:
    """Cut a box into segments ≤ max_px tall at the blankest rows."""
    row_ink = ink[:, box.x0:box.x1].sum(axis=1)
    tiles: list[Tile] = []
    y = box.y0
    while box.y1 - y > max_px:
        lo = y + int(max_px * 0.55)
        hi = y + max_px
        cut = lo + int(np.argmin(row_ink[lo:hi]))
        tiles.append(Tile(box.x0, y, box.x1, cut))
        y = cut
    last = Tile(box.x0, y, box.x1, box.y1)
    if tiles and last.height < min_tile_height:
        prev = tiles.pop()
        last = Tile(box.x0, prev.y0, box.x1, box.y1)
    tiles.append(last)
    return tiles


def _find_gutter(ink: np.ndarray, box: Tile, min_gutter: int = 16) -> int | None:
    """x position of the widest near-blank vertical gutter in the central
    25–75% of the box, or None for single-column content."""
    region = ink[box.y0:box.y1, box.x0:box.x1]
    h, w = region.shape
    if w < 500 or h < 200:
        return None
    col_ink = region.sum(axis=0) / h
    blank = col_ink < 0.004
    best_start = best_len = 0
    run_start = None
    for x in range(w):
        if blank[x]:
            if run_start is None:
                run_start = x
        else:
            if run_start is not None and x - run_start > best_len:
                best_start, best_len = run_start, x - run_start
            run_start = None
    if run_start is not None and w - run_start > best_len:
        best_start, best_len = run_start, w - run_start
    mid = best_start + best_len // 2
    if best_len >= min_gutter and 0.25 * w < mid < 0.75 * w:
        return box.x0 + mid
    return None


def tile_page(
    gray: np.ndarray,
    *,
    max_px: int = 2000,
    min_tile_height: int = 120,
    split_columns: bool = False,
) -> list[Tile]:
    """Cut a grayscale page (H×W uint8) into OCR tiles ≤ max_px tall.

    Default: full-width segments cut at the blankest row (a text line is never
    split); GLM-OCR handles two-column reading order natively, and big tiles
    avoid the repetition loops that small context-free crops provoke.

    split_columns=True is the ESCALATION mode for pages where the model still
    drops content: the page is split at the central gutter (when one exists)
    and each column is OCR'd separately, full left column before right.
    """
    ink = gray < INK_THRESHOLD
    box = _trim_box(ink)
    if box is None:
        return []

    if split_columns:
        gutter_x = _find_gutter(ink, box)
        if gutter_x is not None:
            left = Tile(box.x0, box.y0, gutter_x, box.y1)
            right = Tile(gutter_x, box.y0, box.x1, box.y1)
            return (
                _split_box_by_height(ink, left, max_px, min_tile_height)
                + _split_box_by_height(ink, right, max_px, min_tile_height)
            )

    return _split_box_by_height(ink, box, max_px, min_tile_height)


def pixmap_to_gray(pix) -> np.ndarray:
    """Convert a PyMuPDF grayscale Pixmap to an H×W uint8 numpy array."""
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.height, pix.width if pix.n == 1 else -1)[:, : pix.width]


def crop_png(gray: np.ndarray, tile: Tile, max_px: int = 2000) -> bytes:
    """Encode a tile crop as PNG bytes, downscaling if a side exceeds max_px."""
    import io

    from PIL import Image

    crop = gray[tile.y0:tile.y1, tile.x0:tile.x1]
    img = Image.fromarray(crop, mode="L")
    if img.width > max_px or img.height > max_px:
        scale = max_px / max(img.width, img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
