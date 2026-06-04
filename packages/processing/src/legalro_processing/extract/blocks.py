from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import fitz
from legalro_core.models import Era

PAGE_WIDTH    = 595.0
PAGE_HEIGHT   = 842.0
GUTTER_X      = 297.0
HEADER_BAND_Y = (0, 55)
FOOTER_BAND_Y = (810, 842)

HEADER_RE = re.compile(
    r"MONITORUL\s+OFICIAL\s+AL\s+ROM[ÂA]NIEI,?\s*PARTEA\s+I\s*,?\s*Nr\.\s*[\d/]",
    re.IGNORECASE,
)

_LEGACY_ERAS = {"broken_2002", "broken_2007"}

_DIR_TO_ROTATION = {
    (1, 0):  0,
    (0, 1):  90,
    (-1, 0): 180,
    (0, -1): 270,
}


@dataclass
class Block:
    block_id: str
    page_index: int
    bbox: tuple[float, float, float, float]
    column: str
    role: str
    text: str
    font: str
    font_size: float
    is_bold: bool
    rotation: int
    claimed_by: str | None = None
    confidence: float | None = None


def is_full_width_bbox(bbox, page_w: float, gutter: float = GUTTER_X) -> bool:
    x0, y0, x1, y1 = bbox
    # Strict full-width: headers, wide banners spanning almost the whole page
    if x0 <= page_w * 0.13 and x1 >= page_w * 0.87:
        return True
    # Gutter-spanning: block crosses the column divider by more than 20pt on each side.
    # Catches centered text (issuers, act types, preambles, articles) that span both
    # columns even when they don't reach the page margins — essential for correct
    # reading order on single-column decree pages.
    if x0 < gutter - 20 and x1 > gutter + 20:
        return True
    return False


def is_full_width(block: Block, page_w: float) -> bool:
    return is_full_width_bbox(block.bbox, page_w)


def _assign_column(bbox, page_w: float, gutter: float = GUTTER_X) -> str:
    x0, y0, x1, y1 = bbox
    if is_full_width_bbox(bbox, page_w):
        return "FULL"
    cx = (x0 + x1) / 2
    return "L" if cx < gutter else "R"


def extract_fitz_blocks(pdf_path, era: Era) -> list[list[Block]]:
    doc = fitz.open(str(pdf_path))
    is_legacy = era.value in _LEGACY_ERAS

    # Use the complete era-specific normalization table from core (which includes
    # all 11 BROKEN_2007 mappings — e.g. „→ă, ∫→ș, ˛→ț, ‚→â — not just the 3
    # in roles.repair_legacy_encoding). This is critical: without it, 'București'
    # appears as 'Bucure˛ti', breaking ACT_CLOSING regex and losing entire acts.
    if is_legacy:
        from legalro_core.normalize import normalize_text as _normalize_text
        def _repair(t: str) -> str:
            return _normalize_text(t, era)
    else:
        _repair = None

    pages: list[list[Block]] = []

    for page_index, page in enumerate(doc):
        page_w = page.rect.width
        raw = page.get_text("dict")
        page_blocks: list[Block] = []
        idx = 0

        for fitz_block in raw["blocks"]:
            if fitz_block["type"] != 0:
                continue

            line_texts: list[str] = []
            font_names: list[str] = []
            font_sizes: list[float] = []
            any_bold = False
            rotation = 0

            for line in fitz_block["lines"]:
                span_parts: list[str] = []
                for span in line["spans"]:
                    t = span["text"]
                    span_parts.append(t)
                    font_names.append(span["font"])
                    font_sizes.append(span["size"])
                    flags = span.get("flags", 0)
                    if (flags & 16) or "Bold" in span["font"]:
                        any_bold = True
                    d = span.get("dir", (1, 0))
                    key = (round(d[0]), round(d[1]))
                    rotation = _DIR_TO_ROTATION.get(key, 0)
                # Spans within a line join directly; lines join with \n
                line_text = "".join(span_parts)
                if line_text.strip():
                    line_texts.append(line_text)

            text = "\n".join(line_texts).strip()
            if not text:
                continue

            if is_legacy and _repair is not None:
                text = _repair(text)

            font = statistics.mode(font_names) if font_names else ""
            font_size = statistics.median(font_sizes) if font_sizes else 0.0

            bbox_raw = fitz_block["bbox"]
            bbox = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
            column = _assign_column(bbox, page_w)
            block_id = f"p{page_index}-b{idx:03d}"

            page_blocks.append(Block(
                block_id=block_id,
                page_index=page_index,
                bbox=bbox,
                column=column,
                role="unknown",
                text=text,
                font=font,
                font_size=font_size,
                is_bold=any_bold,
                rotation=rotation,
            ))
            idx += 1

        pages.append(page_blocks)

    doc.close()
    return pages


def strip_chrome(
    blocks: list[Block],
    page_w: float,
    page_h: float,
) -> tuple[list[Block], Block | None, Block | None]:
    header_block = None
    footer_block = None
    body = []
    for b in blocks:
        if b.bbox[3] < page_h * 0.07:
            b.role = "running_header"
            header_block = b
            continue
        if b.bbox[1] > page_h * 0.96:
            b.role = "footer_rule"
            footer_block = b
            continue
        if b.bbox[3] < page_h * 0.12 and HEADER_RE.search(b.text):
            b.role = "running_header"
            header_block = b
            continue
        if b.bbox[1] > page_h * 0.90 and re.match(r'^\s*\d{1,3}\s*$', b.text):
            b.role = "page_number"
            continue
        body.append(b)
    return body, header_block, footer_block


def reading_order(
    blocks: list[Block],
    page_w: float = PAGE_WIDTH,
    gutter: float = GUTTER_X,
) -> list[Block]:
    body = [b for b in blocks if b.role not in {"running_header", "footer_rule", "page_number"}]
    full_w = sorted([b for b in body if is_full_width(b, page_w)], key=lambda b: b.bbox[1])
    cols   = [b for b in body if not is_full_width(b, page_w)]
    out = []
    cursor_y = 0.0
    for fw in full_w + [None]:
        ymax = fw.bbox[1] if fw is not None else 1e9
        slab = [b for b in cols if cursor_y <= b.bbox[1] < ymax]
        left  = sorted([b for b in slab if (b.bbox[0] + b.bbox[2]) / 2 < gutter], key=lambda b: b.bbox[1])
        right = sorted([b for b in slab if (b.bbox[0] + b.bbox[2]) / 2 >= gutter], key=lambda b: b.bbox[1])
        out.extend(left)
        out.extend(right)
        if fw is not None:
            out.append(fw)
            cursor_y = fw.bbox[3]
    return out
