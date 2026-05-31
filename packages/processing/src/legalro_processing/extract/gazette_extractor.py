"""
PDF → GazetteDocument extractor.

Reads a gazette PDF and produces a structured GazetteDocument that is
serialised to JSON in extracted/{year}/{month}/{day}/{filename}.json.

Usage (CLI shortcut via scripts/extract_gazette.py):
    uv run python scripts/extract_gazette.py laws/2017/01/30/MO_PI_76_2017-01-30.pdf

The resulting JSON is the canonical source for ingestion; you can edit it
to fix OCR errors, wrong metadata, or missing articles before re-ingesting.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import fitz

from legalro_processing.extract.era import detect_era
from legalro_processing.extract.extract import extract_text
from legalro_core.normalize import normalize_pages
from legalro_processing.extract.structure import strip_structural
from legalro_processing.extract.sumar import parse_sumar as _parse_sumar_entries, SumarBoundary
from legalro_processing.extract.segment import segment_acts
from legalro_processing.extract.metadata import extract_metadata
from legalro_processing.extract.gazette_schema import (
    GazetteDocument, LegalAct, SumarEntry, Article, Annex,
)
from legalro_core.models import Era

EXTRACTION_VERSION = "1.0.0"

FILENAME_PATTERN = re.compile(
    r'MO_P([IV]+)_(\d+(?:Bis)?)_(\d{4})-(\d{2})-(\d{2})\.pdf', re.IGNORECASE
)

ARTICLE_SPLIT = re.compile(r'(?m)^(?=(?:Art\.|Articolul?\s+unic|Articol\s+unic)\s)', re.IGNORECASE)
ARTICLE_NUMBER = re.compile(r'^(?:Art\.|Articolul?)\s*(\d+|unic)', re.IGNORECASE)
ALINEAT_SPLIT = re.compile(r'(?=\s*\(\d+\)\s)', re.MULTILINE)
ANNEX_SPLIT = re.compile(r'(?=ANEX[AĂ]\s+[Nn]r\.?\s*\d+|\bANEX[AĂ]\b)', re.IGNORECASE)
ANNEX_NUMBER = re.compile(r'ANEX[AĂ]\s+[Nn]r\.?\s*(\S+)', re.IGNORECASE)
SIGNED_RE = re.compile(
    r'(?:p\.?\s*)?(?:Prim-ministru|Ministru[l]?|Președint[ele]+|Director[ul]*)\s*[,\n]+\s*([A-ZĂÂÎȘȚ][^\n]{3,50})',
    re.IGNORECASE | re.MULTILINE,
)

# Fallback: extract true act number from body text when segmenter produces a wrong one.
# Match a standalone act-type + number header line (e.g. "DECIZIA  Nr. 922\ndin 18 oct")
# NOT inline references like "art. 21 din Legea nr. 47/1992".
_ACT_NUMBER_IN_TEXT = re.compile(
    r'^(?:DECIZIA|HOTĂRÂREA|DECRETUL|ORDINUL|ORDONAN[TȚ]A)\s{1,4}[Nn]r\.\s*([\d.]+)',
    re.IGNORECASE | re.MULTILINE,
)

# ── Public API ────────────────────────────────────────────────────────────────

def extract_gazette(pdf_path: str | Path, settings=None) -> GazetteDocument:
    """Extract a gazette PDF to a structured GazetteDocument."""
    path = Path(pdf_path).resolve()
    warnings: list[str] = []

    # ── Identity ──────────────────────────────────────────────────────
    match = FILENAME_PATTERN.match(path.name)
    if match:
        part = match.group(1)
        issue_str = match.group(2)
        is_bis = issue_str.lower().endswith("bis")
        issue_number = int(issue_str.replace("Bis", "").replace("bis", ""))
        year = int(match.group(3))
        month = int(match.group(4))
        day = int(match.group(5))
        issue_date = f"{year:04d}-{month:02d}-{day:02d}"
    else:
        warnings.append(f"Filename doesn't match expected pattern: {path.name}")
        part, issue_number, is_bis, year, issue_date = "I", 0, False, 0, "0000-00-00"

    gazette_id = f"P{part}_{issue_number}{'Bis' if is_bis else ''}_{year}"
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    doc_fitz = fitz.open(str(path))
    pdf_page_count = len(doc_fitz)
    doc_fitz.close()

    era = detect_era(str(path))

    # ── Sumar (cover page) — text-based parser for all eras ──────────
    # We always parse sumar from page-0 raw text; the block pipeline handles body pages.
    doc_fitz2 = fitz.open(str(path))
    cover_text = doc_fitz2[0].get_text("text") if pdf_page_count > 0 else ""
    page_heights = [doc_fitz2[i].rect.height for i in range(pdf_page_count)]
    doc_fitz2.close()

    sumar_raw = normalize_pages([cover_text], era)[0] if cover_text else ""
    sumar_entries = _build_sumar(sumar_raw, warnings, era=era)
    year_label, weekday = _parse_header(sumar_raw)

    # ── Acts — route by era ───────────────────────────────────────────
    if era == Era.SCANNED:
        # M1 path: LlamaParse / OCR → text → old text-based segmenter (unchanged).
        try:
            raw_pages = extract_text(str(path), era, settings)
        except Exception as exc:
            warnings.append(f"Text extraction failed: {exc}")
            raw_pages = [""] * pdf_page_count
        pages = normalize_pages(raw_pages, era)
        pages = strip_structural(pages)
        rich_boundaries = [
            SumarBoundary(title=e.title, page_number=e.page_start)
            for e in sumar_entries if e.page_start and e.page_start > 0
        ]
        segmenter_input = rich_boundaries if rich_boundaries else _parse_sumar_entries(sumar_raw)
        raw_acts = segment_acts(pages, segmenter_input, era, expected_n=len(sumar_entries))
    else:
        # M2/M3 path: block-role pipeline (spec §3.2–§3.11).
        from legalro_processing.extract.blocks import (
            extract_fitz_blocks, strip_chrome, reading_order, PAGE_WIDTH,
        )
        from legalro_processing.extract.roles import classify_blocks
        from legalro_processing.extract.segment import segment_acts_from_blocks

        all_page_blocks = extract_fitz_blocks(str(path), era)
        processed: list[list] = []
        for page_idx, page_blocks in enumerate(all_page_blocks):
            if page_idx == 0:
                processed.append(page_blocks)  # cover: placeholder; segmenter skips index 0
                continue
            page_h = page_heights[page_idx] if page_idx < len(page_heights) else 842.0
            body, _, _ = strip_chrome(page_blocks, PAGE_WIDTH, page_h)
            ordered = reading_order(body, PAGE_WIDTH)
            classified = classify_blocks(ordered, page_idx, PAGE_WIDTH)
            processed.append(classified)

        raw_acts = segment_acts_from_blocks(processed, year)

    # ── Reconciliation warnings ───────────────────────────────────────
    expected_n = len(sumar_entries)
    produced_n = len(raw_acts)
    if expected_n >= 2:
        if produced_n < expected_n // 2:
            warnings.append(f"under-segmentation: sumar={expected_n} acts, produced={produced_n}")
        elif produced_n > expected_n * 3:
            warnings.append(f"over-segmentation: sumar={expected_n} acts, produced={produced_n}")

    acts: list[LegalAct] = []
    for act_idx, raw_act in enumerate(raw_acts):
        meta = extract_metadata(raw_act, year)
        meta["_gazette_issue_number"] = issue_number
        act = _build_act(act_idx, raw_act.text, meta, raw_act.page_range, year)
        acts.append(act)

    # ── Propagate metadata from parent acts to annexe acts ────────────
    # _split_by_closing correctly splits ANEXĂ content into separate acts,
    # but those acts have no issuer/act_type header → doc_type=UNKNOWN.
    # Inherit the preceding typed act's doc_type and authority so retrieval
    # filters can reach annexe content too.
    for i in range(1, len(acts)):
        a = acts[i]
        if a.doc_type == "UNKNOWN" and a.full_text.strip()[:10].upper().startswith("ANEX"):
            parent = acts[i - 1]
            if parent.doc_type != "UNKNOWN":
                a.doc_type = parent.doc_type
            if not a.issuing_authority:
                a.issuing_authority = parent.issuing_authority
            # Also propagate act_number so annexe chunks are retrievable via
            # act-number metadata boost (e.g. "HG 1448 taxi norms" → finds annexe).
            if (not a.act_number or a.act_number == "0") and parent.act_number:
                a.act_number = parent.act_number
                a.act_year = parent.act_year

    # ── Assemble ──────────────────────────────────────────────────────
    return GazetteDocument(
        filename=path.name,
        gazette_id=gazette_id,
        part=part,
        issue_number=issue_number,
        is_bis=is_bis,
        issue_year=year,
        issue_date=issue_date,
        era=era.value,
        year_label=year_label,
        weekday=weekday,
        pdf_page_count=pdf_page_count,
        sha256=sha256,
        sumar=sumar_entries,
        sumar_raw=sumar_raw,
        acts=acts,
        extraction_version=EXTRACTION_VERSION,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        extraction_warnings=warnings,
    )


def save_gazette(gazette: GazetteDocument, output_dir: str | Path) -> Path:
    """Serialise GazetteDocument to JSON. Returns the output path."""
    out = Path(output_dir)
    # mirror the laws/{year}/{month}/{day} directory structure
    if gazette.issue_date != "0000-00-00":
        parts = gazette.issue_date.split("-")
        out = out / parts[0] / parts[1] / parts[2]
    out.mkdir(parents=True, exist_ok=True)

    stem = gazette.filename.replace(".pdf", "")
    out_path = out / f"{stem}.json"
    out_path.write_text(
        json.dumps(dataclasses.asdict(gazette), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def load_gazette(json_path: str | Path) -> GazetteDocument:
    """Deserialise a previously extracted GazetteDocument JSON."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    # Reconstruct nested dataclasses
    data["sumar"] = [SumarEntry(**e) for e in data.get("sumar", [])]
    data["acts"] = [_load_act(a) for a in data.get("acts", [])]
    return GazetteDocument(**data)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_header(sumar_text: str) -> tuple[Optional[str], Optional[str]]:
    year_m = re.search(r'(Anul\s+\d+\s*\([IVXLCDM]+\))', sumar_text)
    year_label = year_m.group(1).strip() if year_m else None

    weekday_m = re.search(
        r'\b(Luni|Marți|Miercuri|Joi|Vineri|Sâmbătă|Duminică)\b', sumar_text, re.IGNORECASE
    )
    weekday = weekday_m.group(1) if weekday_m else None
    return year_label, weekday


# Dot-leader patterns:
#   modern:  ..........  (3+ consecutive dots)
#   1989 OCR: . . . . . • • •  (spaced dots and/or bullets)
_DOTS = re.compile(r'(?:\.{3,}|(?:\.\s+){3,}\.?|(?:[•·]\s*){3,})\s*$')

# Standalone page reference on its own line: "9" or "2–8" or "2-8"
_PAGE_ONLY = re.compile(r'^(\d+)(?:[–\-](\d+))?\s*$')

# Line that starts with an act number: "1.607/2016. —" or "115. —" or "115."
_ENTRY_START = re.compile(r'^([\d][\d.]*(?:/\d+)?)\.\s*(?:[—–\-]\s*)?')

# Bare number on its own line (2026 PM style: "115.\n— description...")
_NUMBER_ALONE = re.compile(r'^([\d][\d.]*)\.\s*$')

# All-caps category headers (relaxed to 6 chars to catch "DECRETE", "LEGI" etc.)
_CATEGORY = re.compile(r'^[A-ZĂÂÎȘȚ][A-ZĂÂÎȘȚ ,.\-]{5,}$')

# Lines that are structural noise in the sumar section
_SKIP = re.compile(
    r'^(?:Nr\.?|Pagina|S\s*U\s*M\s*A\s*R|SUMAR|MONITORUL\s+OFICIAL)',
    re.IGNORECASE,
)



def _build_sumar(sumar_text: str, warnings: list[str], era: Era = Era.MODERN) -> list[SumarEntry]:
    """
    Stateful line-walker that handles four sumar layouts found in MO:

      1. Standard (2017):      NUMBER/YEAR. — TITLE (multi-line) ......\n PAGE
      2. Modern PM (2026):     NUMBER.\n— TITLE (multi-line) ......\n PAGE
      3. Unnumbered (CCR):     TITLE (multi-line) ......\n PAGE  (no leading number)
      4. 1989 OCR:             TITLE . . . . • • •\nPagina\n PAGE

    States: HEADER → IN_ENTRY → AFTER_DOTS → (flush) → HEADER

    The page number always appears on its own line immediately after the
    dot-leader line (possibly with a "Pagina" label in between for 1989).
    """
    # Skip everything before SUMAR marker
    sumar_m = re.search(r'S\s*U\s*M\s*A\s*R', sumar_text, re.IGNORECASE)
    body = sumar_text[sumar_m.end():] if sumar_m else sumar_text

    # Fix two-column TOC tables where act numbers with dots were split across
    # columns: "1\t908/2006\t1\t540/2006" → two proper "1.908/2006\t1" entries.
    # Pattern: INTEGER TAB INTEGER/YEAR TAB INTEGER TAB INTEGER/YEAR (4-col row)
    # or:      INTEGER TAB INTEGER/YEAR TAB INTEGER (3-col row, one-sided)
    _TAB_SPLIT_4 = re.compile(r'^(\d+)\t(\d+/\d+)\t(\d+)\t(\d+/\d+)\s*$', re.MULTILINE)
    _TAB_SPLIT_3 = re.compile(r'^(\d+)\t(\d+/\d+)\t(\d+)\s*$', re.MULTILINE)
    _TAB_SPLIT_2 = re.compile(r'^(\d+)\t(\d+/\d+)\s*$', re.MULTILINE)

    def _fix_tab_row_4(m: re.Match) -> str:
        return f"{m.group(1)}.{m.group(2)}.\n{m.group(3)}.{m.group(4)}."

    def _fix_tab_row_3(m: re.Match) -> str:
        return f"{m.group(1)}.{m.group(2)}.\n{m.group(3)}."

    def _fix_tab_row_2(m: re.Match) -> str:
        return f"{m.group(1)}.{m.group(2)}."

    body = _TAB_SPLIT_4.sub(_fix_tab_row_4, body)
    body = _TAB_SPLIT_3.sub(_fix_tab_row_3, body)
    body = _TAB_SPLIT_2.sub(_fix_tab_row_2, body)

    entries: list[SumarEntry] = []
    current_category = ""
    act_number = ""
    title_lines: list[str] = []
    seen_categories: set[str] = set()

    # States: "header", "in_entry", "after_dots"
    state = "header"

    def _flush(page_start: int, page_end: Optional[int]) -> None:
        nonlocal act_number, title_lines, state
        title = " ".join(title_lines).strip()
        title = re.sub(r'^[—–\-]\s*', '', title)
        if title:
            entries.append(SumarEntry(
                act_number=act_number,
                doc_type=_infer_doc_type(title, act_number),
                title=title,
                page_start=page_start,
                page_end=page_end,
                category=current_category,
            ))
        act_number = ""
        title_lines = []
        state = "header"

    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if _SKIP.match(s):
            continue

        # ── AFTER_DOTS: expecting a page number (possibly after a "Pagina" label) ──
        if state == "after_dots":
            page_m = _PAGE_ONLY.match(s)
            if page_m:
                _flush(int(page_m.group(1)), int(page_m.group(2)) if page_m.group(2) else None)
                continue
            # Anything that looks like a new entry or category without a page → flush with page=0
            entry_m = _ENTRY_START.match(s)
            is_cat = _CATEGORY.match(s) and not _DOTS.search(s)
            if entry_m or is_cat:
                _flush(0, None)
                # fall through to handle this line in header/entry logic below
            else:
                # continuation line after dots (e.g. "Pagina" label) — stay in after_dots
                continue

        # ── Category header ───────────────────────────────────────────────────────
        is_cat = _CATEGORY.match(s) and not _DOTS.search(s)
        if is_cat and state in ("header", "in_entry"):
            # For SCANNED era: a repeated category means we've left the sumar and
            # entered the article body (two-column OCR bleeds both together).
            if era == Era.SCANNED and s in seen_categories and entries:
                break
            if state == "in_entry":
                _flush(0, None)
            seen_categories.add(s)
            current_category = s
            state = "header"
            continue

        # ── Bare number on its own line (2026 PM style) ───────────────────────────
        num_alone_m = _NUMBER_ALONE.match(s)
        if num_alone_m and not _DOTS.search(s):
            if state == "in_entry":
                _flush(0, None)
            act_number = num_alone_m.group(1)
            state = "in_entry"
            continue

        # ── Entry start (NUMBER. — ...) ───────────────────────────────────────────
        entry_m = _ENTRY_START.match(s)
        if entry_m:
            if state == "in_entry":
                _flush(0, None)
            act_number = entry_m.group(1)
            remainder = s[entry_m.end():].strip()
            remainder = re.sub(r'^[—–\-]\s*', '', remainder)
            state = "in_entry"
            if _DOTS.search(s):
                # title and dots on same line
                clean = _DOTS.sub('', remainder).strip()
                if clean:
                    title_lines = [clean]
                state = "after_dots"
            elif remainder:
                title_lines = [remainder]
            continue

        # ── Dot-leader line ───────────────────────────────────────────────────────
        if _DOTS.search(s):
            clean = _DOTS.sub('', s).strip()
            clean = re.sub(r'^[—–\-]\s*', '', clean)
            if clean:
                title_lines.append(clean)
            state = "after_dots"
            continue

        # ── Page number on its own line after dot-free entry (e.g. "2–25") ─────────
        if state == "in_entry" and title_lines:
            page_m = _PAGE_ONLY.match(s)
            if page_m:
                _flush(int(page_m.group(1)), int(page_m.group(2)) if page_m.group(2) else None)
                continue

        # ── Plain continuation line ───────────────────────────────────────────────
        if state in ("in_entry", "header"):
            clean = re.sub(r'^[—–\-]\s*', '', s)
            title_lines.append(clean)
            state = "in_entry"

    # flush any trailing open entry
    if title_lines:
        _flush(0, None)

    if not entries:
        warnings.append("Sumar parsing produced 0 entries — may be SCANNED era or unusual layout")

    return entries


_DOC_TYPE_PATTERNS = [
    (re.compile(r'\bordin\b', re.I), "ORDIN"),
    (re.compile(r'\bhotărâre\b|\bhg\b', re.I), "HG"),
    (re.compile(r'\bdecret\s*-\s*lege\b', re.I), "DECRET_LEGE"),
    (re.compile(r'\bdecret\b', re.I), "DECRET"),
    (re.compile(r'\bdecizie\b', re.I), "DECIZIE"),
    (re.compile(r'\blege\b', re.I), "LEGE"),
    (re.compile(r'\bordonan[țt][ăa]\s+de\s+urgen[țt][ăa]\b', re.I), "OUG"),
    (re.compile(r'\bordonan[țt][ăa]\b', re.I), "ORDONANȚĂ"),
    (re.compile(r'\brectificare\b', re.I), "RECTIFICARE"),
    (re.compile(r'\bcomunicat\b', re.I), "COMUNICAT"),
    (re.compile(r'\banun[țt]\b', re.I), "ANUNT"),
]


def _infer_doc_type(title: str, act_number: str) -> str:
    for pattern, dtype in _DOC_TYPE_PATTERNS:
        if pattern.search(title):
            return dtype
    return "ACT"


def _build_act(
    act_idx: int,
    text: str,
    meta: dict,
    page_range: list[int],
    gazette_year: int,
) -> LegalAct:
    act_warnings: list[str] = []

    # Split preamble / body at first article
    parts = ARTICLE_SPLIT.split(text, maxsplit=1)
    preamble = parts[0].strip()
    body = parts[1] if len(parts) > 1 else ""

    # Parse articles
    articles = _parse_articles(body, act_warnings)

    # Parse annexes (appear after last article or at end of text)
    annex_parts = ANNEX_SPLIT.split(text)
    annexes = _parse_annexes(annex_parts[1:]) if len(annex_parts) > 1 else []

    # Signatories
    signed_by = [m.group(1).strip() for m in SIGNED_RE.finditer(text)]
    # deduplicate while preserving order
    seen: set[str] = set()
    signed_by_unique = [s for s in signed_by if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

    # act_year from act_number if present
    act_number = str(meta.get("act_number", ""))
    act_year: Optional[int] = None
    year_m = re.search(r'/(\d{4})$', act_number)
    if year_m:
        act_year = int(year_m.group(1))
        act_number = act_number[: year_m.start()]
    elif gazette_year:
        act_year = gazette_year

    # Fallback: when act_number is missing or looks like the gazette issue number
    # (not a real act number), try to extract the true number from the act body
    # (e.g. "DECIZIA Nr. 922" → "922"). Search up to 4000 chars since the act
    # header may be preceded by signatures from a prior decision on the same page.
    if not act_number or (gazette_year and act_number == str(meta.get("_gazette_issue_number", ""))):
        body_head = text[:4000]
        m_in_text = _ACT_NUMBER_IN_TEXT.search(body_head)
        if m_in_text:
            candidate = m_in_text.group(1).rstrip(".")
            if candidate != act_number:
                act_warnings.append(
                    f"act_number corrected from {act_number!r} to {candidate!r} via body-text fallback"
                )
                act_number = candidate

    page_start = page_range[0] if page_range else 0
    page_end = page_range[-1] if page_range else 0

    return LegalAct(
        act_index=act_idx,
        doc_type=str(meta.get("doc_type", "ACT")),
        act_number=act_number,
        act_year=act_year,
        issuing_authority=str(meta.get("issuing_authority", "")),
        title=str(meta.get("title", "")),
        locality=meta.get("locality") or None,
        preamble=preamble,
        articles=articles,
        annexes=annexes,
        full_text=text,
        page_start=page_start,
        page_end=page_end,
        signed_by=signed_by_unique,
        countersigned_by=[],
        extraction_warnings=act_warnings,
    )


def _parse_articles(body: str, warnings: list[str]) -> list[Article]:
    if not body.strip():
        return []

    raw_articles = ARTICLE_SPLIT.split(body)
    articles = []

    for raw in raw_articles:
        raw = raw.strip()
        if not raw:
            continue

        num_m = ARTICLE_NUMBER.match(raw)
        article_number = num_m.group(1) if num_m else "?"
        if article_number == "?":
            warnings.append(f"Could not parse article number from: {raw[:60]!r}")

        # Extract optional inline title
        first_line = raw.split("\n")[0]
        title_m = re.search(r'—\s+(.+)$', first_line)
        article_title = title_m.group(1).strip() if title_m else None

        # Split into alineate
        alineat_parts = ALINEAT_SPLIT.split(raw)
        alineate = [a.strip() for a in alineat_parts if a.strip()]

        articles.append(Article(
            article_number=article_number,
            title=article_title,
            alineate=alineate,
            raw_text=raw,
        ))

    return articles


def _parse_annexes(parts: list[str]) -> list[Annex]:
    annexes = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        num_m = ANNEX_NUMBER.search(part)
        annex_number = num_m.group(1) if num_m else "?"
        first_line = part.split("\n")[0]
        title_m = re.search(r'—\s+(.+)$', first_line)
        title = title_m.group(1).strip() if title_m else None
        annexes.append(Annex(annex_number=annex_number, title=title, raw_text=part))
    return annexes


def _load_act(data: dict) -> LegalAct:
    data["articles"] = [Article(**a) for a in data.get("articles", [])]
    data["annexes"] = [Annex(**a) for a in data.get("annexes", [])]
    return LegalAct(**data)
