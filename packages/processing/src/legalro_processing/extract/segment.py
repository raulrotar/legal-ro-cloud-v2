"""Split gazette text into individual legal acts."""
import re
from dataclasses import dataclass
from legalro_processing.extract.sumar import SumarBoundary as SumarEntry
from legalro_core.models import Era


@dataclass
class RawAct:
    text: str
    title: str
    page_range: list[int]
    position_in_gazette: int


SPACED_TITLE = re.compile(r'^([A-ZĂÂÎȘȚ]\s+){3,}', re.MULTILINE)
ACT_CLOSING = re.compile(r'Bucure[șs]ti,\s+\d{1,2}\s+\w+\s+\d{4}\.[\s\S]{0,120}?Nr\.\s*[\d.]+\.')

INSTITUTION_HEADERS = [
    "PARLAMENTUL ROMÂNIEI", "CURTEA CONSTITUȚIONALĂ",
    "GUVERNUL ROMÂNIEI", "PREȘEDINTELE ROMÂNIEI",
    "MINISTERUL", "BANCA NAȚIONALĂ",
]
DELIMITER_2002 = '«'

# Page-start authority headers: when a page begins with one of these, it
# signals a new act.  Used as a fallback when SUMAR page numbers are all
# the same (collapsed two-column layout).
PAGE_START_AUTHORITIES = re.compile(
    r'^(?:AGENȚIA\s+NA[ȚT]IONAL[ĂA]\s+DE\s+CADASTRU|'
    r'CURTEA\s+CONSTITU[ȚT]IONAL[ĂA]|'
    r'PARLAMENTUL\s+ROM[ÂA]NIEI|'
    r'GUVERNUL\s+ROM[ÂA]NIEI|'
    r'PRE[ȘS]EDINTELE\s+ROM[ÂA]NIEI|'
    r'BANCA\s+NA[ȚT]IONAL[ĂA]|'
    r'MINISTERUL\b)',  # \b prevents matching genitive "Ministerului" (two-column OCR artifact)
    re.IGNORECASE,
)


def segment_acts(
    pages: list[str],
    sumar_entries: list[SumarEntry],
    era: Era,
    expected_n: int = 0,
) -> list[RawAct]:
    full_text = "\n".join(pages)

    if sumar_entries:
        # If SUMAR page numbers are non-monotonic or exceed the PDF page count
        # (gazette-absolute numbers in two-column layout), fall back to page-header detection.
        if _sumar_degenerate(sumar_entries, len(pages)):
            page_acts = _segment_by_page_headers(pages)
            if page_acts:
                return _split_by_closing(page_acts, max_acts=expected_n)
        else:
            return _split_by_closing(_segment_by_sumar(pages, sumar_entries), max_acts=expected_n)

    if era == Era.BROKEN_2002 and DELIMITER_2002 in full_text:
        return _split_by_closing(_segment_by_delimiter(full_text, DELIMITER_2002), max_acts=expected_n)

    # Try page-header-based segmentation first (works for one-act-per-page
    # patterns like batches of ANCPI or CCR decisions).
    page_acts = _segment_by_page_headers(pages)
    if len(page_acts) > 1:
        return _split_by_closing(page_acts, max_acts=expected_n)

    result = _split_by_closing(_segment_by_patterns(full_text), max_acts=expected_n)
    # If pattern segmentation over-shoots the sumar count significantly, fall
    # back to the whole gazette as one act (SCANNED era without reliable structure).
    if expected_n >= 2 and len(result) > expected_n * 3:
        return [RawAct(text=full_text, title="", page_range=[], position_in_gazette=0)]
    result = _merge_mid_sentence_fragments(result)
    return result


# Minimum length in characters for a standalone act chunk.  Acts shorter than
# this are almost certainly OCR column-break artefacts (e.g. a two-column 1989
# communiqué where the right column continues a sentence from the left column).
_VERY_SHORT_ACT = 50


def _merge_mid_sentence_fragments(raw_acts: list[RawAct]) -> list[RawAct]:
    """Merge consecutive acts where one is a short column-break fragment.

    Fixes two-column PDF layout issues in SCANNED-era documents (e.g. 1989 MOs)
    where an OCR column boundary splits a sentence mid-word, producing a tiny
    dangling act that embeds nonsensically as a standalone chunk.
    """
    if len(raw_acts) <= 1:
        return raw_acts
    merged = [raw_acts[0]]
    for act in raw_acts[1:]:
        if len(act.text.strip()) < _VERY_SHORT_ACT and merged:
            prev = merged[-1]
            merged[-1] = RawAct(
                text=prev.text.rstrip() + "\n" + act.text,
                title=prev.title or act.title,
                page_range=prev.page_range,
                position_in_gazette=prev.position_in_gazette,
            )
        else:
            merged.append(act)
    return merged


def _sumar_degenerate(entries: list[SumarEntry], page_count: int = 0) -> bool:
    """Return True when SUMAR page numbers are clearly unreliable.

    Two failure modes:
    1. Non-monotonic sequence (e.g. [2, 1, 1, 1]) — two-column layout caused
       page numbers to be misplaced or contain act numbers instead.
    2. All-same values — no useful boundary information.
    3. Gazette-absolute page numbers: max page number exceeds PDF page count,
       meaning the SUMAR references the gazette's global pagination rather than
       this PDF's physical pages.
    """
    if len(entries) < 2:
        return False
    nums = [e.page_number for e in entries]
    # Non-monotonic
    if any(nums[i] < nums[i - 1] for i in range(1, len(nums))):
        return True
    # All same
    if len(set(nums)) == 1:
        return True
    # Gazette-absolute: references pages beyond the physical PDF
    if page_count and max(nums) > page_count:
        return True
    return False


def _segment_by_sumar(pages: list[str], entries: list[SumarEntry]) -> list[RawAct]:
    """Segment using sumar page boundaries.

    When multiple acts share the same gazette page (e.g. two short decrees per
    page), we group those entries and emit ONE RawAct per unique page boundary.
    _split_by_closing then divides the shared page text into individual acts.
    This avoids both empty acts (old naive slice) and duplicate acts (giving
    every entry the full shared-page text).
    """
    acts = []
    i = 0
    pos = 0
    while i < len(entries):
        start_page = entries[i].page_number - 1

        # Collect all entries that start on the same gazette page
        j = i + 1
        while j < len(entries) and entries[j].page_number == entries[i].page_number:
            j += 1

        # End of this page group: start of next group's page
        end_page = entries[j].page_number - 1 if j < len(entries) else len(pages)

        text = "\n".join(pages[max(0, start_page):end_page])
        # Carry the first entry's title; _split_by_closing will produce the rest
        acts.append(RawAct(
            text=text.strip(),
            title=entries[i].title,
            page_range=[start_page, end_page - 1],
            position_in_gazette=pos,
        ))
        pos += 1
        i = j
    return acts


def _segment_by_page_headers(pages: list[str]) -> list[RawAct]:
    """Group consecutive pages into acts based on authority headers at page start.

    Each page whose first non-empty line matches an institutional authority
    header starts a new act.  Contiguous pages without such a header are
    appended to the current act (annexes, tables, continuations).
    """
    if not pages:
        return []

    act_start_pages: list[int] = []
    for i, page in enumerate(pages):
        stripped = page.lstrip()
        if i == 0:
            # Page 0 is always the SUMAR/cover — never treat it as an act start,
            # even if its first line matches an authority header (TOC lines).
            continue
        if PAGE_START_AUTHORITIES.match(stripped):
            act_start_pages.append(i)

    if not act_start_pages:
        return []

    acts = []
    # If the first authority header is not on page 1, include pages 1..first-1
    # as a leading act (e.g. HG text before PM decisions in BROKEN_2007 gazettes).
    first = act_start_pages[0]
    if first > 1:
        preamble_text = "\n".join(pages[1:first]).strip()
        if preamble_text:
            acts.append(RawAct(
                text=preamble_text,
                title="",
                page_range=[1, first - 1],
                position_in_gazette=0,
            ))
    for j, start in enumerate(act_start_pages):
        end = act_start_pages[j + 1] if j + 1 < len(act_start_pages) else len(pages)
        text = "\n".join(pages[start:end]).strip()
        if text:
            acts.append(RawAct(
                text=text,
                title="",
                page_range=[start, end - 1],
                position_in_gazette=len(acts),
            ))
    return acts


def _split_by_closing(acts: list[RawAct], max_acts: int = 0) -> list[RawAct]:
    """Further split any act that contains multiple closing-signature blocks.

    Each "București, <date>. Nr. <N>." is the end of one act; text after it
    belongs to the next act. This recovers merged acts from single-page-boundary
    segmentation (e.g. PI_820_2007 where the MAI taxi order follows an ANCE
    order without a page break).
    """
    result = []
    pos_offset = 0
    for act in acts:
        closings = list(ACT_CLOSING.finditer(act.text))
        if len(closings) <= 1:
            result.append(act)
            continue
        prev = 0
        for j, m in enumerate(closings):
            part_text = act.text[prev:m.end()].strip()
            if part_text:
                result.append(RawAct(
                    text=part_text,
                    title=act.title if j == 0 else "",
                    page_range=act.page_range,
                    position_in_gazette=act.position_in_gazette + pos_offset + j,
                ))
            prev = m.end()
        # remainder after last closing (next act's header)
        remainder = act.text[prev:].strip()
        if remainder:
            result.append(RawAct(
                text=remainder,
                title="",
                page_range=act.page_range,
                position_in_gazette=act.position_in_gazette + pos_offset + len(closings),
            ))
        pos_offset += len(closings)

    # Over-segmentation guard: if _split_by_closing more than doubled the input
    # count AND the result exceeds the sumar-expected count, the ACT_CLOSING
    # pattern is firing spuriously (e.g. 1989 decree numbers inside act bodies).
    # Revert to the pre-split page-header acts.
    if len(acts) >= 3 and len(result) >= len(acts) * 2 and (max_acts == 0 or len(result) > max_acts):
        return acts
    return result


def _segment_by_delimiter(full_text: str, delimiter: str) -> list[RawAct]:
    parts = full_text.split(delimiter)
    return [
        RawAct(text=p.strip(), title="", page_range=[], position_in_gazette=i)
        for i, p in enumerate(parts) if p.strip()
    ]


_CLOSING_NR = re.compile(r'Nr\.\s*[\d.]+', re.IGNORECASE)


def segment_acts_from_blocks(
    all_page_blocks: list,   # list[list[Block]] — index 0 = cover page, 1+ = body pages
    gazette_year: int,
) -> list[RawAct]:
    """Segment legal acts from role-tagged Block objects.

    Walks the block stream (pages 1+) and opens a new act on 'issuer' or
    'act_type' blocks, closing it on 'place_and_date'/'act_act_number' blocks
    that contain a closing 'Nr. NNN.' pattern.
    """

    acts = []
    current_texts: list[str] = []
    current_pages: list[int] = []
    current_title: str = ""

    ACT_CLOSE_ROLES = {"place_and_date", "act_act_number"}
    # Track whether the last act was formally closed (used to decide if act_type opens a new one)
    last_was_closed = True   # start as if previous act closed, so first act_type opens a new act

    def _flush() -> None:
        nonlocal current_texts, current_pages, current_title, last_was_closed
        if not current_texts:
            return
        text = "\n".join(t for t in current_texts if t.strip())
        if text.strip():
            acts.append(RawAct(
                text=text.strip(),
                title=current_title,
                page_range=sorted(set(current_pages)),
                position_in_gazette=len(acts),
            ))
        current_texts.clear()
        current_pages.clear()
        current_title = ""
        last_was_closed = True

    for page_blocks in all_page_blocks[1:]:   # skip cover/sumar page
        for b in page_blocks:
            # issuer always opens a new act (spec §3.11)
            if b.role == "issuer":
                _flush()
            # act_type opens a new act when: no current act open, OR previous act was closed
            elif b.role == "act_type" and (not current_texts or last_was_closed):
                _flush()

            if b.role not in ("section_banner",):  # section banners don't go in act text
                current_texts.append(b.text)
            current_pages.append(b.page_index)

            # Capture title from the first act_type or act_number block
            if not current_title and b.role in ("act_type", "act_number"):
                current_title = b.text.strip()

            last_was_closed = False

            # Closing: place_and_date or act_act_number containing "Nr. NNN."
            if b.role in ACT_CLOSE_ROLES and _CLOSING_NR.search(b.text):
                _flush()

    _flush()  # final flush

    # Safety: apply closing-block splitting in case any acts are still merged
    if acts:
        acts = _split_by_closing(acts, max_acts=0)

    # Re-index position_in_gazette
    for i, a in enumerate(acts):
        a.position_in_gazette = i

    return acts


def _segment_by_patterns(full_text: str) -> list[RawAct]:
    boundaries = []
    for header in INSTITUTION_HEADERS:
        for m in re.finditer(re.escape(header), full_text, re.IGNORECASE):
            boundaries.append(m.start())
    for m in SPACED_TITLE.finditer(full_text):
        boundaries.append(m.start())

    # For 1989 communiqués and scanned acts without institutional headers:
    # split on act-type headers (COMUNICAT, DECRET-LEGE) first.
    # Only fall back to numbered-item paragraphs when NO act-type header is found,
    # because numbered items are paragraph-level structure WITHIN an act (not act boundaries)
    # and splitting on them in two-column OCR documents causes sentence fragmentation.
    if not boundaries:
        for m in re.finditer(r'(?m)^(Comunicat|COMUNICAT|Decret-lege|DECRET-LEGE)\b', full_text):
            boundaries.append(m.start())
        # Numbered items only as last resort when no act-type header found at all
        if not boundaries:
            for m in re.finditer(r'(?:^|\n\n)(\d{1,2}\.\s+[A-ZĂÂÎȘȚ])', full_text):
                boundaries.append(m.start(1))

    boundaries = sorted(set(boundaries))

    if not boundaries:
        return [RawAct(text=full_text, title="", page_range=[], position_in_gazette=0)]

    acts = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(full_text)
        text = full_text[start:end].strip()
        if text:
            acts.append(RawAct(
                text=text, title="", page_range=[], position_in_gazette=i
            ))
    return acts
