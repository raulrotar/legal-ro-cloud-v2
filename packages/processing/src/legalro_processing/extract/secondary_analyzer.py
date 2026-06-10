"""Secondary PDF analyzer for closing-signature recovery.

When Docling drops a closing signature block (e.g. "București, DATE. Nr. NNN.")
during layout/reading-order reconstruction near long annexes, this module
extracts that information directly from the PDF's text layer.

Architecture: SecondaryAnalyzer protocol + FitzAnalyzer default implementation.
Pluggable upgrades (OCR, cloud) behind the same interface.

Usage in the extraction pipeline
---------------------------------
    from legalro_processing.extract.secondary_analyzer import FitzAnalyzer

    recovered = FitzAnalyzer().recover_closing_numbers(pdf_path)
    # → [ClosingSig(page_no=6, number='356', context='...Cristian David...')]
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ClosingSig:
    """A recovered closing signature from the source PDF."""
    page_no: int           # 1-based page number
    number: str            # the Nr. value, digits only (e.g. "356")
    context: str           # ~300 chars of page text before/around Nr. line


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class SecondaryAnalyzer(Protocol):
    def recover_closing_numbers(self, pdf_path: str | Path) -> list[ClosingSig]:
        """Return all closing signatures found in the PDF by this analyzer."""
        ...


# ── Patterns ──────────────────────────────────────────────────────────────────

# Full closing block: "București, D LUNA YYYY. Nr. NNN."
_CLOSING_BLOCK = re.compile(
    r'Bucure[șs]ti,\s+\d{1,2}\s+\w+\s+\d{4}\.?\s*[\s\S]{0,150}?Nr\.\s*([\d.]+)\.',
    re.MULTILINE,
)

# Standalone "Nr. NNN." on its own line (catches signatures split across lines)
_NR_STANDALONE = re.compile(
    r'(?:^|\n)\s*Nr\.\s*([\d.]+)\.\s*(?:\n|$)',
    re.MULTILINE,
)

# Masthead number pattern — the gazette's own issue number appears on first page
# in the form "Nr. NNN" near "Monitorul Oficial", used to exclude it from results
_MASTHEAD_NR = re.compile(r'Nr\.\s*([\d]+)', re.IGNORECASE)


# ── FitzAnalyzer (default) ────────────────────────────────────────────────────

class FitzAnalyzer:
    """Recover closing signatures from the PDF text layer using PyMuPDF.

    Works for born-digital PDFs where Docling drops content during reading-order
    reconstruction near long annexes. Near-zero memory overhead (~<100 MB) and
    very fast (~180 pages/s). Scanned-era PDFs with no text layer return [].
    """

    def recover_closing_numbers(self, pdf_path: str | Path) -> list[ClosingSig]:
        """Scan all pages for closing signature patterns and return unique hits."""
        try:
            import fitz  # pymupdf
        except ImportError:
            return []

        path = str(pdf_path)
        results: list[ClosingSig] = []
        masthead_nr: str | None = None

        doc = fitz.open(path)
        try:
            # Extract the gazette's own issue number from the first page so we
            # can exclude it from results (it appears in the masthead on every page)
            if doc.page_count > 0:
                first_text = doc[0].get_text("text")
                m = _MASTHEAD_NR.search(first_text[:500])
                if m:
                    masthead_nr = m.group(1)

            for pg_idx in range(doc.page_count):
                page_text = doc[pg_idx].get_text("text")
                page_no = pg_idx + 1  # 1-based

                sigs = _extract_sigs_from_text(page_text, page_no, masthead_nr)
                results.extend(sigs)
        finally:
            # Release Apple-Silicon fitz cache (API varies by pymupdf version)
            try:
                fitz.TOOLS.store_shrink(10)  # pymupdf ≥1.24
            except AttributeError:
                try:
                    fitz.Tools().store_shrink(10)  # older pymupdf
                except Exception:
                    pass
            doc.close()

        # Deduplicate by number — keep first occurrence (earliest page)
        seen: set[str] = set()
        deduped: list[ClosingSig] = []
        for sig in results:
            if sig.number not in seen:
                seen.add(sig.number)
                deduped.append(sig)

        return deduped


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_sigs_from_text(
    page_text: str,
    page_no: int,
    masthead_nr: str | None,
) -> list[ClosingSig]:
    """Extract closing signatures from a single page's text."""
    results: list[ClosingSig] = []

    # Try full closing block pattern first (highest confidence)
    for m in _CLOSING_BLOCK.finditer(page_text):
        nr = m.group(1).replace(".", "").strip()
        if not nr or nr == masthead_nr:
            continue
        # context = text before the match (~300 chars) for signatory matching
        ctx_start = max(0, m.start() - 300)
        context = page_text[ctx_start:m.end()]
        results.append(ClosingSig(page_no=page_no, number=nr, context=context))

    # Fallback: standalone Nr. line not captured by closing block
    for m in _NR_STANDALONE.finditer(page_text):
        nr = m.group(1).replace(".", "").strip()
        if not nr or nr == masthead_nr:
            continue
        # Skip if already found via closing block on this page
        if any(s.number == nr and s.page_no == page_no for s in results):
            continue
        ctx_start = max(0, m.start() - 300)
        context = page_text[ctx_start:m.end()]
        results.append(ClosingSig(page_no=page_no, number=nr, context=context))

    return results


def enrich_markdown_with_fitz(md: str, recovered: list[ClosingSig]) -> tuple[str, int]:
    """Inject missing closing blocks into Docling markdown BEFORE segmentation.

    For each recovered sig whose Nr. is absent from the markdown, this function:
      1. Extracts the full 'București, DATE.\\nNr. NN.' text from the fitz context.
      2. Finds the signatory name in the context (last token before 'București').
      3. Locates the first occurrence of that name in the markdown that is NOT
         already followed by 'Nr.' within 500 chars (case-sensitive — body text
         references use lowercase 'nr.').
      4. Injects the closing block at that position.

    Sigs are processed in page order so positional ordering is preserved for
    gazettes with multiple acts signed by the same person.

    Returns (enriched_md, injection_count).
    """
    injections = 0

    for sig in recovered:
        # Skip if this number is already in the markdown
        if re.search(rf'\bNr\.\s*{re.escape(sig.number)}\.', md):
            continue

        closing = _extract_closing_block(sig.context, sig.number)
        if not closing:
            continue

        # Primary: anchor on the act-UNIQUE text right before the closing
        # block (e.g. the appointment sentence with the person's name).  The
        # signatory name is useless as an anchor when every act in the issue
        # is signed by the same person — injections then land on the wrong
        # acts and shuffle the numbers.
        inject_pos = _find_injection_by_context(md, sig.context)

        if inject_pos is None:
            name_hint = _extract_name_hint(sig.context)
            if not name_hint:
                continue
            inject_pos = _find_injection_point(md, name_hint)
        if inject_pos is None:
            continue

        md = md[:inject_pos] + "\n" + closing + "\n" + md[inject_pos:]
        injections += 1

    return md, injections


_FOLD_DROP_RE = re.compile(r"[\s\-–—|*#>_]")


def _fold_with_map(text: str) -> tuple[str, list[int]]:
    """Fold text for fuzzy matching (drop whitespace/markup) keeping a map
    from folded index → raw index."""
    chars: list[str] = []
    idx: list[int] = []
    for i, c in enumerate(text):
        if not _FOLD_DROP_RE.match(c):
            chars.append(c)
            idx.append(i)
    return "".join(chars), idx


def _find_injection_by_context(md: str, context: str, probe_len: int = 110) -> int | None:
    """Locate the act this closing belongs to via the unique pre-closing text.

    Takes the last `probe_len` folded chars of the context BEFORE the
    signature/closing boilerplate and finds them in the folded markdown.
    Returns the raw offset after that line, or None.
    """
    # cut the context at the start of the closing boilerplate
    cut = re.split(
        r"PRE[ȘS]EDINTELE\s+ROM[ÂA]NIEI|Bucure[șs]ti,\s*\d{1,2}\s+\w+\s+\d{4}",
        context,
    )[0]
    probe_src, _ = _fold_with_map(cut)
    if len(probe_src) < 40:
        return None
    probe = probe_src[-probe_len:]

    md_folded, idx_map = _fold_with_map(md)
    pos = md_folded.find(probe)
    if pos == -1 or md_folded.find(probe, pos + 1) != -1:
        return None  # absent or ambiguous — let the name-hint fallback decide
    raw_end = idx_map[min(pos + len(probe) - 1, len(idx_map) - 1)]
    # don't double-inject when a closing Nr. already follows closely
    if re.search(r"\bNr\.\s*\d", md[raw_end:raw_end + 250]):
        return None
    line_end = md.find("\n", raw_end)
    return (line_end + 1) if line_end != -1 else len(md)


def _extract_closing_block(context: str, number: str) -> str | None:
    """Extract 'București, DATE.\\nNr. NN.' from a fitz context string."""
    m = re.search(
        r'(Bucure[șs]ti,\s*\d{1,2}\s+\w+\s+\d{4}\.?\s*[\n\r]+\s*Nr\.\s*[\d.]+\.)',
        context,
        re.MULTILINE,
    )
    if m:
        return m.group(1).strip()
    # Fallback: single-line closing block
    m2 = re.search(
        r'(Bucure[șs]ti,\s*\d{1,2}\s+\w+\s+\d{4}\.?\s*Nr\.\s*[\d.]+\.)',
        context,
    )
    if m2:
        return m2.group(1).strip()
    # Last resort: just inject the Nr. line
    return f"Nr. {number}."


_BODY_TEXT_HINT = re.compile(
    r'judec[ăa]tor|Judec[ăa]tori[ae]|func[țt]i[ae]|sectorului|tribunalul|curtea\s+de'
    r'|se\s+nume[șs]te|în\s+func[țt]ia|art\.\s*\d|alin\.\s*\(',
    re.IGNORECASE,
)


def _extract_name_hint(context: str) -> str:
    """Extract the signatory name from the line just before 'București' in context.

    Skips attribution title lines (Ministrul, Președintele, Contrasemnează…)
    and body-text lines (judge appointment sentences, article references) —
    these appear between the signatory and the date in Romanian legal gazettes.
    """
    parts = re.split(r'Bucure[șs]ti', context)
    if not parts:
        return ""
    pre = parts[0]
    for line in reversed(pre.strip().splitlines()):
        line = line.strip().rstrip(',;.')
        if not line or len(line) < 4:
            continue
        if re.search(
            r'Ministrul|Ministr|Președintele|Contrasemnează|secretar\s+de\s+stat'
            r'|PRIM-?MINISTR|Guvernatorul|Directorul',
            line, re.IGNORECASE
        ):
            continue
        # Skip body-text sentences (judge appointments, article refs)
        if _BODY_TEXT_HINT.search(line):
            continue
        return line
    return ""


def _find_injection_point(md: str, name_hint: str, window: int = 500) -> int | None:
    """Find the first position after `name_hint` in `md` where 'Nr.' does not
    appear within `window` chars — i.e. the next unmatched signature occurrence.

    Returns the character offset just after the end of the matched line, ready
    for injection, or None if no suitable position is found.
    """
    pattern = re.compile(re.escape(name_hint), re.IGNORECASE)
    for m in pattern.finditer(md):
        section = md[m.end():m.end() + window]
        # Capital-N 'Nr.' — body references use lowercase 'nr.'
        if not re.search(r'\bNr\.\s*\d', section):
            # Inject after the end of this line
            line_end = md.find('\n', m.end())
            return (line_end + 1) if line_end != -1 else m.end()
    return None


def match_recovered_number(
    recovered: list[ClosingSig],
    signatory_hint: str,
    page_hints: list[int],
    abrogation_numbers: list[str],
) -> str | None:
    """Find the best recovered number for an act given its signatory and pages.

    Returns the recovered number string if exactly one non-ambiguous match is found,
    or None if zero or multiple matches (bail on ambiguity — never guess).

    Args:
        recovered: all ClosingSig recovered from the gazette PDF.
        signatory_hint: surname/name from the act's signature line (e.g. "Cristian David").
        page_hints: page numbers the act spans (from MdActBlock.page_hints).
        abrogation_numbers: list of "nr/year" strings that are confirmed abrogation refs.
    """
    if not recovered:
        return None

    abrogation_nrs = {n.split("/")[0] for n in abrogation_numbers}

    # Candidate: sig is on one of the act's pages AND (signatory matches OR no page hint)
    candidates: list[ClosingSig] = []
    for sig in recovered:
        if sig.number in abrogation_nrs:
            continue  # never a real match — it's an abrogation ref
        # Page overlap check (if page_hints available)
        if page_hints and sig.page_no not in page_hints:
            # Allow ±2 page tolerance — page_hints come from stray digit detection
            # which is approximate; long-annex acts have page_hints pointing into
            # the annex body rather than the act's opening/closing pages.
            adjacent = any(abs(sig.page_no - ph) <= 2 for ph in page_hints)
            if not adjacent:
                continue
        # Signatory match: any token of signatory_hint in the context window
        if signatory_hint:
            tokens = [t for t in signatory_hint.split() if len(t) > 3]
            # ALL tokens must appear in the context window — "any" causes false
            # positives when two signatories share a common first name (e.g.
            # "Cristian Munteanu" vs "Cristian David" both match on "Cristian").
            if tokens and not all(t.lower() in sig.context.lower() for t in tokens):
                continue
        candidates.append(sig)

    if len(candidates) == 1:
        return candidates[0].number
    # Zero or multiple → ambiguous, leave draft unchanged
    return None


def find_candidates(
    recovered: list[ClosingSig],
    signatory_hint: str,
    page_hints: list[int],
    abrogation_numbers: list[str],
) -> list[ClosingSig]:
    """Return all ClosingSig candidates that match the given signatory/page filters.

    Same filtering logic as match_recovered_number, but returns the full candidate
    list sorted by page number so the caller can apply a positional tiebreaker
    (e.g. "this is the 2nd act signed by Munteanu, give me the 2nd Munteanu sig").
    """
    if not recovered:
        return []

    abrogation_nrs = {n.split("/")[0] for n in abrogation_numbers}
    candidates: list[ClosingSig] = []

    for sig in recovered:
        if sig.number in abrogation_nrs:
            continue
        if page_hints and sig.page_no not in page_hints:
            adjacent = any(abs(sig.page_no - ph) <= 2 for ph in page_hints)
            if not adjacent:
                continue
        if signatory_hint:
            tokens = [t for t in signatory_hint.split() if len(t) > 3]
            if tokens and not all(t.lower() in sig.context.lower() for t in tokens):
                continue
        candidates.append(sig)

    return sorted(candidates, key=lambda s: s.page_no)
