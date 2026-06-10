"""Recover whole acts that Docling dropped from the Markdown.

Docling's layout stage can silently discard column content: MO 74/2017 has
30 decrees in the PDF text layer but only 23 bodies / 17 closing blocks in
the converted Markdown (verified against Docling 2.96 with the heron layout
model).  The born-digital text layer is complete, so the missing acts can be
recovered deterministically:

  1. enumerate closing blocks ("… Nr. N.") in the PDF text layer and in the MD;
  2. for every number present in the PDF but absent from the MD, cut the act
     span out of the text layer (from the previous closing block to the
     missing one);
  3. clean running headers, promote act headings, and append the span to the
     MD with a provenance marker — the segmenter then mints it as a normal
     act block and sumar reconciliation matches it by number.

Appending changes act order, not act identity; identity is what retrieval
and reconciliation key on.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from legalro_core.models import Era


# Closing line: "Nr. 21." / "Nr. 1.415." — standalone or after "București, …"
_CLOSING_RE = re.compile(r"(?m)(?:^|[.,]\s+)Nr\.\s*([\d.]+)\s*\.?\s*$")

# Running page header + bare page numbers (noise inside recovered spans)
_PAGE_HEADER_RE = re.compile(
    r"(?m)^\s*(?:\d{1,3}\s*)?MONITORUL\s+OFICIAL\s+AL\s+ROM[ÂA]NIEI.*$\n?"
)
_BARE_PAGENO_RE = re.compile(r"(?m)^\s*\d{1,3}\s*$\n?")

# Standalone institution/type lines promoted to headings so the segmenter
# recognises an act boundary (mirrors Docling's own output shape).
_HEADING_RE = re.compile(
    r"(?m)^\s*("
    r"PRE[ȘS]EDINTELE\s+ROM[ÂA]NIEI"
    r"|GUVERNUL\s+ROM[ÂA]NIEI"
    r"|D\s*E\s*C\s*R\s*E\s*T(?:\s*-\s*L\s*E\s*G\s*E)?"
    r"|H\s*O\s*T\s*[ĂA]\s*R\s*[ÂA]\s*R\s*E"
    r"|L\s*E\s*G\s*E"
    r"|O\s*R\s*D\s*I\s*N"
    r"|D\s*E\s*C\s*I\s*Z\s*I\s*E"
    r")\s*$"
)


def _digits(nr: str) -> str:
    return re.sub(r"\D", "", nr)


def _fold_ws(text: str) -> str:
    """Drop whitespace and Markdown/reflow artifacts — probe-match form."""
    return re.sub(r"[\s\-–—|*#>_]+", "", text)


def _body_in_md(span: str, md_folded: str) -> bool:
    """True when the act body already exists in the MD (only its closing
    block was dropped).

    Acts in one gazette are often template-identical except for a short
    unique stretch (a person's name) near the end, so ALL probes must match:
    boilerplate probes match other acts' bodies, but the act-unique probe
    only matches when this act's own body is present.  Probes skew toward
    the span end, where the unique content lives.
    """
    flat = _fold_ws(span)
    if len(flat) < 240:
        return _fold_ws(span[:120]) in md_folded
    probes = [flat[int(len(flat) * f): int(len(flat) * f) + 60] for f in (0.45, 0.65, 0.85)]
    return all(p in md_folded for p in probes if p)


def _closing_numbers(text: str) -> list[tuple[str, int, int]]:
    """All closing numbers as (digits, match_start, match_end), in order."""
    out = []
    for m in _CLOSING_RE.finditer(text):
        d = _digits(m.group(1))
        if d and len(d) <= 5:
            out.append((d, m.start(), m.end()))
    return out


def _clean_span(span: str) -> str:
    span = _PAGE_HEADER_RE.sub("", span)
    span = _BARE_PAGENO_RE.sub("", span)
    span = _HEADING_RE.sub(lambda m: "## " + re.sub(r"\s+", " ", m.group(1)).replace(" ", ""), span)
    # un-space "DECRET"-style letterspaced headings collapsed above need a
    # second pass for the multi-word ones
    span = span.replace("## PREȘEDINTELEROMÂNIEI", "## PREȘEDINTELE ROMÂNIEI")
    span = span.replace("## GUVERNULROMÂNIEI", "## GUVERNUL ROMÂNIEI")
    span = span.replace("## DECRET-LEGE", "## DECRET-LEGE")
    return span.strip()


def recover_missing_acts(
    md: str,
    pdf_path: str | Path,
    era: Era,
) -> tuple[str, int, list[str]]:
    """Append acts present in the PDF text layer but missing from the MD.

    Returns (md, n_recovered, recovered_numbers).  No-op for SCANNED era
    (no text layer) and when nothing is missing.
    """
    if era == Era.SCANNED:
        return md, 0, []

    import fitz
    from legalro_core.normalize import normalize_text

    doc = fitz.open(str(pdf_path))
    pdf_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    if len(pdf_text.strip()) < 500:
        return md, 0, []
    pdf_text = normalize_text(pdf_text, era)

    pdf_closings = _closing_numbers(pdf_text)
    md_digits = {d for d, _, _ in _closing_numbers(md)}

    recovered: list[str] = []
    numbers: list[str] = []
    ends = [e for _, _, e in pdf_closings]
    md_folded = _fold_ws(md)

    for d, s, e in pdf_closings:
        # span: end of the previous closing block (any number) → this Nr. line
        prev_end = max((pe for pe in ends if pe < s), default=0)
        span = pdf_text[prev_end:e]
        if not (120 <= len(span) <= 25_000):
            continue
        number_in_md = d in md_digits

        # Act-UNIQUE tail: ~120 folded chars right before the closing
        # boilerplate.  For template-twin acts (identical except a name) this
        # is the only part that distinguishes them, so it decides whether THIS
        # act's body is in the MD — the whole-span probes only see the shared
        # boilerplate.
        # the segment right before the LAST boilerplate match is the body end
        # (the FIRST match may be the act's own opening header)
        _parts = re.split(
            r"PRE[ȘS]EDINTELE\s+ROM[ÂA]NIEI|Bucure[șs]ti,\s*\d{1,2}\s+\w+\s+\d{4}",
            span,
        )
        # the longest segment is the act body (headers/signatures/counter-
        # signatures are short shared boilerplate); its tail holds the
        # act-unique content (e.g. the appointee's name)
        tail_cut = max(_parts, key=len)
        tail = _fold_ws(tail_cut)[-120:]
        tail_present = len(tail) < 60 or tail in md_folded

        if number_in_md:
            # OCR-derived MD (hybrid router) never matches the text layer
            # verbatim — the tail/probe checks would re-append acts that are
            # already present.  All closings being present is the completeness
            # signal there; skip this class entirely.
            if "md-source=glm-ocr" in md:
                continue
            # Number present: recover only when BOTH the unique tail and the
            # body probes are absent (duplicated-body drop).  Tail alone
            # false-positives on acts whose body was reformatted by table
            # triage; probes alone false-negative on template twins.
            if tail_present or _body_in_md(span, md_folded):
                continue
        else:
            # Number missing: the unique tail decides.  If it is present the
            # body survived and only the closing block is missing — that is
            # the fitz_enrich step's job, recovering here would duplicate.
            if tail_present:
                continue

        span = _clean_span(span)
        if len(span) < 120:
            continue
        marker = "" if not number_in_md else " reason=duplicated-body"
        recovered.append(
            f"<!-- legalro:recovered-act nr={d} source=pdf-text-layer{marker} -->\n\n{span}"
        )
        numbers.append(d)

    if not recovered:
        return md, 0, []

    md = md.rstrip() + "\n\n<!-- legalro:page-break -->\n\n" + "\n\n".join(recovered) + "\n"
    print(
        f"[md-recovery] {Path(pdf_path).stem}: recovered {len(recovered)} act(s) "
        f"dropped by Docling (nr: {', '.join(numbers[:10])}{'…' if len(numbers) > 10 else ''})",
        file=sys.stderr, flush=True,
    )
    return md, len(recovered), numbers
