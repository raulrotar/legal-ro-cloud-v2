"""Stage 1 of the Draft-then-Verify pipeline: deterministic rule-based extraction.

Produces a RuleDraft from a MdActBlock using existing regex patterns plus
two new full-document passes:

  1. Full-block closing signature scan — searches the *entire* block for
     "București, DATE. Nr. N." rather than just the last 800 chars.
  2. ORDIN header pattern — catches "ORDIN nr. 346/2007 din 3 dec. 2007"
     which appears in the header of many ministry orders.
  3. Abrogation-number extraction — collects numbers from abrogation clauses
     ("Ordinul nr. 275/2003 ... se abrogă") so the LLM verifier can be
     explicitly told NOT to use them as the act's own number.
  4. Signature-without-number detector — when a minister signature line is
     found but no closing date+nr is present, marks act_number_confidence=low
     and appends a targeted extraction hint.

Confidence semantics
--------------------
  "high"  → strong regex evidence (CLOSING_BLOCK matched, known authority, etc.)
             LLM should only override when it has unambiguous contradicting text.
  "low"   → no strong signal or multiple plausible candidates.
             LLM verifies freely and corrects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from legalro_processing.extract.metadata import (
    CLOSING_BLOCK,
    ACT_TYPE_HEADERS,
    AUTHORITY_PATTERNS,
    _extract_locality,
    _extract_title,
)
try:
    from legalro_processing.extract.roles import strip_letterspacing as _strip_ls
except ImportError:
    def _strip_ls(text: str) -> str: return text  # type: ignore[misc]
from legalro_processing.extract.md_segmenter import MdActBlock

Confidence = Literal["high", "low"]


@dataclass
class RuleDraft:
    """Deterministic extraction result with per-field confidence ratings."""
    doc_type: str
    doc_type_confidence: Confidence
    act_number: str
    act_number_confidence: Confidence
    act_year: Optional[int]
    act_year_confidence: Confidence
    issuing_authority: str
    issuing_authority_confidence: Confidence
    title: str
    locality: Optional[str]
    abrogation_numbers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Patterns ──────────────────────────────────────────────────────────────────

# ORDIN/DECIZIE header: "ORDIN nr. 346/2007 din 3 decembrie 2007"
_ORDIN_HEADER_NR = re.compile(
    r'^\s*ORDIN(?:UL)?\b.*?[Nn][Rr]\.\s*([\d.]+)(?:/(\d{4}))?',
    re.MULTILINE,
)
_DECIZIE_HEADER_NR = re.compile(
    r'^\s*DECIZ(?:IE|IA?)\b.*?[Nn][Rr]\.\s*([\d.]+)(?:/(\d{4}))?',
    re.MULTILINE,
)

# Abrogation/modification: "Ordinul/Decizia nr. NNN/YYYY ... se abrogă/modifică/completează"
# or the reverse order.  All four verbs indicate a *referenced* act, not the current one.
_ABROGATION_NR = re.compile(
    r'[Nn][Rr]\.\s*([\d.]+)/(\d{4})\b.{0,400}?'
    r'se\s+(?:abrog[ăa]|modific[ăa]|complet[eă]|înlocuie[șs]te)'
    r'|se\s+(?:abrog[ăa]|modific[ăa]|complet[eă]|înlocuie[șs]te)'
    r'.{0,400}?[Nn][Rr]\.\s*([\d.]+)/(\d{4})\b',
    re.IGNORECASE | re.DOTALL,
)

# Preamble citations: "în temeiul/baza/potrivit/conform ... nr. NNN" — these are
# *legal basis* references, not the act's own number.
_PREAMBLE_CITATION = re.compile(
    r'(?:în\s+temeiul|în\s+baza|potrivit|conform)\s+.{0,200}?[Nn][Rr]\.\s*([\d.]+)',
    re.IGNORECASE | re.DOTALL,
)

# Standalone "Nr. N." line (not inside a sentence — preceded/followed by newline)
_NR_STANDALONE = re.compile(
    r'(?:^|\n)\s*Nr\.\s*([\d.]+)\.\s*(?:\n|$)',
    re.MULTILINE,
)

# Signature line: "Ministrul/Președintele X, Prenume Nume" with trailing comma
# (indicates the act has a signature block but the date/nr may have been missed)
_SIGNATURE_LINE = re.compile(
    r'(?:^|\n)\s*(?:Ministrul|Ministr(?:ul|a)|Președintele|Directorul|Guvernatorul'
    r'|p\.\s+Ministrul|Secretarul\s+de\s+stat)'
    r'[^\n]{0,120},\s*\n',
    re.IGNORECASE | re.MULTILINE,
)

# Docling orphaned-signature marker injected by _normalize_gazette_md
_ORPHANED_SIG_MARKER = "<!-- legalro:orphaned-signature -->"


# ── Public API ────────────────────────────────────────────────────────────────

def extract_rule_draft(block: MdActBlock, gazette_year: int) -> RuleDraft:
    """Run deterministic extraction on a MdActBlock and return a RuleDraft."""
    md_text   = block.markdown
    plain_text = block.plain_text

    doc_type, doc_type_conf = _extract_doc_type(plain_text)
    act_number, act_year, num_conf = _extract_number_full(
        plain_text, gazette_year, doc_type
    )
    abrogation_numbers = _find_abrogation_numbers(plain_text)
    preamble_numbers   = _find_preamble_numbers(plain_text)
    authority_name, auth_conf = _extract_authority(plain_text)
    title    = block.title_hint or _extract_title(plain_text, doc_type)
    locality = _extract_locality(plain_text) or None

    warnings: list[str] = []

    if abrogation_numbers:
        warnings.append(
            f"abrogation/modification-clause numbers found — must NOT be used as act_number: "
            f"{abrogation_numbers}"
        )
    if preamble_numbers:
        warnings.append(
            f"preamble legal-basis numbers found (în temeiul/baza …) — "
            f"must NOT be used as act_number: {preamble_numbers}"
        )

    if act_number == "0":
        # Check if Docling flagged an orphaned signature (date+Nr. missed in OCR)
        if _ORPHANED_SIG_MARKER in md_text:
            warnings.append(
                "Docling missed the closing date+Nr. block for this act "
                "(orphaned-signature detected) — act_number cannot be recovered "
                "from the current markdown; if the SUMAR provides the number use it, "
                "otherwise return '0'"
            )
        elif _SIGNATURE_LINE.search(plain_text):
            warnings.append(
                "signature line found without accompanying date+Nr — "
                "Docling may have missed the closing block; "
                "search the act text carefully for the final 'Nr. NNN.' line"
            )
        else:
            warnings.append(
                "no closing signature found — act_number unknown; "
                "search the full act text for 'Nr. NNN.' if present"
            )

    return RuleDraft(
        doc_type=doc_type,
        doc_type_confidence=doc_type_conf,
        act_number=act_number,
        act_number_confidence=num_conf,
        act_year=act_year,
        act_year_confidence=num_conf,
        issuing_authority=authority_name,
        issuing_authority_confidence=auth_conf,
        title=title,
        locality=locality,
        abrogation_numbers=abrogation_numbers,
        warnings=warnings,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_doc_type(plain_text: str) -> tuple[str, Confidence]:
    header = plain_text[:800]
    # Normalize letterspacing line-by-line (mirrors metadata.py:116-118).
    # Catches "D E C R E T" → "DECRET" in Docling output for born-digital PDFs.
    header_norm = "\n".join(_strip_ls(line) for line in header.split("\n"))
    for atype, pattern in ACT_TYPE_HEADERS:
        if pattern.search(header_norm):
            return atype, "high"
    if "CADASTRU" in header_norm[:400]:
        return "ORDIN", "high"
    return "UNKNOWN", "low"


def _extract_number_full(
    plain_text: str,
    gazette_year: int,
    doc_type: str,
) -> tuple[str, Optional[int], Confidence]:
    """
    Multi-pass act number extraction across the entire block text.

    Priority:
      1. CLOSING_BLOCK (Budapest + Nr.) anywhere in full plain_text  → high
      2. ORDIN/DECIZIE header pattern in first 600 chars              → high
      3. Standalone Nr. line in last 5000 chars                       → low
      4. "0" — not found                                              → low
    """
    # 1. Standard closing block — scan the ENTIRE block (not just tail)
    matches = list(CLOSING_BLOCK.finditer(plain_text))
    if matches:
        m = matches[-1]
        return m.group(2).replace(".", ""), int(m.group(1)), "high"

    # 2. Header patterns for acts that embed the number in the heading line
    header = plain_text[:600]
    if doc_type in ("ORDIN", "UNKNOWN"):
        m = _ORDIN_HEADER_NR.search(header)
        if m:
            yr = int(m.group(2)) if m.group(2) else gazette_year
            return m.group(1).rstrip("."), yr, "high"
    if doc_type in ("DECIZIE", "DCC", "UNKNOWN"):
        m = _DECIZIE_HEADER_NR.search(header)
        if m:
            yr = int(m.group(2)) if m.group(2) else gazette_year
            return m.group(1).rstrip("."), yr, "high"

    # 3. Standalone Nr. line in last 5000 chars — catches acts where the
    #    signature is mid-document (before a long annex) and CLOSING_BLOCK
    #    fails because the date is on a different line from Nr.
    #    Guard: skip candidates that come from preamble citations (în temeiul/baza …)
    #    since those are legal-basis references, not the act's own number.
    tail = plain_text[-5000:]
    m = _NR_STANDALONE.search(tail)
    if m:
        candidate = m.group(1).replace(".", "")
        preamble_nums = _find_preamble_numbers(plain_text)
        if candidate not in preamble_nums:
            return m.group(1).rstrip("."), gazette_year, "low"

    return "0", gazette_year, "low"


def _find_preamble_numbers(plain_text: str) -> list[str]:
    """Collect act numbers cited in preamble legal-basis clauses.

    These are *references* to enabling legislation (în temeiul Legii nr. X,
    în baza OUG nr. Y, potrivit HG nr. Z) and must NOT be used as the current
    act's own number.
    """
    seen: set[str] = set()
    results: list[str] = []
    for m in _PREAMBLE_CITATION.finditer(plain_text):
        nr = m.group(1).replace(".", "")
        if nr and nr not in seen:
            seen.add(nr)
            results.append(nr)
    return results


def _find_abrogation_numbers(plain_text: str) -> list[str]:
    """Collect number/year pairs from abrogation clauses."""
    seen: set[str] = set()
    results: list[str] = []
    for m in _ABROGATION_NR.finditer(plain_text):
        nr  = (m.group(1) or m.group(3) or "").replace(".", "")
        yr  = m.group(2) or m.group(4) or ""
        key = f"{nr}/{yr}"
        if nr and key not in seen:
            seen.add(key)
            results.append(key)
    return results


def _extract_authority(plain_text: str) -> tuple[str, Confidence]:
    for name, _tag, pattern in AUTHORITY_PATTERNS:
        if pattern.search(plain_text):
            return name, "high"
    return "", "low"
