"""Pre-embedding cleanup (Phase 0 of docs/EMBEDDINGS_PLAN.md).

The extraction evaluator found three classes of embedding hazards that are
cheapest to remove at build time, before chunking:

  1. body pollution — running page headers, literal page-break markers, and
     the gazette colophon glued to the last act of nearly every issue;
  2. OCR repetition-loop acts (pure embedding poison, e.g. 9.7 KB of
     "NU. NU. NU…") — their pages already failed the verify gate;
  3. near-duplicate acts (a 135 KB annex embedded twice would dominate
     BM25 statistics for the whole corpus).
"""
from __future__ import annotations

import re

from legalro_processing.extract.ocr_verify import repeated_shingle


_POLLUTION_RES = [
    # running page header, any case/spacing, with trailing issue ref
    re.compile(r"(?im)^\s*\d*\s*MONITORUL\s+OFICIAL\s+AL\s+ROM[ÂA]NIEI[^\n]*$\n?"),
    re.compile(r"(?im)^\s*Monitorul\s+Oficial\s+al\s+Rom[âa]niei,\s*Partea\s+I[^\n]*$\n?"),
    # literal page-break artifacts that leaked out of comments
    re.compile(r"(?im)^\s*(?:<!--\s*legalro:)?page-break(?:\s*-->)?\s*$\n?"),
    re.compile(r"<!--\s*legalro:[^>]*-->\n?"),
    # colophon tail: EDITOR line through end of text (price/ISSN/abonamente block)
    re.compile(
        r"(?is)\n[^\n]{0,10}EDITOR\s*:\s*(?:PARLAMENTUL|CONSILIUL|MONITORUL)"
        r".{0,2500}\Z"
    ),
    # bare page numbers on their own line
    re.compile(r"(?m)^\s*\d{1,3}\s*$\n?"),
]


def clean_act_text(text: str) -> str:
    """Strip body pollution before chunking/embedding. Idempotent."""
    for rx in _POLLUTION_RES:
        text = rx.sub("", text)
    # collapse adjacent duplicate blocks (OCR loop residue) instead of
    # quarantining the whole act — content survives once
    from legalro_processing.extract.ocr_verify import collapse_repeated_line_blocks
    text = collapse_repeated_line_blocks(text)
    # collapse the blank-line craters left behind
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_embedding_poison(text: str) -> bool:
    """True for OCR repetition-loop bodies that must not be embedded.

    Two detectors: tandem block repetition (colophon ×3) and degenerate
    lexical diversity (tiny-period loops like "NU. NU. NU…" whose repeated
    unit is too short for the shingle detector)."""
    # NOTE: no tandem-shingle check here — legal annexes legitimately repeat
    # section templates ("Competiții sportive interne:" blocks), and adjacent
    # identical copies were already collapsed by clean_act_text.  Only
    # degenerate vocabulary marks true OCR garbage.
    # Diversity must be measured on a FIXED window: type-token ratio falls
    # naturally with length, so a whole-document ratio would flag any long
    # legitimate annex (a 136K-char methodology is not a loop).
    words = re.findall(r"[a-zăâîșț]{2,}", text.lower())
    if len(words) <= 150:
        return False
    window = words[:1000]
    return len(set(window)) / len(window) < 0.12


def _shingles(text: str, n: int = 8) -> set[str]:
    """Word n-grams — position-independent, so shifted near-copies still
    overlap (char shingles sampled at fixed offsets miss aligned content)."""
    words = re.findall(r"[a-zăâîșț0-9]{2,}", text.lower())
    return {" ".join(words[i:i + n]) for i in range(0, max(1, len(words) - n), 2)}


def near_duplicate_act_indices(acts: list, threshold: float = 0.55,
                               min_chars: int = 5000) -> set[int]:
    """Indices of acts whose body near-duplicates an EARLIER act
    (shingle-overlap containment >= threshold).  The shorter copy loses.
    Catches duplicates the number-keyed dedup cannot (different act numbers,
    e.g. the MO_PI_75 annex embedded under both nr 1640 and nr 115)."""
    # min_chars guard: template-twin decrees share >55% boilerplate but are
    # short and LEGITIMATE — only annex-scale bodies can near-duplicate.
    # threshold 0.55 because the duplicate copies come from different
    # extraction passes (Docling vs recovery) and differ at word level
    # (measured containment 0.61–0.67 on the MO_PI_75 pair).
    texts = [getattr(a, "full_text", "") or "" for a in acts]
    sh = [_shingles(t, n=4) if len(t) >= min_chars else set() for t in texts]
    drop: set[int] = set()
    for i in range(len(acts)):
        if i in drop or len(sh[i]) < 8:
            continue
        for j in range(i + 1, len(acts)):
            if j in drop or len(sh[j]) < 8:
                continue
            inter = len(sh[i] & sh[j])
            contain = inter / min(len(sh[i]), len(sh[j]))
            if contain >= threshold:
                # keep the longer body
                loser = i if len(sh[i]) < len(sh[j]) else j
                drop.add(loser)
                if loser == i:
                    break
    return drop
