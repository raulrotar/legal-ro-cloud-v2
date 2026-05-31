"""Coverage audit — the lossless invariant (spec §3.13).

Compares whitespace-stripped character counts: everything in the raw PyMuPDF
text stream must end up either mapped into the structured output or captured in
an "unmapped" catch-all. coverage_ratio == 1.0 means no text was silently lost.

For the pilot this is a LOGGED METRIC + warning (per-PDF, surfaced on the
dashboard), not a hard CI gate — promote to a gate once the corpus is clean.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def _stripped_len(text: str) -> int:
    return len(_WS.sub("", text or ""))


def coverage_ratio(raw_text: str, mapped_texts: list[str], unmapped_texts: list[str]) -> float:
    raw = _stripped_len(raw_text)
    if raw == 0:
        return 1.0
    mapped = sum(_stripped_len(t) for t in mapped_texts)
    unmapped = sum(_stripped_len(t) for t in unmapped_texts)
    return (mapped + unmapped) / raw
