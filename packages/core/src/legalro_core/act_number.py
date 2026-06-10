"""Act-number domain constants and helpers.

Single source of truth for:
  - NO_NUMBER_DOC_TYPES   — doc types that legitimately have no act number
  - is_malformed_act_number() — detect structurally invalid act_number strings

These live in core so both the processing extractor (pipeline.py) and the
validator (extraction_validator.py) can share them without duplication.
No ML or heavy deps — only stdlib re.
"""
from __future__ import annotations

import re

# Doc types that legitimately carry no own act number (communiqués, notices,
# corrections).  Used both by the validator (downgrade ACT_NUMBER_ZERO to INFO)
# and by the per-act retry trigger (skip retry when number is absent by design).
NO_NUMBER_DOC_TYPES: frozenset[str] = frozenset({
    "COMUNICAT",
    "RECTIFICARE",
    "ANUNT",
    "ANUNȚ",
})

# ── Malformed-number detection ────────────────────────────────────────────────
# A well-formed act_number is one of:
#   (a) a plain integer string, optionally with Romanian thousands-separator
#       dots: "576", "1027", "1.027", "1.642"  → clean
#   (b) a clean number (possibly with thousands dots) followed by a single
#       /YEAR suffix: "699/2024", "1.642/2016"  → clean (number + year)
# Anything that doesn't fit either form is considered malformed.

# Matches a plain integer or a Romanian thousands-dotted integer
# e.g. "576", "1027", "1.027", "20.022", "1.642"
_PLAIN_INT = re.compile(r"^\d{1,3}(\.\d{3})*$")

# Matches (plain_int)/YEAR where YEAR is a 4-digit plausible calendar year
_WITH_YEAR = re.compile(r"^(\d{1,3}(\.\d{3})*)/(\d{4})$")


def is_malformed_act_number(raw: str) -> tuple[bool, str]:
    """Return (is_malformed, reason).

    Returns (False, "") for values that are syntactically valid act numbers,
    including the zero/empty sentinels which are handled by separate validator
    checks.  Returns (True, reason) for structurally unexpected values such as
    embedded dates ("162 din 20 decembrie 2006"), stray letter prefixes
    ("E 356"), or composite refs with extra path components ("2.412/C/2013").

    Conservative: anything ambiguous is treated as clean to avoid false
    positives during the transition period.
    """
    s = raw.strip()

    # Sentinels handled elsewhere — never flag them here
    if s in ("", "0"):
        return False, ""

    # Known textual placeholders — handled by ACT_NUMBER_PLACEHOLDER
    if s.lower() in {"necunoscut", "unknown", "lipsă", "lipsa", "n/a"}:
        return False, ""

    # (a) plain integer / thousands-dotted integer → clean
    if _PLAIN_INT.match(s):
        return False, ""

    # (b) plain/YEAR → clean
    if _WITH_YEAR.match(s):
        return False, ""

    # Heuristics for common malformed patterns (in order of specificity)

    # "din" date suffix: "162 din 20 decembrie 2006"
    if re.search(r'\bdin\b', s, re.IGNORECASE):
        return True, f"act_number contains a 'din <date>' phrase: {s!r}"

    # Extra slash segments: "2.412/C/2013", "1234/2007/extra"
    slash_parts = s.split("/")
    if len(slash_parts) > 2:
        return True, f"act_number has multiple '/' segments: {s!r}"
    if len(slash_parts) == 2:
        left, right = slash_parts
        # Right side should be a 4-digit year; if not, it's a composite ref
        if not re.match(r"^\d{4}$", right.strip()):
            return True, f"act_number slash suffix is not a plain year: {s!r}"
        # Left side should be a plain/thousands int
        if not _PLAIN_INT.match(left.strip()):
            return True, f"act_number has non-numeric left side before '/': {s!r}"

    # Stray leading/trailing non-numeric characters (e.g. "E 356", "Nr. 12")
    if re.match(r'^[A-Za-zÀ-ÿ]', s):
        return True, f"act_number starts with a letter: {s!r}"

    # Catch-all: contains whitespace (likely a phrase, not a number)
    if re.search(r'\s', s):
        return True, f"act_number contains whitespace: {s!r}"

    return False, ""
