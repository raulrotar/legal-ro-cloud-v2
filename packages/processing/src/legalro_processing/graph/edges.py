"""Graph edge extraction from legal act text — spec §3.12.

Extracts citations (cites), promulgation chains, and approval links
using deterministic regex. No LLM, no external deps.
"""
from __future__ import annotations
import re

CITATION_RE = re.compile(
    r"\b(?P<type>Lege(?:a)?|Ordonan[țt]a\s+(?:de\s+urgen[țt][ăa]\s+a\s+)?Guvernului|"
    r"Hot[ăa]r[âa]rea\s+Guvernului|Decretul|Decizia\s+Cur[țt]ii\s+Constitu[țt]ionale)"
    r"\s+nr\.\s*(?P<num>[\d\.]+)\s*/\s*(?P<year>\d{4})"
    r"(?:\s*,\s*art\.\s*(?P<art>\d+(?:\s*alin\.\s*\(\d+\))?))?",
    re.UNICODE | re.IGNORECASE,
)

PROMULGATES_RE = re.compile(
    r"pentru\s+promulgarea\s+Legii\s+(?:privind|pentru|nr\.\s*[\d./]+)",
    re.IGNORECASE,
)

_TYPE_MAP = {
    "lege": "lege",
    "legea": "lege",
    "ordonanta": "ordonanta",
    "ordonanța": "ordonanta",
    "hotararea": "hotarare_guvern",
    "hotărârea": "hotarare_guvern",
    "decretul": "decret",
    "decizia": "decizie",
}


def _canonical_type(raw: str) -> str:
    key = raw.strip().lower().split()[0]
    return _TYPE_MAP.get(key, "other")


def extract_citations(text: str) -> list[dict]:
    """Return list of {type, number, year, article, raw} dicts from act text."""
    results = []
    for m in CITATION_RE.finditer(text):
        results.append({
            "type": _canonical_type(m.group("type")),
            "number": m.group("num").replace(".", ""),
            "year": int(m.group("year")),
            "article": m.group("art") or None,
            "raw": m.group(0),
        })
    return results
