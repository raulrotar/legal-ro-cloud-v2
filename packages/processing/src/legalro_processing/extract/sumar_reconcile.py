"""Sumar ↔ acts reconciliation: the table of contents as a completeness oracle.

The gazette SUMAR lists every act with number, type, title and page range.
Treating it as ground truth for ENUMERATION (never for content) makes both
silent failure modes detectable:

  - a sumar entry with no matching extracted act  → MISSING act
    (OCR dropped a page, segmenter merged two acts, …)
  - an extracted act with no matching sumar entry → PHANTOM candidate
    (segmentation artifact, table row minted as an act, …)

It also fixes the empty-title defect: the body of an act often carries only
the bare heading ("DECRET") while the sumar holds the real title ("Decret
pentru numirea unui judecător") — a matched act with a generic title gets
the sumar title backfilled.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# Titles that are just the document-type word — not real titles.
_GENERIC_TITLES = {
    "", "DECRET", "DECRETLEGE", "LEGE", "HOTARARE", "ORDIN", "DECIZIE",
    "ORDONANTA", "ORDONANTADEURGENTA", "COMUNICAT", "RAPORT", "CIRCULARA",
    "REGULAMENT", "NORMA", "NORME", "ANEXA", "RECTIFICARE", "UNKNOWN",
}


def _fold(text: str) -> str:
    """Uppercase, strip diacritics and non-alphanumerics — match key form."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _norm_nr(nr: str | None) -> str:
    """Canonical act-number form: '1.415/2006' → '1415', '020' → '20'."""
    nr = str(nr or "").split("/")[0]
    nr = re.sub(r"[^0-9A-Za-z]", "", nr)
    return nr.lstrip("0") or ("0" if nr else "")


# Masthead line minted as a sumar entry/title by the parser:
# "Anul 175 (XIX) — Nr. 2" — never a real act title.
_MASTHEAD_TITLE = re.compile(r"^\s*Anul\s+[IVXLCDM\d]+\s*\(", re.IGNORECASE)


def is_generic_title(title: str | None) -> bool:
    t = title or ""
    return _fold(t) in _GENERIC_TITLES or bool(_MASTHEAD_TITLE.match(t))


_JUNK_SUMAR_TITLE = re.compile(
    r"^\s*(Luni|Mar[țt]i|Miercuri|Joi|Vineri|S[âa]mb[ăa]t[ăa]|Duminic[ăa])\b[,\s]",
    re.IGNORECASE,
)


def _is_junk_entry(entry) -> bool:
    title = str(getattr(entry, "title", "") or "")
    return bool(_JUNK_SUMAR_TITLE.match(title) or _MASTHEAD_TITLE.match(title))


@dataclass
class ReconcileReport:
    act_to_sumar: dict[int, int] = field(default_factory=dict)  # act_index → sumar idx
    missing_sumar: list[int] = field(default_factory=list)      # sumar idx with no act
    unmatched_acts: list[int] = field(default_factory=list)     # act_index not in sumar
    titles_backfilled: int = 0
    warnings: list[str] = field(default_factory=list)


def reconcile(acts: list, sumar_entries: list,
              legacy_junk_filter: bool = False) -> ReconcileReport:
    """Match extracted acts to sumar entries and report discrepancies.

    Matching passes (an act/entry is consumed by the first pass that claims it):
      1. (doc_type, number) exact — unique pairs only
      2. number-only — unique pairs only
      3. positional alignment of the remaining unmatched, in document order,
         accepted only when doc_type is compatible (equal or one side unknown)

    legacy_junk_filter: the SCANNED-era compatibility mode. Historically junk
    entries (weekday lines) were dropped by REBINDING the list, which shifted
    the sj indices that backfill_titles/repair_numbers later use against the
    caller's original list — effectively randomizing those reads. The 1989
    extraction baseline was validated WITH that behavior (it suppresses
    unreliable OCR sumar-title backfill), so the scanned era keeps it until
    the scanned-era sumar work lands. All other eras get correct indices.
    """
    report = ReconcileReport()
    if not sumar_entries or not acts:
        return report
    if legacy_junk_filter:
        sumar_entries = [
            e for e in sumar_entries
            if not _JUNK_SUMAR_TITLE.match(str(getattr(e, "title", "") or ""))
        ]
        if not sumar_entries:
            return report
        junk_s: set[int] = set()
    else:
        # Junk entries (weekday/date lines, masthead) are excluded from
        # matching by pre-claiming their slots — NOT by filtering the list,
        # which would shift the sj indices used downstream.
        junk_s = {sj for sj, e in enumerate(sumar_entries) if _is_junk_entry(e)}

    def _type_key(obj) -> str:
        t = _fold(getattr(obj, "doc_type", ""))
        # "ACT" is the sumar parser's generic placeholder, "UNKNOWN" the
        # extractor's — both act as wildcards, like an empty type
        return "" if t in ("ACT", "UNKNOWN") else t

    s_keys = [
        (_type_key(e), _norm_nr(getattr(e, "act_number", "")))
        for e in sumar_entries
    ]
    a_keys = [
        (_type_key(a), _norm_nr(getattr(a, "act_number", "")))
        for a in acts
    ]

    matched_s: set[int] = set(junk_s)  # junk slots can never be claimed
    matched_a: set[int] = set()

    def _claim(ai: int, sj: int) -> None:
        report.act_to_sumar[getattr(acts[ai], "act_index", ai)] = sj
        matched_a.add(ai)
        matched_s.add(sj)

    # pass 1: (type, number) — only when the pair is unique on both sides
    for ai, ak in enumerate(a_keys):
        if not ak[1]:
            continue
        hits = [sj for sj, sk in enumerate(s_keys) if sj not in matched_s and sk == ak]
        if len(hits) == 1 and a_keys.count(ak) == 1:
            _claim(ai, hits[0])

    # pass 2: number only
    for ai, ak in enumerate(a_keys):
        if ai in matched_a or not ak[1]:
            continue
        hits = [sj for sj, sk in enumerate(s_keys) if sj not in matched_s and sk[1] == ak[1]]
        nrs = [k[1] for k in a_keys]
        if len(hits) == 1 and nrs.count(ak[1]) == 1:
            _claim(ai, hits[0])

    # pass 2.5: type-unique pairing — when exactly one unmatched act and one
    # unmatched sumar entry share a (non-wildcard) doc_type, pair them even if
    # the act's number is wrong (duplicate-number acts get repaired later)
    for t in {k[0] for k in a_keys if k[0]}:
        a_hits = [ai for ai in range(len(acts)) if ai not in matched_a and a_keys[ai][0] == t]
        s_hits = [sj for sj in range(len(sumar_entries)) if sj not in matched_s and s_keys[sj][0] == t]
        if len(a_hits) == 1 and len(s_hits) == 1:
            _claim(a_hits[0], s_hits[0])

    # pass 3: positional alignment of leftovers with compatible doc_type
    rest_a = [ai for ai in range(len(acts)) if ai not in matched_a]
    rest_s = [sj for sj in range(len(sumar_entries)) if sj not in matched_s]
    for ai, sj in zip(rest_a, rest_s):
        at, st = a_keys[ai][0], s_keys[sj][0]
        if not at or not st or at == st:
            _claim(ai, sj)

    report.missing_sumar = [sj for sj in range(len(sumar_entries)) if sj not in matched_s]
    report.unmatched_acts = [
        getattr(acts[ai], "act_index", ai) for ai in range(len(acts)) if ai not in matched_a
    ]

    for sj in report.missing_sumar:
        e = sumar_entries[sj]
        report.warnings.append(
            f"sumar_reconcile: MISSING act — sumar[{sj}] "
            f"{getattr(e, 'doc_type', '?')} nr={getattr(e, 'act_number', '?')!r} "
            f"p.{getattr(e, 'page_start', '?')} {str(getattr(e, 'title', ''))[:80]!r} "
            f"has no matching extracted act"
        )
    for idx in report.unmatched_acts:
        a = next((x for x in acts if getattr(x, "act_index", None) == idx), None)
        if a is not None:
            report.warnings.append(
                f"sumar_reconcile: act[{idx}] {getattr(a, 'doc_type', '?')} "
                f"nr={getattr(a, 'act_number', '?')!r} not found in sumar (phantom candidate)"
            )
    return report


# Doc-type token → human title prefix (sumar style: "Decret pentru numirea…")
_TYPE_WORDS = {
    "DECRET": "Decret", "DECRET_LEGE": "Decret-lege", "DECRET-LEGE": "Decret-lege",
    "LEGE": "Lege", "HOTARARE": "Hotărâre", "HG": "Hotărâre", "ORDIN": "Ordin",
    "DECIZIE": "Decizie", "COMUNICAT": "Comunicat", "OUG": "Ordonanță de urgență",
    "ORDONANȚĂ": "Ordonanță", "RAPORT": "Raport",
}

_TITLE_START = re.compile(
    r"^(privind|pentru|cu privire la|referitoare? la|prin care)\b", re.IGNORECASE
)
# Body-opening phrases — both as full-line starts and run on inside the title
# line (OCR often merges "…title Consiliul Frontului… decretează:" into one line)
_BODY_PHRASE = (
    r"Consiliul\b|În temeiul\b|Av[âî]nd în vedere\b|Art\.\s|Articol\b"
    r"|PRE[ȘS]EDINTELE\b|GUVERNUL\b|Parlamentul\b|CURTEA\b|decret[ăe]a?z[ăa]"
)
_BODY_START = re.compile(r"^(" + _BODY_PHRASE + r")", re.IGNORECASE)
_BODY_INLINE = re.compile(r"\s+(" + _BODY_PHRASE + r").*$", re.IGNORECASE)


def backfill_title_from_body(act) -> bool:
    """Derive a title from the act body when no sumar title is available.

    Gazette bodies open with the doc-type heading followed by the title line
    ("privind …" / "pentru …"); reconstruct "Decret privind …" from it.
    Returns True when a title was set.
    """
    if not is_generic_title(getattr(act, "title", "")):
        return False
    lines = [l.strip() for l in (getattr(act, "full_text", "") or "").splitlines() if l.strip()]
    for idx, line in enumerate(lines[:6]):
        if not _TITLE_START.match(line):
            continue
        title = _BODY_INLINE.sub("", line).strip()
        if len(title.split()) < 3:
            continue  # truncation left nothing usable
        for nxt in lines[idx + 1: idx + 3]:
            if _BODY_START.match(nxt) or len(title) > 180:
                break
            cut = _BODY_INLINE.sub("", nxt).strip()
            title += " " + cut
            if cut != nxt.strip():
                break  # body phrase reached inside this line
        type_word = _TYPE_WORDS.get(getattr(act, "doc_type", ""), "")
        act.title = (f"{type_word} {title}".strip())[:250]
        if hasattr(act, "extraction_warnings"):
            act.extraction_warnings.append("title derived from act body heading")
        return True

    # Communiqués and similar untyped acts have no "privind…" line; use the
    # opening words of the body so the act is at least identifiable.
    if getattr(act, "doc_type", "") == "COMUNICAT" and lines:
        body_lines = [l for l in lines if not is_generic_title(l)]
        if body_lines:
            opening = re.sub(r"\s+", " ", body_lines[0])[:90].strip()
            if len(opening.split()) >= 3:
                act.title = f"Comunicat: {opening}"
                if hasattr(act, "extraction_warnings"):
                    act.extraction_warnings.append("title derived from communiqué opening")
                return True
    return False


def sanitize_title(act) -> bool:
    """Truncate body-text leakage from a title set by the rule extractor.

    Scanned-era titles often run on into the body ("DECRET privind numirea …
    Consiliul Frontului Salvării Naționale decretează: Articol unic …").
    Cuts at the first body phrase; returns True when the title changed.
    """
    title = (getattr(act, "title", "") or "").strip()
    if not title or is_generic_title(title):
        return False
    cut = _BODY_INLINE.sub("", title).strip()
    if cut == title and len(title) <= 220:
        return False
    cut = cut[:220].strip()
    if len(cut.split()) < 3:
        return False  # truncation would destroy the title; leave it
    act.title = cut
    if hasattr(act, "extraction_warnings"):
        act.extraction_warnings.append("title truncated at body-text leakage")
    return True


def repair_numbers_from_sumar(acts: list, sumar_entries: list, report: ReconcileReport) -> int:
    """Fix duplicate act numbers using the sumar sequence.

    Template-twin acts (two decrees identical except a name) sometimes both
    get the FIRST twin's number when the layout engine scrambles the closing
    blocks.  When a positionally matched act's number duplicates another
    act's, its doc_type equals the sumar entry's, and the sumar number is
    unique, the sumar number wins.  Returns the number of repairs.
    """
    from collections import Counter

    by_index = {getattr(a, "act_index", i): a for i, a in enumerate(acts)}
    act_counts = Counter(_norm_nr(getattr(a, "act_number", "")) for a in acts)
    sumar_counts = Counter(_norm_nr(getattr(e, "act_number", "")) for e in sumar_entries)
    fixed = 0
    for act_idx, sj in report.act_to_sumar.items():
        act = by_index.get(act_idx)
        if act is None:
            continue
        entry = sumar_entries[sj]
        a_nr = _norm_nr(getattr(act, "act_number", ""))
        s_raw = str(getattr(entry, "act_number", "") or "")
        s_nr = _norm_nr(s_raw)
        if not s_nr or s_nr == "0" or a_nr == s_nr:
            continue
        if act_counts[a_nr] < 2 or sumar_counts[s_nr] != 1:
            continue
        t_a = _fold(getattr(act, "doc_type", ""))
        t_s = _fold(getattr(entry, "doc_type", ""))
        t_a = "" if t_a in ("ACT", "UNKNOWN") else t_a
        t_s = "" if t_s in ("ACT", "UNKNOWN") else t_s
        if t_a and t_s and t_a != t_s:
            continue
        prev = act.act_number
        act.act_number = s_raw.split("/")[0]
        m_year = re.search(r"/(\d{4})", s_raw)
        if m_year and hasattr(act, "act_year"):
            act.act_year = int(m_year.group(1))
        if hasattr(act, "extraction_warnings"):
            act.extraction_warnings.append(
                f"act_number repaired from sumar (was duplicate {prev!r} → {act.act_number!r})"
            )
        act_counts[a_nr] -= 1
        fixed += 1
    return fixed


def dedup_repeated_acts(acts: list) -> tuple[list, int]:
    """Drop acts that duplicate an earlier act (OCR emission loops mint the
    same decree several times).

    Two acts are duplicates when they share a non-zero act number and doc_type
    AND one body prefix contains the other (whitespace/diacritic-folded).
    The act with the longer full_text wins.
    """
    kept: list = []
    dropped = 0
    for act in acts:
        nr = _norm_nr(getattr(act, "act_number", ""))
        dup_of = None
        if nr and nr != "0":
            for prev in kept:
                if _norm_nr(getattr(prev, "act_number", "")) != nr:
                    continue
                if _fold(getattr(prev, "doc_type", "")) != _fold(getattr(act, "doc_type", "")):
                    continue
                # FULL-body containment required: template-twin acts (two
                # decrees identical except a name mid-text) share their whole
                # preamble, so a prefix comparison would wrongly merge them.
                a = _fold(getattr(prev, "full_text", ""))
                b = _fold(getattr(act, "full_text", ""))
                if a and b and (a == b or a in b or b in a):
                    dup_of = prev
                    break
        if dup_of is None:
            kept.append(act)
            continue
        dropped += 1
        if len(getattr(act, "full_text", "")) > len(getattr(dup_of, "full_text", "")):
            kept[kept.index(dup_of)] = act  # keep the fuller copy
    return kept, dropped


def backfill_titles(acts: list, sumar_entries: list, report: ReconcileReport) -> None:
    """Replace generic/empty act titles with the matched sumar title."""
    by_index = {getattr(a, "act_index", i): a for i, a in enumerate(acts)}
    for act_idx, sj in report.act_to_sumar.items():
        act = by_index.get(act_idx)
        if act is None:
            continue
        sumar_title = str(getattr(sumar_entries[sj], "title", "") or "").strip()
        if not sumar_title or is_generic_title(sumar_title):
            continue
        if is_generic_title(getattr(act, "title", "")):
            act.title = sumar_title
            if hasattr(act, "extraction_warnings"):
                act.extraction_warnings.append(
                    f"title backfilled from sumar[{sj}] (was generic)"
                )
            report.titles_backfilled += 1


# ── Sumar number authority (era-gated) ───────────────────────────────────────
# Eras whose sumar is reliable enough to OVERRIDE closing-block numbers.
# BROKEN_2007: Docling displaces/drops closing blocks on two-column pages, so
# the closing-derived number can belong to the neighbouring act, while the
# sumar carries correct numbers, /YYYY years, and near-verbatim act titles.
SUMAR_AUTHORITY_ERAS = {"broken_2007"}

# Decisive title-containment score, and the margin under which two candidate
# entries are considered ambiguous (template twins).
_ANCHOR_SCORE_MIN = 0.65
_ANCHOR_MARGIN = 0.20


def _title_tokens(text: str) -> list[str]:
    """Diacritic-folded lowercase word tokens (len ≥ 3) for title matching."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3]


def _body_heading_tokens(act) -> list[str]:
    """Tokens of the act's own title line as printed in its body.

    Gazette bodies open with the type heading then a "privind…/pentru…" title
    line — the same line the sumar reproduces. Returns [] when no usable
    heading is found (e.g. table-only annex acts)."""
    lines = [l.strip() for l in (getattr(act, "full_text", "") or "").splitlines() if l.strip()]
    for idx, line in enumerate(lines[:8]):
        if not _TITLE_START.match(line):
            continue
        title = _BODY_INLINE.sub("", line).strip()
        for nxt in lines[idx + 1: idx + 3]:
            if _BODY_START.match(nxt) or len(title) > 200:
                break
            cut = _BODY_INLINE.sub("", nxt).strip()
            title += " " + cut
            if cut != nxt.strip():
                break
        toks = _title_tokens(title)
        if len(toks) >= 3:
            return toks[:25]
    return []


def _containment(body_toks: list[str], entry_toks: set[str]) -> float:
    """Fraction of the body-heading tokens present in the sumar title.

    The sumar title is a superset of the body heading in this era, so
    containment is the right direction (Jaccard would punish the sumar's
    extra type-word prefix)."""
    if not body_toks:
        return 0.0
    return sum(1 for t in body_toks if t in entry_toks) / len(body_toks)


def split_nr_year(raw: str | None) -> tuple[str, int | None]:
    """'1.441/2006' → ('1441', 2006); '226' → ('226', None)."""
    s = str(raw or "")
    m = re.search(r"/((?:1[89]|20)\d{2})\s*$", s)
    year = int(m.group(1)) if m else None
    return _norm_nr(s), year


def anchor_numbers_to_sumar(acts: list, sumar_entries: list, era, mode: str = "enforce") -> int:
    """Era-scoped: override closing-derived act numbers with the sumar number
    whose TITLE matches the act's own body heading.

    Safety properties:
      * era-gated to SUMAR_AUTHORITY_ERAS — structural no-op elsewhere;
      * absent/degenerate sumar (fewer than 2 usable entries) → no-op;
      * double-evidence rule: an override requires BOTH a decisive title match
        on the new entry AND a title MISmatch on the entry carrying the act's
        current number — a correct closing number can never be overridden;
      * template twins (titles within _ANCHOR_MARGIN of each other) fall back
        to monotone first-unclaimed assignment, the same positional semantics
        the pipeline already uses — no behaviour change for twins.

    mode: "off" → no-op; "warn" → log would-be overrides without mutating;
    "enforce" → apply. Returns the number of (would-be) overrides — callers
    can use a high count as a re-segmentation quality signal.
    """
    if mode == "off":
        return 0
    era_val = getattr(era, "value", era)
    if era_val not in SUMAR_AUTHORITY_ERAS:
        return 0

    usable: list[tuple[int, object, set[str], str, int | None]] = []
    for sj, e in enumerate(sumar_entries):
        title = str(getattr(e, "title", "") or "")
        s_nr, s_year = split_nr_year(getattr(e, "act_number", ""))
        if not s_nr or s_nr == "0" or _is_junk_entry(e) or is_generic_title(title):
            continue
        usable.append((sj, e, set(_title_tokens(title)), s_nr, s_year))
    if len(usable) < 2:
        return 0

    by_nr = {u[3]: u for u in usable}
    claimed: set[int] = set()
    overrides = 0
    for act in acts:
        body_toks = _body_heading_tokens(act)
        if not body_toks:
            continue
        scored = sorted(
            ((_containment(body_toks, toks), sj) for sj, _e, toks, _nr, _yr in usable
             if sj not in claimed),
            key=lambda x: (-x[0], x[1]),
        )
        if not scored or scored[0][0] < _ANCHOR_SCORE_MIN:
            continue
        top_score = scored[0][0]
        within = [sj for s, sj in scored if top_score - s < _ANCHOR_MARGIN]
        if len(within) > 1:
            # ambiguous (template twins) → monotone: first unclaimed in sumar order
            sj = min(within)
        else:
            sj = scored[0][1]
        entry = next(u for u in usable if u[0] == sj)
        _sj, _e, _toks, s_nr, s_year = entry
        a_nr = _norm_nr(getattr(act, "act_number", ""))

        if a_nr != s_nr:
            # double-evidence: keep the closing number when its OWN sumar
            # entry title-matches this body (i.e. the closing agrees)
            own = by_nr.get(a_nr)
            if own is not None and _containment(body_toks, own[2]) >= _ANCHOR_SCORE_MIN:
                continue  # entry NOT claimed — it belongs to another act
            claimed.add(sj)
            overrides += 1
            if mode == "warn":
                if hasattr(act, "extraction_warnings"):
                    act.extraction_warnings.append(
                        f"sumar title-anchor WOULD override act_number "
                        f"{act.act_number!r}→{s_nr!r} (warn mode)"
                    )
                continue
            prev = act.act_number
            act.act_number = s_nr
            if s_year is not None and hasattr(act, "act_year"):
                act.act_year = s_year
            if hasattr(act, "extraction_warnings"):
                act.extraction_warnings.append(
                    f"act_number overridden by sumar title-anchor "
                    f"({prev!r}→{s_nr!r}; closing block displaced by layout engine)"
                )
        else:
            claimed.add(sj)
            if mode == "enforce" and s_year is not None and hasattr(act, "act_year") \
                    and getattr(act, "act_year", None) != s_year:
                act.act_year = s_year
                if hasattr(act, "extraction_warnings"):
                    act.extraction_warnings.append(
                        f"act_year corrected from sumar title-anchor (→{s_year})"
                    )
    return overrides


def backfill_years_from_sumar(acts: list, sumar_entries: list, report: ReconcileReport) -> int:
    """Set act_year from the matched sumar entry's 'N/YYYY' form.

    Covers acts whose closing block was dropped entirely (no signing year to
    parse). Only fires when the numbers agree — a positional match with a
    different number is not year evidence. Scanned-era sumar entries carry no
    years → structural no-op there. Returns the number of corrections."""
    by_index = {getattr(a, "act_index", i): a for i, a in enumerate(acts)}
    fixed = 0
    for act_idx, sj in report.act_to_sumar.items():
        act = by_index.get(act_idx)
        if act is None or not hasattr(act, "act_year"):
            continue
        s_nr, s_year = split_nr_year(getattr(sumar_entries[sj], "act_number", ""))
        if s_year is None or s_nr != _norm_nr(getattr(act, "act_number", "")):
            continue
        if getattr(act, "act_year", None) != s_year:
            act.act_year = s_year
            if hasattr(act, "extraction_warnings"):
                act.extraction_warnings.append(
                    f"act_year corrected from sumar (→{s_year})"
                )
            fixed += 1
    return fixed


# Shared with md_act_recovery's act-unique-tail concept: the longest segment
# between boilerplate markers holds the act-unique content (appointee names…).
_BOILERPLATE_SPLIT = re.compile(
    r"PRE[ȘS]EDINTELE\s+ROM[ÂA]NIEI|Bucure[șs]ti,\s*\d{1,2}\s+\w+\s+\d{4}"
)


def act_unique_tail(full_text: str) -> str:
    """Folded ~120-char tail of the act's body (its act-unique content)."""
    parts = _BOILERPLATE_SPLIT.split(full_text or "")
    if not parts:
        return ""
    return _fold(max(parts, key=len))[-120:]


def drop_contained_unmatched(acts: list, report: ReconcileReport,
                             mode: str = "enforce") -> tuple[list, list[str]]:
    """Drop sumar-unmatched acts whose act-unique tail is contained in a
    sumar-MATCHED act's body — they are misnumbered shadows of that act
    (e.g. a recovery duplicate under a wrong number/doc_type, which
    dedup_repeated_acts cannot catch because it requires equal number+type).

    A sumar-matched act is never dropped. Returns (kept_acts, descriptions)."""
    matched_idx = set(report.act_to_sumar)
    matched_folds = [
        (i, _fold(getattr(a, "full_text", ""))) for i, a in enumerate(acts)
        if getattr(a, "act_index", i) in matched_idx
    ]
    kept, dropped = [], []
    for i, act in enumerate(acts):
        if getattr(act, "act_index", i) in matched_idx:
            kept.append(act)
            continue
        tail = act_unique_tail(getattr(act, "full_text", ""))
        if len(tail) >= 60 and any(tail in f for j, f in matched_folds if j != i):
            desc = (f"act[{getattr(act, 'act_index', i)}] "
                    f"{getattr(act, 'doc_type', '?')} nr={getattr(act, 'act_number', '?')!r} "
                    f"dropped: sumar-unmatched, body contained in a matched act "
                    f"(misnumbered shadow)")
            dropped.append(desc)
            if mode == "warn":
                kept.append(act)
            continue
        kept.append(act)
    return kept, dropped
