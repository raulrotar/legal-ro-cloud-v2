"""Tests for the metadata-boost layer in retrieval/search.py.

Pure-function tests — no DB, no Atlas. They pin:
  * fold_act_number canonicalization (dotted / compound / sentinel forms),
  * _parse_query_metadata on real QA-question phrasings,
  * _compute_boost year-gate semantics (±1 tolerance, exact-year bonus,
    distant-year gating) including a replay of the Q43 inversion.
"""
from legalro_core.act_number import fold_act_number
from legalro_core.retrieval.search import (
    _ACT_YEAR_TOLERANCE,
    _METADATA_BOOST,
    _MO_EXACT_MULTIPLIER,
    _TITLE_KEYWORD_BOOST,
    _compute_boost,
    _parse_query_metadata,
)


# ── fold_act_number ───────────────────────────────────────────────────────────

def test_fold_dotted():
    assert fold_act_number("1.439") == "1439"
    assert fold_act_number("20.022") == "20022"


def test_fold_year_suffix_stripped():
    assert fold_act_number("1.642/2016") == "1642"
    assert fold_act_number("699/2024") == "699"


def test_fold_compound_number_kept():
    # "999/726" is a joint-order compound number, not a year suffix
    assert fold_act_number("999/726") == "999/726"


def test_fold_sentinels():
    assert fold_act_number("0") == ""
    assert fold_act_number("") == ""
    assert fold_act_number(None) == ""


# ── _parse_query_metadata (real QA phrasings) ────────────────────────────────

def test_parse_q43_decret_slash():
    meta = _parse_query_metadata(
        "Ce tratat internațional a fost supus aprobării Parlamentului prin "
        "Decretul nr. 1440/2006, publicat în MO nr. 2/3.I.2007?")
    assert meta["act_number"] == "1440"
    assert meta["act_year"] == 2006
    assert meta["mo_number"] == "2"
    assert meta["mo_year"] == 2007


def test_parse_q31_dotted_number():
    meta = _parse_query_metadata(
        "Ce județ vizează Ordinul directorului general al ANCPI nr. 1.642/2016 "
        "privind închiderea evidențelor de cadastru publicat în MO nr. 76/30.I.2017?")
    assert meta["act_number"] == "1642"
    assert meta["act_year"] == 2016


def test_parse_q11_mo_exact():
    meta = _parse_query_metadata(
        "Ce reglementează Ordinul ministrului internelor și reformei administrative "
        "nr. 356/2007 privind transportul în regim de taxi, publicat în "
        "Monitorul Oficial nr. 820/3.XII.2007?")
    assert meta["act_number"] == "356"
    assert meta["act_year"] == 2007
    assert meta["mo_number"] == "820"
    assert meta["mo_year"] == 2007


# ── _compute_boost ───────────────────────────────────────────────────────────

def _doc(num, year, title="", issue=""):
    return {"act_number": num, "act_year": year, "title": title,
            "source_issue_id": issue}


META_1440 = {"act_number": "1440", "act_year": 2006,
             "mo_number": "2", "mo_year": 2007}


def test_exact_match_score_unchanged():
    # number + exact year: 0.01 + 0.005 — identical to the pre-fix scoring
    meta = {"act_number": "922", "act_year": 2007}
    b = _compute_boost(_doc("922", 2007), meta)
    assert b == _METADATA_BOOST * 1.5


def test_skewed_year_now_boosted():
    # signing-vs-publication skew: stored 2007, cited 2006 → base boost only
    meta = {"act_number": "1440", "act_year": 2006}
    b = _compute_boost(_doc("1440", 2007), meta)
    assert b == _METADATA_BOOST


def test_distant_year_still_gated():
    meta = {"act_number": "2", "act_year": 2007}
    assert _compute_boost(_doc("2", 1989), meta) == 0.0


def test_dotted_stored_number_matches():
    meta = {"act_number": "1908", "act_year": 2006}
    assert _compute_boost(_doc("1.908", 2007), meta) == _METADATA_BOOST


def test_compound_stored_number_matches_prefix():
    meta = {"act_number": "999", "act_year": 2026}
    assert _compute_boost(_doc("999/726", 2026), meta) == _METADATA_BOOST * 1.5


def test_standalone_year_half_boost_removed():
    # year matches but number doesn't → no boost at all (was +0.005 noise)
    meta = {"act_number": "1440", "act_year": 2006}
    assert _compute_boost(_doc("1.439", 2006), meta) == 0.0


def test_exact_year_outranks_skew_for_dual_stored_act():
    # the same act stored under both years: exact-cited copy must win
    meta = {"act_number": "233", "act_year": 2006}
    exact = _compute_boost(_doc("233", 2006), meta)
    skew = _compute_boost(_doc("233", 2007), meta)
    assert exact > skew > 0


def test_title_keyword_requires_number_match():
    meta = {"act_number": "346", "act_year": 2007}
    q = "normele metodologice pentru transportul în regim de taxi"
    boosted = _compute_boost(_doc("346", 2007, title="norme taxi"), meta, q)
    other = _compute_boost(_doc("345", 2007, title="norme taxi"), meta, q)
    assert boosted >= _METADATA_BOOST * 1.5 + _TITLE_KEYWORD_BOOST
    assert other == 0.0


def test_q43_replay_ranking_fixed():
    """The audited Q43 inversion: base RRF gap 0.0004 in the wrong chunk's
    favour; post-boost the correct (CITES, stored '1440'/2007) chunk must win."""
    query = ("Ce tratat internațional a fost supus aprobării Parlamentului prin "
             "Decretul nr. 1440/2006, publicat în MO nr. 2/3.I.2007?")
    right = _doc("1440", 2007, issue="MO_PI_2_2007",
                 title="Decret pentru supunerea spre aprobare Parlamentului a "
                       "acceptării Amendamentului la Convenția CITES")
    wrong = _doc("1.439", 2006, issue="MO_PI_2_2007",
                 title="Decret privind supunerea spre ratificare Parlamentului "
                       "a Convenției de cooperare polițienească")
    meta = _parse_query_metadata(query)
    # measured base scores from the audit (wrong chunk slightly ahead)
    right_score = 0.16639 - _METADATA_BOOST * _MO_EXACT_MULTIPLIER
    wrong_score = 0.17070 - _METADATA_BOOST * _MO_EXACT_MULTIPLIER
    right_total = right_score + _compute_boost(right, meta, query)
    wrong_total = wrong_score + _compute_boost(wrong, meta, query)
    assert right_total > wrong_total


def test_mo_exact_issue_boost_unchanged():
    meta = {"mo_number": "2", "mo_year": 2007}
    b = _compute_boost(_doc("x", None, issue="MO_PI_2_2007"), meta)
    assert b == _METADATA_BOOST * _MO_EXACT_MULTIPLIER


def test_year_tolerance_is_one():
    assert _ACT_YEAR_TOLERANCE == 1
