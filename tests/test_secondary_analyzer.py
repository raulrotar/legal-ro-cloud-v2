"""Regression tests for secondary_analyzer.py — FitzAnalyzer + match_recovered_number."""
import pytest
from pathlib import Path

from legalro_processing.extract.secondary_analyzer import (
    FitzAnalyzer,
    ClosingSig,
    match_recovered_number,
    _extract_sigs_from_text,
)


# ── _extract_sigs_from_text unit tests (no fitz required) ────────────────────

def test_extract_full_closing_block():
    text = (
        "Ministrul internelor și reformei administrative,\n"
        "Cristian David\n"
        "București, 3 decembrie 2007.\n"
        "Nr. 356.\n"
    )
    sigs = _extract_sigs_from_text(text, page_no=6, masthead_nr="820")
    assert len(sigs) == 1
    assert sigs[0].number == "356"
    assert sigs[0].page_no == 6
    assert "Cristian David" in sigs[0].context


def test_extract_standalone_nr():
    text = "Cristian David\n\nNr. 356.\n"
    sigs = _extract_sigs_from_text(text, page_no=6, masthead_nr="820")
    assert any(s.number == "356" for s in sigs)


def test_masthead_excluded():
    # The gazette issue number (820) must not be returned as a closing sig
    text = "Monitorul Oficial Nr. 820\nNr. 820.\n"
    sigs = _extract_sigs_from_text(text, page_no=1, masthead_nr="820")
    assert all(s.number != "820" for s in sigs)


def test_no_nr_returns_empty():
    text = "Some act body text without any closing number.\n"
    sigs = _extract_sigs_from_text(text, page_no=3, masthead_nr=None)
    assert sigs == []


# ── FitzAnalyzer integration test (requires actual PDF) ──────────────────────

GAZETTE_820 = Path("laws/2007/12/03/MO_PI_820_2007-12-03.pdf")


@pytest.mark.skipif(
    not GAZETTE_820.exists(),
    reason="MO_PI_820 PDF not present in laws/ directory",
)
def test_fitz_analyzer_820():
    analyzer = FitzAnalyzer()
    sigs = analyzer.recover_closing_numbers(GAZETTE_820)

    numbers = [s.number for s in sigs]
    # Must recover 356 (taxi ordin) and 346 (ANCEX ordin)
    assert "356" in numbers, f"Expected '356' in {numbers}"
    assert "346" in numbers, f"Expected '346' in {numbers}"
    # Must NOT include the gazette masthead number
    assert "820" not in numbers, f"Masthead 820 leaked into results: {numbers}"


@pytest.mark.skipif(
    not GAZETTE_820.exists(),
    reason="MO_PI_820 PDF not present in laws/ directory",
)
def test_fitz_analyzer_page_numbers():
    sigs = FitzAnalyzer().recover_closing_numbers(GAZETTE_820)
    sig_356 = next((s for s in sigs if s.number == "356"), None)
    assert sig_356 is not None
    # Nr. 356 appears on page 6 of the 16-page gazette PDF
    assert sig_356.page_no == 6, f"Expected page 6, got {sig_356.page_no}"


# ── match_recovered_number unit tests ─────────────────────────────────────────

def _make_sigs(*entries):
    return [ClosingSig(page_no=pg, number=nr, context=ctx) for pg, nr, ctx in entries]


def test_match_single_hit():
    sigs = _make_sigs(
        (6, "356", "Ministrul internelor Cristian David București 2007 Nr. 356."),
    )
    result = match_recovered_number(
        sigs,
        signatory_hint="Cristian David",
        page_hints=[6],
        abrogation_numbers=["275/2003"],
    )
    assert result == "356"


def test_match_abrogation_excluded():
    sigs = _make_sigs(
        (6, "275", "se abrogă Ordinul nr. 275/2003 București 2007 Nr. 275."),
        (6, "356", "Cristian David București 2007 Nr. 356."),
    )
    result = match_recovered_number(
        sigs,
        signatory_hint="Cristian David",
        page_hints=[6],
        abrogation_numbers=["275/2003"],
    )
    # 275 is an abrogation ref — should be excluded; 356 matches
    assert result == "356"


def test_match_ambiguous_returns_none():
    # Two candidates match — must bail on ambiguity
    sigs = _make_sigs(
        (5, "344", "Popescu București 2007 Nr. 344."),
        (6, "356", "Popescu București 2007 Nr. 356."),
    )
    result = match_recovered_number(
        sigs,
        signatory_hint="Popescu",
        page_hints=[5, 6],
        abrogation_numbers=[],
    )
    assert result is None


def test_match_empty_recovered_returns_none():
    result = match_recovered_number(
        [],
        signatory_hint="Cristian David",
        page_hints=[6],
        abrogation_numbers=[],
    )
    assert result is None


def test_match_no_page_overlap_returns_none():
    sigs = _make_sigs((6, "356", "Cristian David Nr. 356."))
    # Act is on pages [1, 2] — page 6 is far out of range (even with ±1 tolerance)
    result = match_recovered_number(
        sigs,
        signatory_hint="Cristian David",
        page_hints=[1, 2],
        abrogation_numbers=[],
    )
    assert result is None


def test_match_without_page_hints_uses_signatory():
    # No page_hints → signatory is the only anchor
    sigs = _make_sigs(
        (6, "356", "Cristian David București 2007 Nr. 356."),
        (10, "500", "Ionescu București 2007 Nr. 500."),
    )
    result = match_recovered_number(
        sigs,
        signatory_hint="Cristian David",
        page_hints=[],  # empty → no page gate
        abrogation_numbers=[],
    )
    assert result == "356"


# ── find_candidates + positional tiebreaker ───────────────────────────────────

from legalro_processing.extract.secondary_analyzer import find_candidates


def test_find_candidates_returns_sorted_by_page():
    sigs = _make_sigs(
        (6, "346", "Cristian Munteanu București 2007 Nr. 346."),
        (4, "345", "Cristian Munteanu București 2007 Nr. 345."),
    )
    candidates = find_candidates(sigs, "Cristian Munteanu", [], [])
    assert [c.number for c in candidates] == ["345", "346"]  # sorted by page


def test_positional_tiebreaker_first_act_gets_first_sig():
    sigs = _make_sigs(
        (4, "345", "Cristian Irinel Munteanu București Nr. 345."),
        (6, "346", "Cristian Irinel Munteanu București Nr. 346."),
    )
    assign_count: dict[str, int] = {}
    hint = "Cristian Irinel Munteanu"

    # Simulate first act needing recovery (same signatory)
    result1 = match_recovered_number(sigs, hint, [], [])
    assert result1 is None  # ambiguous → must bail without tiebreaker

    candidates = find_candidates(sigs, hint, [], [])
    assert len(candidates) == 2
    nth = assign_count.get(hint, 0)
    result1 = candidates[nth].number
    assign_count[hint] = nth + 1
    assert result1 == "345"  # first act → first sig (page 4)


def test_positional_tiebreaker_second_act_gets_second_sig():
    sigs = _make_sigs(
        (4, "345", "Cristian Irinel Munteanu București Nr. 345."),
        (6, "346", "Cristian Irinel Munteanu București Nr. 346."),
    )
    assign_count = {"Cristian Irinel Munteanu": 1}  # already assigned once
    hint = "Cristian Irinel Munteanu"

    candidates = find_candidates(sigs, hint, [], [])
    nth = assign_count.get(hint, 0)
    result = candidates[nth].number if nth < len(candidates) else None
    assert result == "346"  # second act → second sig (page 6)


def test_positional_tiebreaker_counter_per_gazette():
    # Counter is local (dict) — two independent "gazette runs" don't share state
    sigs = _make_sigs(
        (4, "345", "Cristian Munteanu București Nr. 345."),
        (6, "346", "Cristian Munteanu București Nr. 346."),
    )
    hint = "Cristian Munteanu"

    # Gazette 1: fresh counter
    count_g1: dict[str, int] = {}
    c = find_candidates(sigs, hint, [], [])
    n = count_g1.get(hint, 0); count_g1[hint] = n + 1
    assert c[n].number == "345"

    # Gazette 2: fresh counter — same result (not contaminated by gazette 1)
    count_g2: dict[str, int] = {}
    n2 = count_g2.get(hint, 0)
    assert c[n2].number == "345"
