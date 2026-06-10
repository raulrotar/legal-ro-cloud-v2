"""Tests for sumar ↔ acts reconciliation and title backfill."""
from __future__ import annotations

from dataclasses import dataclass, field

from legalro_processing.extract.sumar_reconcile import (
    backfill_title_from_body,
    backfill_titles,
    dedup_repeated_acts,
    is_generic_title,
    reconcile,
)


@dataclass
class FakeAct:
    act_index: int
    doc_type: str
    act_number: str
    title: str = ""
    full_text: str = ""
    extraction_warnings: list = field(default_factory=list)


@dataclass
class FakeSumar:
    act_number: str
    doc_type: str
    title: str
    page_start: int = 1


def test_exact_match_and_title_backfill():
    acts = [
        FakeAct(0, "DECRET", "20", title="DECRET"),
        FakeAct(1, "DECRET", "21", title="DECRET"),
    ]
    sumar = [
        FakeSumar("20", "DECRET", "Decret pentru numirea unui judecător"),
        FakeSumar("21", "DECRET", "Decret pentru numirea unui procuror"),
    ]
    rep = reconcile(acts, sumar)
    assert rep.act_to_sumar == {0: 0, 1: 1}
    assert not rep.missing_sumar and not rep.unmatched_acts

    backfill_titles(acts, sumar, rep)
    assert acts[0].title == "Decret pentru numirea unui judecător"
    assert acts[1].title == "Decret pentru numirea unui procuror"
    assert rep.titles_backfilled == 2


def test_missing_act_detected():
    # MO 1/2007 case: 24 sumar entries, 22 acts — must be flagged
    acts = [FakeAct(0, "DECRET", "1415")]
    sumar = [
        FakeSumar("1.415/2006", "DECRET", "Decret privind înaintarea în grad"),
        FakeSumar("1.416/2006", "DECRET", "Decret privind înaintarea în grad"),
    ]
    rep = reconcile(acts, sumar)
    assert rep.act_to_sumar == {0: 0}  # 1.415 normalizes to 1415
    assert rep.missing_sumar == [1]
    assert any("MISSING act" in w for w in rep.warnings)


def test_phantom_act_detected():
    acts = [
        FakeAct(0, "DECRET", "20"),
        FakeAct(1, "UNKNOWN", "0"),  # segmentation fragment with no sumar entry
    ]
    sumar = [FakeSumar("20", "DECRET", "Decret pentru numirea unui judecător")]
    rep = reconcile(acts, sumar)
    assert rep.unmatched_acts == [1]
    assert any("phantom candidate" in w for w in rep.warnings)


def test_positional_fallback_for_unnumbered():
    # scanned-era COMUNICAT with no number matches positionally
    acts = [FakeAct(0, "COMUNICAT", "", title="")]
    sumar = [FakeSumar("", "COMUNICAT", "Comunicat din partea CFSN")]
    rep = reconcile(acts, sumar)
    assert rep.act_to_sumar == {0: 0}
    backfill_titles(acts, sumar, rep)
    assert acts[0].title == "Comunicat din partea CFSN"


def test_real_title_never_overwritten():
    acts = [FakeAct(0, "LEGE", "10", title="Lege privind aprobarea bugetului de stat")]
    sumar = [FakeSumar("10", "LEGE", "Alt titlu din sumar")]
    rep = reconcile(acts, sumar)
    backfill_titles(acts, sumar, rep)
    assert acts[0].title == "Lege privind aprobarea bugetului de stat"
    assert rep.titles_backfilled == 0


def test_duplicate_numbers_not_misassigned():
    # two acts share a number (annex split) — pass 1/2 must not guess
    acts = [
        FakeAct(0, "ORDIN", "55"),
        FakeAct(1, "ORDIN", "55"),
    ]
    sumar = [FakeSumar("55", "ORDIN", "Ordin privind aprobarea normelor")]
    rep = reconcile(acts, sumar)
    # positional pass pairs act 0 with the single entry; act 1 is unmatched
    assert 0 in rep.act_to_sumar
    assert rep.unmatched_acts == [1]


def test_generic_act_type_is_wildcard():
    # PI_821_2007: sumar parser emits doc_type='ACT' with empty numbers;
    # positional matching must still pair them with typed acts
    acts = [
        FakeAct(0, "DECIZIE", "920", title="DECIZIE"),
        FakeAct(1, "DECIZIE", "922", title="DECIZIE"),
    ]
    sumar = [
        FakeSumar("", "ACT", "Decizie privind numirea unui consilier"),
        FakeSumar("", "ACT", "Decizie privind eliberarea unui consilier"),
    ]
    rep = reconcile(acts, sumar)
    assert rep.act_to_sumar == {0: 0, 1: 1}
    backfill_titles(acts, sumar, rep)
    assert rep.titles_backfilled == 2


def test_generic_title_detection():
    assert is_generic_title("DECRET")
    assert is_generic_title("Hotărâre")
    assert is_generic_title("")
    assert not is_generic_title("Decret pentru numirea unui judecător")


def test_title_from_body():
    act = FakeAct(0, "DECRET_LEGE", "3", title="DECRET-LEGE", full_text=(
        "DECRET-LEGE\nprivind stabilirea tarifului la energia electrică "
        "livrată populației\nConsiliul Frontului Salvării Naționale decretează:"
    ))
    assert backfill_title_from_body(act)
    assert act.title.startswith("Decret-lege privind stabilirea tarifului")
    # body-start line must not leak into the title
    assert "Consiliul" not in act.title


def test_title_from_body_skips_real_title():
    act = FakeAct(0, "LEGE", "10", title="Lege privind bugetul",
                  full_text="LEGE\nprivind altceva\n")
    assert not backfill_title_from_body(act)
    assert act.title == "Lege privind bugetul"


def test_dedup_repeated_acts():
    body = "DECRET\nprivind numirea viceprim-miniștrilor guvernului\nArticol unic."
    acts = [
        FakeAct(0, "DECRET", "6", full_text=body),
        FakeAct(1, "DECRET", "6", full_text=body + " — Se numesc în funcție."),
        FakeAct(2, "DECRET", "6", full_text=body),
        FakeAct(3, "DECRET", "7", full_text="DECRET\nprivind eliberarea din funcție\n"),
    ]
    kept, dropped = dedup_repeated_acts(acts)
    assert dropped == 2
    assert [a.act_number for a in kept] == ["6", "7"]
    # the fuller copy of nr. 6 wins
    assert "Se numesc" in kept[0].full_text


def test_dedup_keeps_distinct_same_number():
    # same number, different body (e.g. act + its annex split) — keep both
    acts = [
        FakeAct(0, "ORDIN", "55", full_text="ORDIN privind aprobarea normelor metodologice"),
        FakeAct(1, "ORDIN", "55", full_text="ANEXĂ la ordinul nr. 55 — tabel cu valori"),
    ]
    kept, dropped = dedup_repeated_acts(acts)
    assert dropped == 0 and len(kept) == 2


def test_sanitize_runon_title():
    from legalro_processing.extract.sumar_reconcile import sanitize_title
    act = FakeAct(0, "DECRET", "5", title=(
        "DECRET privind numirea viceprim-miniștrilor guvernului "
        "Consiliul Frontului Salvării Naționale decretează: Articol unic"
    ))
    assert sanitize_title(act)
    assert act.title == "DECRET privind numirea viceprim-miniștrilor guvernului"


def test_sanitize_keeps_clean_title():
    from legalro_processing.extract.sumar_reconcile import sanitize_title
    act = FakeAct(0, "LEGE", "10", title="Lege privind aprobarea bugetului de stat")
    assert not sanitize_title(act)
    assert act.title == "Lege privind aprobarea bugetului de stat"


def test_empty_sumar_noop():
    acts = [FakeAct(0, "DECRET", "5", title="DECRET")]
    rep = reconcile(acts, [])
    assert rep.act_to_sumar == {}
    assert not rep.warnings
