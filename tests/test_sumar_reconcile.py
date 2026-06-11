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
    act_year: int = 0
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


# ── Sumar number authority (title anchor) ────────────────────────────────────

from legalro_processing.extract.sumar_reconcile import (
    anchor_numbers_to_sumar,
    backfill_years_from_sumar,
    drop_contained_unmatched,
)


def _chimera_fixture():
    """The MO_PI_2_2007 case: closing blocks displaced one act down."""
    acts = [
        # body = police convention (1439), closing-derived number = 1440
        FakeAct(0, "DECRET", "1440", title="DECRET", full_text=(
            "DECRET\nprivind supunerea spre ratificare Parlamentului a "
            "Convenției de cooperare polițienească pentru Europa de Sud-Est, "
            "adoptată la Viena la 5 mai 2006\nÎn temeiul prevederilor art. 91\n"
            "PREȘEDINTELE ROMÂNIEI\nBucurești, 27 decembrie 2006. Nr. 1.440.")),
        # body = CITES (1440), positional fallback gave it 1.439
        FakeAct(1, "DECRET", "1.439", title="DECRET", full_text=(
            "DECRET\npentru supunerea spre aprobare Parlamentului a acceptării "
            "Amendamentului adus la Convenția privind comerțul internațional cu "
            "specii sălbatice de faună și floră, adoptat la Gaborone la 30 aprilie 1983\n"
            "În temeiul prevederilor art. 91\nPREȘEDINTELE ROMÂNIEI")),
    ]
    sumar = [
        FakeSumar("1.439/2006", "DECRET",
                  "Decret privind supunerea spre ratificare Parlamentului a "
                  "Convenției de cooperare polițienească pentru Europa de Sud-Est, "
                  "adoptată la Viena la 5 mai 2006"),
        FakeSumar("1.440/2006", "DECRET",
                  "Decret pentru supunerea spre aprobare Parlamentului a acceptării "
                  "Amendamentului adus la Convenția privind comerțul internațional cu "
                  "specii sălbatice de faună și floră, adoptat la Gaborone la 30 aprilie 1983"),
    ]
    return acts, sumar


def test_anchor_fixes_chimera_pair():
    acts, sumar = _chimera_fixture()
    n = anchor_numbers_to_sumar(acts, sumar, "broken_2007")
    assert n == 2
    assert acts[0].act_number == "1439"
    assert acts[1].act_number == "1440"
    assert acts[0].act_year == 2006
    assert acts[1].act_year == 2006


def test_anchor_era_gated():
    acts, sumar = _chimera_fixture()
    assert anchor_numbers_to_sumar(acts, sumar, "modern") == 0
    assert acts[0].act_number == "1440"  # untouched


def test_anchor_warn_mode_does_not_mutate():
    acts, sumar = _chimera_fixture()
    n = anchor_numbers_to_sumar(acts, sumar, "broken_2007", mode="warn")
    assert n == 2
    assert acts[0].act_number == "1440"  # unchanged
    assert any("WOULD override" in w for w in acts[0].extraction_warnings)


def test_anchor_double_evidence_keeps_correct_closing():
    # a correct act: closing number's own sumar entry title-matches the body
    acts = [FakeAct(0, "DECIZIE", "226", title="DECIZIE", full_text=(
        "DECIZIE\npentru numirea domnului Armean Petru în funcția de secretar "
        "de stat la Ministerul Sănătății Publice\nÎn temeiul art. 15"))]
    sumar = [
        FakeSumar("226/2006", "DECIZIE",
                  "Decizie pentru numirea domnului Armean Petru în funcția de "
                  "secretar de stat la Ministerul Sănătății Publice"),
        FakeSumar("227/2006", "DECIZIE",
                  "Decizie pentru eliberarea domnului Cătălin Florin Teodorescu "
                  "din funcția de secretar de stat"),
    ]
    n = anchor_numbers_to_sumar(acts, sumar, "broken_2007")
    assert acts[0].act_number == "226"
    assert acts[0].act_year == 2006  # year still corrected on confirm


def test_anchor_empty_or_generic_sumar_noop():
    acts, _ = _chimera_fixture()
    assert anchor_numbers_to_sumar(acts, [], "broken_2007") == 0
    generic = [FakeSumar("1", "DECRET", "DECRET"), FakeSumar("2", "DECRET", "")]
    assert anchor_numbers_to_sumar(acts, generic, "broken_2007") == 0


def test_anchor_template_twins_monotone():
    # identical titles except a name → ambiguous → first-unclaimed in order
    body = ("DECRET\nprivind înaintarea în gradul următor a unui general "
            "din Ministerul Apărării\nÎn temeiul prevederilor art. 94")
    acts = [
        FakeAct(0, "DECRET", "1420", title="DECRET", full_text=body),
        FakeAct(1, "DECRET", "1421", title="DECRET", full_text=body),
    ]
    sumar = [
        FakeSumar("1.420/2006", "DECRET",
                  "Decret privind înaintarea în gradul următor a unui general din Ministerul Apărării"),
        FakeSumar("1.421/2006", "DECRET",
                  "Decret privind înaintarea în gradul următor a unui general din Ministerul Apărării"),
    ]
    anchor_numbers_to_sumar(acts, sumar, "broken_2007")
    assert acts[0].act_number == "1420"
    assert acts[1].act_number == "1421"  # no swap


def test_masthead_title_is_generic():
    assert is_generic_title("Anul 175 (XIX) — Nr. 2")
    assert not is_generic_title("Decret privind numirea unui judecător")


def test_masthead_sumar_entry_never_matched():
    acts = [FakeAct(0, "DECRET", "1438", title="DECRET")]
    sumar = [
        FakeSumar("1438", "ACT", "Anul 175 (XIX) — Nr. 2"),
        FakeSumar("1.438/2006", "DECRET", "Decret privind aderarea la acord"),
    ]
    rep = reconcile(acts, sumar)
    # must match the REAL entry (index 1, original list), not the masthead
    assert rep.act_to_sumar == {0: 1}
    backfill_titles(acts, sumar, rep)
    assert acts[0].title == "Decret privind aderarea la acord"


# ── act_year backfill ────────────────────────────────────────────────────────

def test_backfill_years_from_sumar():
    acts = [FakeAct(0, "DECIZIE", "226", title="Decizie pentru numire")]
    acts[0].act_year = 2007  # wrong (issue year)
    sumar = [FakeSumar("226/2006", "DECIZIE", "Decizie pentru numirea domnului X")]
    rep = reconcile(acts, sumar)
    n = backfill_years_from_sumar(acts, sumar, rep)
    assert n == 1
    assert acts[0].act_year == 2006


def test_backfill_years_requires_number_agreement():
    # positional match with a different number must NOT transfer the year
    acts = [FakeAct(0, "DECIZIE", "999", title="Decizie pentru numire")]
    acts[0].act_year = 2007
    sumar = [FakeSumar("226/2006", "DECIZIE", "Decizie pentru numirea domnului X")]
    rep = reconcile(acts, sumar)
    backfill_years_from_sumar(acts, sumar, rep)
    assert acts[0].act_year == 2007  # unchanged


# ── contained-unmatched shadow drop ──────────────────────────────────────────

def test_shadow_drop_removes_misnumbered_duplicate():
    body = ("ORDIN\nprivind modificarea Ordinului MFP nr. 160/2004 pentru "
            "proceduri vamale speciale aplicabile importurilor\n"
            "Ministrul finanțelor publice emite următorul ordin\n"
            "Art. 1 Se modifică procedura de înregistrare a operatorilor "
            "economici autorizați pentru regimuri vamale suspensive XYZQW")
    acts = [
        FakeAct(0, "HG", "1908", title="ORDIN greșit", full_text=body),     # shadow
        FakeAct(1, "ORDIN", "2199", title="Ordin corect", full_text=body),  # real
    ]
    sumar = [FakeSumar("2199/2006", "ORDIN", "Ordin privind modificarea Ordinului MFP")]
    rep = reconcile(acts, sumar)
    assert 1 in rep.act_to_sumar and 0 not in rep.act_to_sumar
    kept, msgs = drop_contained_unmatched(acts, rep)
    assert len(kept) == 1
    assert kept[0].act_number == "2199"
    assert msgs


def test_shadow_drop_never_drops_matched_or_distinct():
    acts = [
        FakeAct(0, "DECRET", "10", title="t", full_text="A" * 500),
        FakeAct(1, "DECRET", "999", title="t",
                full_text="Conținut complet diferit despre alt subiect " * 20),
    ]
    sumar = [FakeSumar("10", "DECRET", "Decret unu")]
    rep = reconcile(acts, sumar)
    kept, msgs = drop_contained_unmatched(acts, rep)
    assert len(kept) == 2  # distinct unmatched act survives
    assert not msgs
