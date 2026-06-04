"""Regression tests for md_segmenter._is_artefact — specifically the publisher-footer rule."""
from legalro_processing.extract.md_segmenter import _is_artefact, _make_block


def _block(text: str):
    return _make_block(text)


# ── Publisher footer → must be artefact ──────────────────────────────────────

def test_editor_iban_footer_is_artefact():
    # Exact content from MO_PI_74_2017 line 594+
    text = (
        "EDITOR: GUVERNUL ROMÂNIEI\n\n"
        "'Monitorul Oficial' R.A., Str. Parcului nr. 65, sectorul 1, București; "
        "C.I.F. RO427282, IBAN: RO55RNCB0082006711100001 Banca Comercială Română - S.A.\n"
        "Tel. 021.318.51.29/150, fax 021.318.51.15, e-mail: marketing@ramo.ro, "
        "internet: www.monitoruloficial.ro\n"
        "Tiparul: 'Monitorul Oficial' R.A.\n"
    )
    assert _is_artefact(_block(text))


def test_footer_with_url_is_artefact():
    text = (
        "Banca Comercială C.I.F. RO12345 IBAN: RO99TEST0001 "
        "www.monitoruloficial.ro Tiparul: 'Monitorul Oficial' R.A. "
        "Adresa pentru publicitate sectorul 5 tel 021.401.00.70 "
        "fax 021.401.00.71 si 021.401.00.72\n"
        "EDITOR: GUVERNUL ROMÂNIEI\n"
    )
    assert _is_artefact(_block(text))


# ── Real act bodies with "Monitorul Oficial" → must NOT be artefact ──────────

def test_act_mentioning_mo_not_artefact():
    # Common phrasing in act bodies: "se publică în Monitorul Oficial"
    text = (
        "Art. 1. — Prezentul ordin se publică în Monitorul Oficial al României, "
        "Partea I.\n"
        "Ministrul finanțelor,\n"
        "Ion Popescu\n"
        "București, 15 mai 2007.\n"
        "Nr. 202.\n"
    )
    assert not _is_artefact(_block(text))


def test_act_with_act_body_signal_not_artefact():
    # Even if it contains IBAN (hypothetical edge case), act body signal gates it
    text = (
        "IBAN: RO55TEST01\n"
        "Art. 1. - Se aprobă normele metodologice.\n"
        "Art. 2. - Prezentul ordin se aplică.\n"
    )
    # Has _ACT_BODY_SIGNAL ("Art."), should NOT be artefact even with IBAN token
    assert not _is_artefact(_block(text))


# ── Footnote fragment → must be artefact ─────────────────────────────────────

def test_footnote_fragment_is_artefact():
    text = "*) A se vedea nota de subsol.\n"
    assert _is_artefact(_block(text))


# ── Short block without body signal → must be artefact ───────────────────────

def test_short_block_no_signal_is_artefact():
    text = "ACTE ALE PARTIDELOR POLITICE\n"
    assert _is_artefact(_block(text))


# ── Real act block with body → must NOT be artefact ─────────────────────────

def test_real_act_block_not_artefact():
    text = (
        "ORDIN\n\n"
        "privind aprobarea normelor de transport în regim de taxi\n\n"
        "Având în vedere prevederile art. 7 din OUG nr. 30/2007,\n"
        "în temeiul art. 10 alin. (3) din HG nr. 920/2005,\n\n"
        "ministrul internelor și reformei administrative emite următorul ordin:\n\n"
        "Art. 1. — Se aprobă normele metodologice.\n"
    )
    assert not _is_artefact(_block(text))
