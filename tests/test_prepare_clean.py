"""Tests for the Phase-0 pre-embedding cleanup."""
from dataclasses import dataclass

from legalro_processing.prepare.clean import (
    clean_act_text,
    is_embedding_poison,
    near_duplicate_act_indices,
)


def test_strips_running_headers_and_pagebreaks():
    text = (
        "Art. 1. — Se aprobă bugetul.\n"
        "MONITORUL OFICIAL AL ROMÂNIEI, PARTEA I, Nr. 311/20.IV.2026\n"
        "Monitorul Oficial al României, Partea I, Nr. 74 din 30 ianuarie 2017\n"
        "page-break\n"
        "<!-- legalro:page-break -->\n"
        "12\n"
        "Art. 2. — Prezenta lege intră în vigoare.\n"
    )
    out = clean_act_text(text)
    assert "MONITORUL" not in out and "Monitorul" not in out
    assert "page-break" not in out
    assert "Art. 1" in out and "Art. 2" in out


def test_strips_colophon_tail():
    text = (
        "Art. 1. — Conținut util al actului normativ aprobat.\n\n"
        "EDITOR: PARLAMENTUL ROMÂNIEI — CAMERA DEPUTAȚILOR\n"
        "Monitorul Oficial al României... abonamente... Prețul 5 lei ISSN 1453-4495\n"
    )
    out = clean_act_text(text)
    assert "EDITOR" not in out and "abonamente" not in out
    assert "Conținut util" in out


def test_poison_detection():
    # tiny-period OCR garbage → degenerate vocabulary window
    assert is_embedding_poison("nu. nu. nu. nc. nc. vol vol vol " * 80)
    varied = " ".join(
        f"Art. {i}. — Se numește în funcția de consilier persoana desemnată "
        f"prin ordinul numărul {i*3} din anul două mii șaptesprezece." for i in range(1, 7)
    )
    assert not is_embedding_poison(varied)


def test_adjacent_duplicate_blocks_collapsed_not_quarantined():
    # colophon ×3 is cleaned by collapse, and the act survives
    body = "Art. 1. — Conținut normativ valid al actului juridic.\n"
    colophon = "EDITOR: CONSILIUL FRONTULUI SALVĂRII NATIONALE\nAdresa pentru publicitate\n"
    out = clean_act_text(body + colophon * 3)
    assert out.count("Adresa pentru publicitate") <= 1
    assert not is_embedding_poison(out)


@dataclass
class FakeAct:
    full_text: str


def test_near_duplicate_detection():
    annex = " ".join(
        f"Articolul {i} stabilește condițiile de participare la concursul "
        f"de admitere pentru specializarea numărul {i} conform metodologiei."
        for i in range(1, 60)
    )
    acts = [
        FakeAct("ORDIN nr. 115 pentru aprobarea metodologiei.\n" + annex),
        FakeAct(annex),  # duplicated annex under a different number
        FakeAct("DECRET privind numirea unui judecător la Judecătoria Iași." * 10),
    ]
    drop = near_duplicate_act_indices(acts)
    assert drop == {1}
