#!/usr/bin/env python3
"""Table/Nomenclator retrieval-correctness questions for the HTML-table work.

Same tuple format and runner contract as tools/test_questions_cloud.py, so these
can be merged into the main suite once they pass. Expected answers are grounded
in docs/qa_html_tables_2026-06-15.md §1 (verified against
db/extracted/2026/04/14/MO_PI_294Bis_2026-04-14.json).

Run (needs the serving stack up):
    LEGALRO_API_URL=http://localhost:7861 uv run python tools/test_questions_tables.py

QT1-QT4 exercise the NEW merged-header capability (expected PARTIAL/WRONG on the
pre-change baseline). QT5-QT6 are POSITIVE regression guards on tables that are
already clean — they must keep passing after the change.
"""
from __future__ import annotations

# (Q-id, question, expected-answer-substring/oracle)
QUESTIONS = [
    ("QT1", "Câte credite ECTS are specializarea «Inteligență artificială» din "
            "Nomenclatorul aprobat prin HG nr. 191/2026, publicat în MO nr. 294 "
            "bis/14.IV.2026?",
            "180"),
    ("QT2", "Ce cod ISCED F-2013 corespunde domeniului «Matematică» în "
            "Nomenclatorul domeniilor și al specializărilor universitare "
            "(MO 294 bis/2026)?",
            "0541"),
    ("QT3", "În ce domeniu de studii se încadrează specializarea «Securitate "
            "informatică și știința datelor» și câte credite ECTS are, conform "
            "Nomenclatorului din MO 294 bis/2026?",
            "Informatică; 180"),
    ("QT4", "Câte credite ECTS sunt prevăzute pentru specializările din domeniul "
            "«Inginerie chimică și procese» în Nomenclatorul din MO 294 bis/2026?",
            "240"),
    ("QT5", "Care este cuantumul total al cotizațiilor primite în luna ianuarie, "
            "conform tabelului din MO nr. 311/20.IV.2026?",
            "conform tabelului din documentul publicat în MO nr. 311/2026"),
    ("QT6", "Lista beneficiarilor/finanțărilor din MO nr. 822/3.XII.2007 conține "
            "rânduri cu sume și beneficiari — confirmă prezența tabelului și a "
            "datelor pe rânduri.",
            "tabel cu rânduri de beneficiari/sume în MO nr. 822/2007"),
]

if __name__ == "__main__":
    # Reuse the cloud harness runner so reporting/scoring stays identical.
    import tools.test_questions_cloud as base  # type: ignore

    base.QUESTIONS = QUESTIONS
    if hasattr(base, "main"):
        base.main()
    else:
        raise SystemExit(
            "tools/test_questions_cloud.py exposes no main(); call its runner "
            "directly or merge QUESTIONS into it."
        )
