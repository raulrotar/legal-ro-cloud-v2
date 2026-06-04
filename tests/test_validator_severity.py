"""Regression tests for extraction_validator severity rules.

Specifically guards: doc_type-gated downgrade of ACT_NUMBER_ZERO for COMUNICAT,
RECTIFICARE, ANUNT (legitimately have no own number) while keeping ERROR for
numbered act types (ORDIN, DECRET, HG, etc.).
"""
import json
import tempfile
from pathlib import Path

import pytest

from legalro_processing.extraction_validator import validate_file, Severity


def _write_gazette(acts: list[dict], tmp_path: Path) -> Path:
    gazette = {
        "gazette_id": "TEST_GAZETTE",
        "sumar": [],
        "sumar_raw": "",
        "acts": acts,
        "extraction_warnings": [],
    }
    p = tmp_path / "test.json"
    p.write_text(json.dumps(gazette, ensure_ascii=False), encoding="utf-8")
    return p


def _act(doc_type: str, act_number: str, full_text: str = "Body text here.") -> dict:
    return {
        "act_index": 0,
        "doc_type": doc_type,
        "act_number": act_number,
        "full_text": full_text,
        "issuing_authority": "Autoritatea X",
        "title": "Titlu act",
        "extraction_warnings": [],
    }


# ── COMUNICAT / RECTIFICARE / ANUNT → INFO, not ERROR ────────────────────────

@pytest.mark.parametrize("doc_type", ["COMUNICAT", "RECTIFICARE", "ANUNT", "ANUNȚ"])
def test_zero_number_comunicat_types_is_info(doc_type, tmp_path):
    p = _write_gazette([_act(doc_type, "0", "Scurt comunicat.")], tmp_path)
    issues = validate_file(p)
    zero_issues = [i for i in issues if i.check == "ACT_NUMBER_ZERO"]
    assert zero_issues, "Expected ACT_NUMBER_ZERO issue"
    for issue in zero_issues:
        assert issue.severity == Severity.INFO, (
            f"Expected INFO for {doc_type} with act_number=0, got {issue.severity}"
        )


# ── Numbered act types → ERROR ────────────────────────────────────────────────

@pytest.mark.parametrize("doc_type", ["ORDIN", "DECRET", "HG", "LEGE", "DCC", "DECIZIE", "OUG"])
def test_zero_number_numbered_types_is_error(doc_type, tmp_path):
    p = _write_gazette(
        [_act(doc_type, "0", "Art. 1. Prezentul act stabilește ceva important.")],
        tmp_path,
    )
    issues = validate_file(p)
    zero_issues = [i for i in issues if i.check == "ACT_NUMBER_ZERO"]
    assert zero_issues, f"Expected ACT_NUMBER_ZERO issue for {doc_type}"
    for issue in zero_issues:
        assert issue.severity == Severity.ERROR, (
            f"Expected ERROR for {doc_type} with act_number=0, got {issue.severity}"
        )


# ── ACT_NUMBER_ABROGATION stays ERROR regardless of doc_type ─────────────────

def test_abrogation_number_stays_error(tmp_path):
    act = _act("ORDIN", "275")
    # Inject full_text with an abrogation clause referencing nr. 275/2003
    act["full_text"] = (
        "Art. 2. — La data intrării în vigoare a prezentului ordin, "
        "Ordinul nr. 275/2003 privind ceva se abrogă."
    )
    p = _write_gazette([act], tmp_path)
    issues = validate_file(p)
    abr_issues = [i for i in issues if i.check == "ACT_NUMBER_ABROGATION"]
    assert abr_issues, "Expected ACT_NUMBER_ABROGATION"
    assert all(i.severity == Severity.ERROR for i in abr_issues)


# ── DOC_TYPE_UNKNOWN stays ERROR ──────────────────────────────────────────────

def test_doc_type_unknown_is_error(tmp_path):
    p = _write_gazette([_act("UNKNOWN", "0", "Vague content without type.")], tmp_path)
    issues = validate_file(p)
    unk_issues = [i for i in issues if i.check == "DOC_TYPE_UNKNOWN"]
    assert unk_issues
    assert all(i.severity == Severity.ERROR for i in unk_issues)
