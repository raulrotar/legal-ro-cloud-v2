"""
Validate an extracted GazetteDocument against a set of structural rules.

Returns a list of ValidationIssue objects.  Each issue has a severity
(ERROR / WARNING / INFO) and a path indicating where in the document the
problem was found.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from legalro_processing.extract.gazette_schema import GazetteDocument, LegalAct

Severity = Literal["ERROR", "WARNING", "INFO"]


@dataclass
class ValidationIssue:
    severity: Severity
    path: str        # e.g. "acts[2].articles[0]"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.path}: {self.message}"


def validate_gazette(g: GazetteDocument) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    # ── Gazette-level ─────────────────────────────────────────────────
    if not g.filename:
        issues.append(ValidationIssue("ERROR", "gazette", "filename is empty"))
    if g.issue_number == 0:
        issues.append(ValidationIssue("WARNING", "gazette", "issue_number is 0 — filename may not match pattern"))
    if g.pdf_page_count == 0:
        issues.append(ValidationIssue("ERROR", "gazette", "pdf_page_count is 0"))
    if not g.sha256:
        issues.append(ValidationIssue("ERROR", "gazette", "sha256 is missing"))
    if not g.acts:
        issues.append(ValidationIssue("ERROR", "gazette", "no acts extracted"))
    if not g.sumar:
        issues.append(ValidationIssue("WARNING", "gazette.sumar", "sumar is empty — may be SCANNED era"))

    # Check sumar ↔ act count alignment
    if g.sumar and g.acts:
        if abs(len(g.sumar) - len(g.acts)) > 2:
            issues.append(ValidationIssue(
                "WARNING", "gazette",
                f"sumar has {len(g.sumar)} entries but {len(g.acts)} acts were segmented — segmentation may be off"
            ))

    # ── Act-level ─────────────────────────────────────────────────────
    for i, act in enumerate(g.acts):
        path = f"acts[{i}]"
        issues.extend(_validate_act(act, path))

    return issues


def _validate_act(act: LegalAct, path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not act.doc_type:
        issues.append(ValidationIssue("ERROR", path, "doc_type is empty"))
    if not act.title:
        issues.append(ValidationIssue("WARNING", path, "title is empty"))
    if not act.full_text or len(act.full_text) < 20:
        issues.append(ValidationIssue("ERROR", path, "full_text is empty or too short"))
    if not act.issuing_authority:
        issues.append(ValidationIssue("WARNING", path, "issuing_authority is empty"))

    # Acts that normally have articles but none were parsed
    structural_types = {"ORDIN", "HOTARARE", "LEGE", "ORDONANTA"}
    if act.doc_type in structural_types and not act.articles and len(act.full_text) > 200:
        issues.append(ValidationIssue(
            "WARNING", path,
            f"{act.doc_type} has no parsed articles (full_text {len(act.full_text)} chars) — may be annex-only or parsing failed"
        ))

    # Article-level
    for j, article in enumerate(act.articles):
        apath = f"{path}.articles[{j}]"
        if article.article_number == "?":
            issues.append(ValidationIssue("WARNING", apath, "article number could not be parsed"))
        if not article.raw_text:
            issues.append(ValidationIssue("ERROR", apath, "raw_text is empty"))

    # Propagate per-act extraction warnings as INFO
    for w in act.extraction_warnings:
        issues.append(ValidationIssue("INFO", path, w))

    return issues


def format_report(issues: list[ValidationIssue], gazette_id: str) -> str:
    if not issues:
        return f"✓ {gazette_id}: no issues found"

    lines = [f"Validation report — {gazette_id}", "=" * 60]
    counts: dict[str, int] = {"ERROR": 0, "WARNING": 0, "INFO": 0}
    for issue in issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
        lines.append(str(issue))

    lines.append("-" * 60)
    lines.append(
        f"ERRORS: {counts['ERROR']}  WARNINGS: {counts['WARNING']}  INFO: {counts['INFO']}"
    )
    return "\n".join(lines)
