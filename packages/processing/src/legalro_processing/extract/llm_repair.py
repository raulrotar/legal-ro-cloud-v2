"""Vision-LLM repair pass for acts that fail validation.

Called ONLY when extraction_validator flags an act with a fixable ERROR
(ACT_NUMBER_ZERO, DOC_TYPE_UNKNOWN, ACT_NUMBER_ABROGATION, ACT_NUMBER_MALFORMED,
LLM_FAILED).  Sends the broken fields, specific error codes, the act's markdown
excerpt, AND the source PDF page images to llama3.2-vision:11b so it can recover
the correct values from the original document — not from the possibly-broken
markdown alone.

Key constraints:
- Only patches the FLAGGED fields; never touches a field that validated clean.
- act_number / act_year are patched only when the validation error specifically
  targets them (ACT_NUMBER_*), preserving the law_id regression constraint.
- If the repair LLM also fails or returns garbage, the original regex result is
  kept and an extraction_warning is appended.
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legalro_core.config import Settings
    from legalro_processing.extract.gazette_schema import LegalAct
    from legalro_processing.extract.md_segmenter import MdActBlock


# ── Validation error codes that this module can fix ──────────────────────────
_FIXABLE = {
    "ACT_NUMBER_ZERO",
    "ACT_NUMBER_ABROGATION",
    "ACT_NUMBER_MALFORMED",
    "ACT_NUMBER_PLACEHOLDER",
    "DOC_TYPE_UNKNOWN",
    "LLM_FAILED",
    "AUTHORITY_MISSING",
}

# Fields the vision model is asked to return per error code
_ERROR_TO_FIELDS: dict[str, list[str]] = {
    "ACT_NUMBER_ZERO":        ["act_number", "act_year"],
    "ACT_NUMBER_ABROGATION":  ["act_number", "act_year"],
    "ACT_NUMBER_MALFORMED":   ["act_number", "act_year"],
    "ACT_NUMBER_PLACEHOLDER": ["act_number", "act_year"],
    "DOC_TYPE_UNKNOWN":       ["doc_type", "issuing_authority"],
    "LLM_FAILED":             ["act_number", "act_year", "doc_type", "issuing_authority"],
    "AUTHORITY_MISSING":      ["issuing_authority"],
}

_VALID_DOC_TYPES = {
    "LEGE", "HOTĂRÂRE", "ORDIN", "DECIZIE", "DECRET", "ORDONANȚĂ",
    "ORDONANȚĂ DE URGENȚĂ", "COMUNICAT", "RECTIFICARE", "ANUNȚ",
    "INSTRUCȚIUNI", "REGULAMENT", "NORMĂ", "NORMATIVE", "UNKNOWN",
}

_SYSTEM = """\
Ești un expert în extragerea de date din documente oficiale românești (Monitorul Oficial).
Primești imagini ale paginilor PDF ale unui act normativ și câmpurile cu probleme.
Returnează EXCLUSIV un obiect JSON cu valorile corecte pentru câmpurile cerute — \
fără explicații, fără text în afara JSON-ului.

Reguli stricte:
- act_number: DOAR numărul propriu al actului din blocul de semnătură/titlu \
(ex: "353", "1.287"). NU folosi numere din clauze de abrogare/modificare \
(„se abrogă nr. X") și NU folosi numere din preambul („în temeiul/baza nr. X").
- act_year: anul din blocul de semnătură (ex: 2007). Dacă lipsește, folosește anul din titlul gazetei.
- doc_type: unul din: LEGE, HOTĂRÂRE, ORDIN, DECIZIE, DECRET, ORDONANȚĂ, \
ORDONANȚĂ DE URGENȚĂ, COMUNICAT, RECTIFICARE, ANUNȚ, INSTRUCȚIUNI, REGULAMENT, NORMĂ. \
Dacă nu poți determina, returnează "UNKNOWN".
- issuing_authority: instituția emitentă din antet/semnătură (ex: "MINISTERUL INTERNELOR").
"""


def render_pdf_pages(pdf_path: str | Path, page_hints: list[int], dpi: int = 200) -> list[bytes]:
    """Render specific PDF pages (0-based) to PNG bytes using PyMuPDF."""
    import fitz
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    n_pages = len(doc)
    pages_to_render = page_hints if page_hints else list(range(min(n_pages, 4)))
    images: list[bytes] = []
    for pg_idx in pages_to_render:
        if 0 <= pg_idx < n_pages:
            pix = doc[pg_idx].get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            images.append(pix.tobytes("png"))
    doc.close()
    return images


def repair_act(
    act: "LegalAct",
    block: "MdActBlock",
    pdf_path: str | Path,
    issue_codes: list[str],
    settings: "Settings | None" = None,
) -> bool:
    """Attempt to repair flagged fields using llama3.2-vision:11b.

    Mutates `act` in-place for fixed fields.
    Returns True if at least one field was patched, False otherwise.
    """
    try:
        import ollama as _ollama
    except ImportError as exc:
        _warn(act, f"repair skipped — ollama SDK not installed: {exc}")
        return False

    fixable_codes = [c for c in issue_codes if c in _FIXABLE]
    if not fixable_codes:
        return False

    # Collect the fields we need to fix (deduplicated, ordered)
    fields_needed: list[str] = []
    seen: set[str] = set()
    for code in fixable_codes:
        for f in _ERROR_TO_FIELDS.get(code, []):
            if f not in seen:
                fields_needed.append(f)
                seen.add(f)

    # Config
    cfg = getattr(settings, "repair_llm", None) if settings else None
    model = getattr(cfg, "model", "llama3.2-vision:11b") if cfg else "llama3.2-vision:11b"
    dpi   = 72   # minimum render — ~4K tokens/page, stays within 16K context

    _name = getattr(act, "act_number", "?")
    _t0 = time.time()
    print(
        f"[repair] {Path(pdf_path).stem} | act_nr={act.act_number!r} "
        f"codes={fixable_codes} fields={fields_needed} model={model}",
        file=sys.stderr, flush=True,
    )

    # Render the act's PDF pages as images
    # Cap at 1 page to stay within GLM-OCR's 16K token context window
    page_hints = list(block.page_hints)[:1] if block and block.page_hints else []
    images_bytes = render_pdf_pages(pdf_path, page_hints, dpi=dpi)
    images_bytes = images_bytes[:1]
    if not images_bytes:
        _warn(act, "repair skipped — no PDF pages could be rendered")
        return False

    # Build prompt
    current_state = {
        "act_number":        act.act_number,
        "doc_type":          act.doc_type,
        "issuing_authority": act.issuing_authority,
        "act_year":          act.act_year,
    }
    md_excerpt = (block.plain_text or "")[:500] if block else ""
    user_content = (
        f"Câmpuri cu probleme: {fixable_codes}\n"
        f"Câmpuri care trebuie returnate: {fields_needed}\n"
        f"Starea curentă (posibil greșită): {json.dumps(current_state, ensure_ascii=False)}\n\n"
        f"Extras din Markdown (extragere cu erori — referință, nu sursă de adevăr):\n"
        f"```\n{md_excerpt}\n```\n\n"
        f"INSTRUCȚIUNI STRICTE:\n"
        f"1. Folosește IMAGINILE PAGINILOR PDF ca sursă principală de adevăr.\n"
        f"2. Corectează DOAR câmpurile din {fields_needed} — nu modifica celelalte.\n"
        f"3. Nu inventa conținut care nu apare vizibil în imagine.\n"
        f"4. Păstrează structura actului; nu adăuga acte noi.\n"
        f"Returnează EXCLUSIV un obiect JSON cu exact câmpurile: {fields_needed}"
    )

    b64_images = [base64.b64encode(img).decode() for img in images_bytes]

    try:
        client = _ollama.Client()
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_content, "images": b64_images},
            ],
            options={"temperature": 0},
        )
        raw = (resp.message.content or "").strip()
    except Exception as exc:
        _warn(act, f"repair LLM call failed: {exc}")
        return False

    # Parse JSON from response
    patch = _parse_json(raw)
    if not patch:
        _warn(act, f"repair LLM returned unparseable response: {raw[:200]!r}")
        return False

    # Apply patch — only to fields that were flagged, never to valid fields
    patched: list[str] = []

    if "act_number" in fields_needed and "act_number" in patch:
        new_nr = str(patch["act_number"] or "").strip()
        if new_nr and new_nr not in ("0", "UNKNOWN", "unknown", "necunoscut"):
            act.act_number = new_nr
            patched.append(f"act_number={new_nr!r}")

    if "act_year" in fields_needed and "act_year" in patch:
        try:
            new_yr = int(patch["act_year"])
            if 1989 <= new_yr <= 2100:
                act.act_year = new_yr
                patched.append(f"act_year={new_yr}")
        except (TypeError, ValueError):
            pass

    if "doc_type" in fields_needed and "doc_type" in patch:
        new_dt = str(patch["doc_type"] or "").strip().upper()
        if new_dt and new_dt != "UNKNOWN":
            act.doc_type = new_dt
            patched.append(f"doc_type={new_dt!r}")

    if "issuing_authority" in fields_needed and "issuing_authority" in patch:
        new_auth = str(patch["issuing_authority"] or "").strip()
        if new_auth:
            act.issuing_authority = new_auth
            patched.append(f"issuing_authority={new_auth!r}")

    elapsed = time.time() - _t0
    if patched:
        msg = f"repair:{model} patched [{', '.join(patched)}] in {elapsed:.1f}s"
        act.extraction_warnings.append(msg)
        act.extraction_warnings.append("repair_flag:true")
        print(f"[repair] {Path(pdf_path).stem} | ✅ {msg}", file=sys.stderr, flush=True)
        return True
    else:
        _warn(act, f"repair LLM responded but patch was empty/invalid (raw={raw[:200]!r})")
        return False


def _parse_json(text: str) -> dict | None:
    """Extract the first valid JSON object from an LLM response."""
    # Strip all markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # Find first { … } — use non-greedy to grab the first complete object only
    for m in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and obj:
                return obj
        except json.JSONDecodeError:
            continue
    # Fallback: greedy match for nested objects
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _warn(act: "LegalAct", msg: str) -> None:
    print(f"[repair] ⚠️  {msg}", file=sys.stderr, flush=True)
    act.extraction_warnings.append(f"repair_skipped: {msg}")
