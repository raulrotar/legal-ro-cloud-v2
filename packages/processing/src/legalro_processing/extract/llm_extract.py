"""LLM-based extraction stage — all three phases.

Phase 1 — LLM metadata extraction
  ``resolve_metadata(raw_act, gazette_year, settings)``
  Replaces the regex ``extract_metadata()`` call when
  ``settings.extraction_llm.metadata_enabled`` is True.  The LLM reads only
  structured metadata fields from the per-act OCR text; the verbatim
  ``full_text`` is NEVER in the LLM output path.

Phase 2 — LLM segmentation via character-offset slicing
  ``resolve_segmentation(pages, sumar_entries, era, gazette_year, settings)``
  Returns a list of ``RawAct`` whose ``.text`` is sliced from the verbatim
  concatenated page text — never LLM-generated prose.  Only active for the
  SCANNED era (where text-based segmentation is weakest).  Falls back to
  ``None`` on any failure so the caller can use the regular segmenter.

Phase 3 — Vision-language model (VLM) on page images
  ``resolve_metadata_vlm(raw_act, gazette_year, pdf_path, settings)``
  Sends page images (rendered via fitz) to a vision-capable model and
  extracts the same ``ActMetadataLLM`` DTO as phase 1.  Only called for the
  SCANNED era when ``settings.extraction_llm.vlm_enabled`` is True.
  The verbatim text still comes from OCR; VLM output is metadata only.

All public functions return ``None`` (or fall back to regex) on any validation
failure, so the pipeline is fully resilient.  Every outcome is recorded in
``LegalAct.extraction_warnings`` for audit/A-B attribution.

Provider abstraction
  All calls go through ``legalro_core.llm_client.call_llm`` which speaks the
  OpenAI /chat/completions protocol.  Swap the backend by changing
  ``settings.extraction_llm.{base_url,model,api_key}`` — cloud Gemini, a
  local MLX server, or a rented-GPU vLLM node are all supported.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Literal, Optional, Any

from pydantic import BaseModel, ValidationError, field_validator

from legalro_core.llm_client import call_llm
from legalro_processing.extract.metadata import (
    extract_metadata,
    AUTHORITY_PATTERNS,
    CLOSING_BLOCK,
    BARE_NR,
    _extract_locality,
)
from legalro_processing.extract.segment import RawAct

# Lazy import — avoid pulling in fitz at module level so servers that don't
# need VLM don't pay the PyMuPDF import cost.
_fitz = None


def _get_fitz():
    global _fitz
    if _fitz is None:
        import fitz as _f
        _fitz = _f
    return _fitz


# ── Allowed doc_type values ────────────────────────────────────────────────────

_DOC_TYPES = Literal[
    "LEGE", "HG", "OUG", "ORDONANȚĂ", "DECRET", "DECRET_LEGE", "DCC",
    "DECIZIE", "ORDIN", "COMUNICAT", "RAPORT", "ANUNT", "RECTIFICARE", "UNKNOWN",
]

_DOC_TYPE_SET = {
    "LEGE", "HG", "OUG", "ORDONANȚĂ", "DECRET", "DECRET_LEGE", "DCC",
    "DECIZIE", "ORDIN", "COMUNICAT", "RAPORT", "ANUNT", "RECTIFICARE", "UNKNOWN",
}

# ── Pydantic DTOs — separate from LegalAct so a bad LLM response can never
#    corrupt the dataclass directly ─────────────────────────────────────────────

class ActMetadataLLM(BaseModel):
    """Structured metadata extracted by the LLM for one legal act."""
    doc_type: str
    act_number: str = ""
    act_year: Optional[int] = None
    issuing_authority: str = ""
    title: str = ""
    locality: Optional[str] = None

    @field_validator("doc_type")
    @classmethod
    def validate_doc_type(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in _DOC_TYPE_SET:
            raise ValueError(f"doc_type {v!r} not in allowed set")
        return v

    @field_validator("act_number")
    @classmethod
    def clean_act_number(cls, v: str) -> str:
        # Remove any trailing "." that crept in from Romanian number formatting.
        return v.strip().rstrip(".")

    @field_validator("title")
    @classmethod
    def strip_dot_leaders(cls, v: str) -> str:
        # If the LLM still returned a TOC fragment with dot-leaders, strip them.
        clean = re.sub(r'\.{3,}.*$', '', v).strip()
        return clean if clean else v.strip()


class ActBoundaryLLM(BaseModel):
    """Character-offset boundary for one act within the concatenated OCR text."""
    char_start: int
    char_end: int
    sumar_index: Optional[int] = None   # index into sumar_entries this act maps to
    title: str = ""                     # carry the sumar title hint forward


# ── Prompt templates ──────────────────────────────────────────────────────────

_METADATA_SYSTEM = """\
Ești un asistent specializat în extragerea metadatelor din actele normative
publicate în Monitorul Oficial al României (MO).

Sarcina ta este să analizezi textul unui act normativ (extras prin OCR) și să
returnezi un obiect JSON cu câmpurile cerute. Nu inventa nicio informație care
nu apare explicit în text.

Reguli stricte:
1. doc_type trebuie să fie EXACT unul din: LEGE, HG, OUG, ORDONANȚĂ, DECRET,
   DECRET_LEGE, DCC, DECIZIE, ORDIN, COMUNICAT, RAPORT, ANUNT, RECTIFICARE, UNKNOWN.
   - HG = Hotărâre de Guvern
   - DCC = Decizie a Curții Constituționale (emitent: Curtea Constituțională)
   - DECIZIE = Decizie a Prim-Ministrului sau a unui minister/agenție
   - ORDIN = Ordin al unui minister/agenție
   Dacă nu poți determina tipul, folosește UNKNOWN.
2. act_number = numărul actului din blocul de semnare de la final:
   „București, ZZ LUNA AAAA. Nr. NNN." — extrage NNN.
   Dacă nu există bloc de semnare, caută „Nr. NNN" în antet.
   Returnează DOAR cifre (și puncte pentru numerele compuse, ex. „1.642").
3. act_year = anul din blocul de semnare (AAAA). Dacă lipsește, null.
4. issuing_authority = instituția emitentă. Caută în antetul actului SAU în
   blocul de semnare. Returnează denumirea completă oficială în română.
   Exemple: „Curtea Constituțională", „Guvernul României",
   „Agenția Națională de Cadastru și Publicitate Imobiliară",
   „Ministerul Finanțelor".
5. title = titlul descriptiv al actului. ATENȚIE: NU folosi linia din sumarul
   (cuprins) care conține puncte de ghidaj „......". Titlul corect apare în
   corpul actului, imediat după cuvântul-cheie (ex. ORDIN, HOTĂRÂRE) și
   înaintea primului articol. Poate începe cu „privind", „pentru", „referitoare
   la" etc. Dacă titlul depășește 300 de caractere, trunchiază la 300.
6. locality = județul menționat explicit (ex. „județul Ilfov"), sau null dacă
   actul are aplicabilitate națională.

Returnează EXCLUSIV un obiect JSON valid, fără text suplimentar, fără markdown.
Schema exactă:
{
  "doc_type": "...",
  "act_number": "...",
  "act_year": <int sau null>,
  "issuing_authority": "...",
  "title": "...",
  "locality": <"..." sau null>
}
"""

_METADATA_USER_TMPL = """\
Gazette year: {gazette_year}
Sumar hint (TOC line — poate conține puncte de ghidaj, NU folosi ca titlu):
{sumar_hint}

TEXT ACT (primele ~2500 caractere + ultimele ~800 caractere):
--- INCEPUT ---
{head}
--- ... ---
{tail}
--- SFARSIT ---

Extrage metadatele și returnează JSON.
"""

_SEGMENTATION_SYSTEM = """\
Ești un asistent specializat în segmentarea textului Monitorului Oficial al
României în acte normative individuale.

Primești textul complet al unui număr de Monitorul Oficial (obținut prin OCR)
și o listă cu actele așteptate din sumar. Trebuie să returnezi pozițiile
exacte (char_start, char_end) din textul concatenat unde începe și se termină
fiecare act.

Reguli:
1. char_start și char_end sunt indecși în șirul complet primit (0-based,
   slice Python: text[char_start:char_end]).
2. Fiecare act trebuie să acopere exact textul acelui act — de la antetul
   instituției emitente până la blocul de semnare inclusiv
   („București, ... Nr. ...").
3. NU suprapune actele. Acoperă tot textul (fiecare caracter trebuie să
   aparțină unui act).
4. Returnează un array JSON ordonat cu obiectele:
   [{"char_start": N, "char_end": M, "sumar_index": I, "title": "..."}, ...]
   unde sumar_index este indexul 0-based al actului din lista din sumar.
5. Returnează EXCLUSIV array JSON valid, fără text suplimentar.
"""

_SEGMENTATION_USER_TMPL = """\
Gazette year: {gazette_year}

Acte așteptate din sumar ({n_sumar} acte):
{sumar_list}

TEXT COMPLET (lungime {text_len} caractere):
{full_text}

Returnează array JSON cu granițele fiecărui act.
"""

_VLM_METADATA_SYSTEM = _METADATA_SYSTEM  # same rules; VLM reads images

_VLM_METADATA_USER_TMPL = """\
Gazette year: {gazette_year}
Sumar hint: {sumar_hint}

Analizează imaginile de mai jos (pagini din Monitorul Oficial scanate).
Extrage metadatele actului normativ și returnează JSON conform schemei.
"""


# ── Helper: resolve effective extraction LLM config ──────────────────────────

def _effective_config(settings) -> dict[str, Any]:
    """Return {base_url, model, api_key, temperature, max_tokens, max_retries}
    resolving empty extraction_llm fields from settings.llm."""
    ecfg = settings.extraction_llm
    base_url = ecfg.base_url or settings.llm.base_url
    model    = ecfg.model    or settings.llm.model
    api_key  = ecfg.api_key  or settings.llm.api_key
    return {
        "base_url":    base_url,
        "model":       model,
        "api_key":     api_key,
        "temperature": ecfg.temperature,
        "max_tokens":  ecfg.max_tokens,
        "max_retries": ecfg.max_retries,
    }


# ── Authority tag derivation (deterministic, never from LLM) ─────────────────

def _derive_authority_tag(authority_name: str) -> str:
    """Derive the short authority_tag from the display name using the existing
    AUTHORITY_PATTERNS table — ensures law_id is always deterministic."""
    if not authority_name:
        return ""
    for _name, tag, pattern in AUTHORITY_PATTERNS:
        if pattern.search(authority_name):
            return tag
    # Generic fallback: first 6 lowercase letters of first word
    return re.sub(r'[^a-z]', '', authority_name.lower())[:6]


# ── Convert ActMetadataLLM → the dict shape extract_metadata returns ──────────

def _llm_dto_to_meta_dict(dto: ActMetadataLLM, gazette_year: int) -> dict:
    authority_tag = _derive_authority_tag(dto.issuing_authority)
    act_year = dto.act_year or gazette_year
    type_slug = dto.doc_type.lower()
    if authority_tag:
        law_id = f"{type_slug}_{authority_tag}_{dto.act_number}_{act_year}_v1"
    else:
        law_id = f"{type_slug}_{dto.act_number}_{act_year}_v1"

    return {
        "doc_type":          dto.doc_type,
        "act_number":        dto.act_number,
        "act_year":          act_year,
        "issuing_authority": dto.issuing_authority,
        "authority_tag":     authority_tag,
        "locality":          dto.locality,
        "title":             dto.title,
        "law_id":            law_id,
        "_via":              "llm_phase1",
    }


# ── Field-level fallback: prefer regex for specific fields if LLM missed them ─

def _apply_field_fallbacks(llm_meta: dict, raw_act: RawAct, gazette_year: int) -> dict:
    """For fields the LLM left empty, fall back to the specific regex that
    extract_metadata uses — without re-running the whole regex path."""
    text = raw_act.text

    # act_number: if LLM missed it, use CLOSING_BLOCK then BARE_NR
    if not llm_meta.get("act_number") or llm_meta["act_number"] == "0":
        matches = list(CLOSING_BLOCK.finditer(text))
        if matches:
            m = matches[-1]
            llm_meta["act_number"] = m.group(2).replace(".", "")
            llm_meta["act_year"]   = int(m.group(1))
            llm_meta["_via"] += "+regex_nr"
        else:
            bare = list(BARE_NR.finditer(text))
            if bare:
                llm_meta["act_number"] = bare[-1].group(1).replace(".", "")
                llm_meta["_via"] += "+regex_bare_nr"

    # locality: always use the regex (LLM may hallucinate counties)
    if not llm_meta.get("locality"):
        llm_meta["locality"] = _extract_locality(text) or None

    return llm_meta


# ── Phase 1: LLM metadata extraction ─────────────────────────────────────────

def resolve_metadata(
    raw_act: RawAct,
    gazette_year: int,
    settings,
    era=None,
) -> dict:
    """Router: returns LLM-extracted metadata dict when enabled, else falls back
    to the deterministic ``extract_metadata()`` regex path.

    Phase 3 (VLM) is NOT called here — it requires the PDF path and is
    invoked separately from gazette_extractor when vlm_enabled=True.

    The returned dict always has the same keys as ``extract_metadata()``
    returns so ``_build_act()`` is unaffected.
    """
    ecfg = settings.extraction_llm if settings else None

    if not ecfg or not ecfg.enabled or not ecfg.metadata_enabled:
        return extract_metadata(raw_act, gazette_year)

    # Build the user prompt
    text = raw_act.text
    head = text[:2500] if len(text) > 2500 else text
    tail = text[-800:] if len(text) > 3300 else ""  # show tail only when there's a gap
    sumar_hint = raw_act.title or "(nu există hint din sumar)"

    user_msg = _METADATA_USER_TMPL.format(
        gazette_year=gazette_year,
        sumar_hint=sumar_hint,
        head=head,
        tail=tail,
    )

    cfg = _effective_config(settings)
    try:
        raw_json = call_llm(
            messages=[
                {"role": "system", "content": _METADATA_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg["api_key"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            json_mode=True,
            timeout=60.0,
            max_retries=cfg["max_retries"],
        )
        data = json.loads(raw_json)
        dto = ActMetadataLLM(**data)
    except (Exception, ValidationError) as exc:
        # Validation or network failure → regex fallback
        meta = extract_metadata(raw_act, gazette_year)
        meta["_via"] = f"regex_fallback(llm_failed:{type(exc).__name__})"
        meta.setdefault("extraction_warnings", []).append(
            f"LLM metadata failed ({type(exc).__name__}): {str(exc)[:120]}; regex used"
        )
        return meta

    # Convert DTO → dict
    llm_meta = _llm_dto_to_meta_dict(dto, gazette_year)
    llm_meta = _apply_field_fallbacks(llm_meta, raw_act, gazette_year)
    llm_meta.setdefault("extraction_warnings", []).append("metadata via LLM phase1")
    return llm_meta


# ── Phase 2: LLM segmentation via character offsets ───────────────────────────

_MAX_SEGMENTATION_CHARS = 80_000   # truncate very long gazettes to avoid token overflow


def resolve_segmentation(
    pages: list[str],
    sumar_entries,
    era,
    gazette_year: int,
    settings,
) -> list[RawAct] | None:
    """Router for segmentation: returns LLM-derived RawAct list or None.

    When None is returned, the caller should use the standard regex
    ``segment_acts()`` path.  The returned RawAct texts are sliced from the
    verbatim concatenated page text — no LLM-generated prose enters full_text.

    Only active for the SCANNED era and when
    ``settings.extraction_llm.segmentation_enabled`` is True.
    """
    try:
        from legalro_core.models import Era
        if era != Era.SCANNED:
            return None
    except Exception:
        return None

    ecfg = settings.extraction_llm if settings else None
    if not ecfg or not ecfg.enabled or not ecfg.segmentation_enabled:
        return None

    if not sumar_entries:
        return None

    concat_text = "\n".join(pages)
    text_for_llm = concat_text[:_MAX_SEGMENTATION_CHARS]

    sumar_list = "\n".join(
        f"{i}. {getattr(e, 'title', str(e))}"
        for i, e in enumerate(sumar_entries)
    )

    user_msg = _SEGMENTATION_USER_TMPL.format(
        gazette_year=gazette_year,
        n_sumar=len(sumar_entries),
        sumar_list=sumar_list,
        text_len=len(concat_text),
        full_text=text_for_llm,
    )

    cfg = _effective_config(settings)
    try:
        raw_json = call_llm(
            messages=[
                {"role": "system", "content": _SEGMENTATION_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg["api_key"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            json_mode=True,
            timeout=120.0,
            max_retries=cfg["max_retries"],
        )
        boundaries_raw = json.loads(raw_json)
        if not isinstance(boundaries_raw, list):
            # Some providers wrap the array in a key
            boundaries_raw = next(
                (v for v in boundaries_raw.values() if isinstance(v, list)), []
            )
        boundaries = [ActBoundaryLLM(**b) for b in boundaries_raw]
    except Exception as exc:
        print(f"[llm_extract] segmentation LLM failed ({type(exc).__name__}): {exc}; regex fallback", flush=True)
        return None

    if not boundaries:
        return None

    # Validate: coverage_ratio — sliced fragments must cover the text
    slices = [concat_text[b.char_start:b.char_end] for b in boundaries if b.char_start < b.char_end]
    coverage = _coverage_ratio(concat_text, slices)
    if coverage < 0.9:
        print(f"[llm_extract] segmentation rejected: coverage={coverage:.2f} < 0.90; regex fallback", flush=True)
        return None

    # Validate act count
    if len(boundaries) < 1:
        return None

    raw_acts: list[RawAct] = []
    for pos, b in enumerate(boundaries):
        char_start = max(0, b.char_start)
        char_end   = min(len(concat_text), b.char_end)
        slice_text = concat_text[char_start:char_end].strip()
        if not slice_text:
            continue
        title = b.title or (
            getattr(sumar_entries[b.sumar_index], "title", "")
            if b.sumar_index is not None and b.sumar_index < len(sumar_entries)
            else ""
        )
        raw_acts.append(RawAct(
            text=slice_text,
            title=title,
            page_range=[],
            position_in_gazette=pos,
        ))

    print(f"[llm_extract] segmentation: produced {len(raw_acts)} acts (sumar={len(sumar_entries)})", flush=True)
    return raw_acts if raw_acts else None


def _coverage_ratio(source: str, slices: list[str]) -> float:
    """Fraction of unique source characters covered by the slices.

    Uses a simple set-of-offsets approach so the ratio is meaningful even
    when slices overlap or have small gaps.
    """
    if not source:
        return 1.0
    covered: set[int] = set()
    offset = 0
    for s in slices:
        idx = source.find(s, offset)
        if idx == -1:
            # Slice not found at expected offset — try from start
            idx = source.find(s)
        if idx != -1:
            covered.update(range(idx, idx + len(s)))
            offset = idx + len(s)
    return len(covered) / len(source)


# ── Phase 3: VLM on page images (SCANNED era) ─────────────────────────────────

_MAX_VLM_PAGES = 4   # cap to avoid huge payloads; first N pages of the act


def resolve_metadata_vlm(
    raw_act: RawAct,
    gazette_year: int,
    pdf_path: str,
    settings,
) -> dict | None:
    """Vision-language model metadata extraction for scanned acts.

    Renders page images from the act's page_range (via fitz), encodes them
    as base64 PNG, and sends to a vision-capable model via the OpenAI
    multipart content API.  Returns the same dict shape as
    ``extract_metadata()`` or None on failure (caller falls back to
    regex/phase-1).

    Only called when:
      - ``settings.extraction_llm.vlm_enabled`` is True
      - era == Era.SCANNED
      - The model supports vision (image_url content parts)
    """
    ecfg = settings.extraction_llm if settings else None
    if not ecfg or not ecfg.enabled or not ecfg.vlm_enabled:
        return None

    # Render page images
    fitz = _get_fitz()
    try:
        doc = fitz.open(str(pdf_path))
        page_range = raw_act.page_range or list(range(min(_MAX_VLM_PAGES, len(doc))))
        pages_to_render = page_range[:_MAX_VLM_PAGES]

        image_parts: list[dict] = []
        for page_no in pages_to_render:
            if page_no >= len(doc):
                break
            page = doc[page_no]
            mat = fitz.Matrix(1.5, 1.5)   # 1.5× zoom — legible without huge payload
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("ascii")
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        doc.close()
    except Exception as exc:
        print(f"[llm_extract] VLM image render failed: {exc}", flush=True)
        return None

    if not image_parts:
        return None

    sumar_hint = raw_act.title or "(nu există hint din sumar)"
    user_text = _VLM_METADATA_USER_TMPL.format(
        gazette_year=gazette_year,
        sumar_hint=sumar_hint,
    )

    # Build multipart user message: text + images
    user_content: list[dict] = [{"type": "text", "text": user_text}] + image_parts

    cfg = _effective_config(settings)
    try:
        raw_json = call_llm(
            messages=[
                {"role": "system", "content": _VLM_METADATA_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg["api_key"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            json_mode=True,
            timeout=90.0,
            max_retries=cfg["max_retries"],
        )
        data = json.loads(raw_json)
        dto = ActMetadataLLM(**data)
    except (Exception, ValidationError) as exc:
        print(f"[llm_extract] VLM metadata failed ({type(exc).__name__}): {exc}", flush=True)
        return None

    llm_meta = _llm_dto_to_meta_dict(dto, gazette_year)
    llm_meta = _apply_field_fallbacks(llm_meta, raw_act, gazette_year)
    llm_meta["_via"] = llm_meta.get("_via", "llm").replace("llm_phase1", "llm_phase3_vlm")
    llm_meta.setdefault("extraction_warnings", []).append("metadata via VLM phase3")
    return llm_meta
