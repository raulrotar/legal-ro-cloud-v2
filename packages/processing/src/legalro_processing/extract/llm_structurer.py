"""LLM structured extraction from per-act Markdown (Option C).

Takes a MdActBlock (Markdown for one act) and returns a fully populated
ActExtractionLLM DTO including:
  - All Option B metadata fields (doc_type, act_number, etc.)
  - full_text_corrected: the act body with OCR artifacts fixed (broken
    diacritics, mojibake, letterspaced headers) — but NOT rewritten.

Hallucination guard
-------------------
The LLM is allowed to make small corrections (diacritics: ă/â/î/ș/ț,
mojibake: c'deri→căderi, Rom,nia→România, letter-spacing: D E C R E T→DECRET).
It must NOT paraphrase, summarize, or rewrite legal clauses.

Guard: edit_distance(source_plain, full_text_corrected) / len(source_plain)
  - ≤ threshold (default 0.15)  → accept
  - > threshold                 → reject full_text_corrected; use source_plain

Provider
--------
Uses legalro_core.llm_client.call_llm — same OpenAI-compatible interface
as Option B.  Works with NuExtract 3 (4B, recommended), Qwen2.5-7B,
Gemini, or any instruction-tuned model served via Ollama/vLLM.

NuExtract 3 tip: it natively supports a JSON schema template in the
system prompt — pass the schema in the system message and it follows it
precisely, reducing hallucination on structured extraction tasks.
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional, Any

from pydantic import BaseModel, ValidationError, field_validator

from legalro_core.llm_client import call_llm, loads_lenient
from legalro_processing.extract.metadata import (
    AUTHORITY_PATTERNS,
    CLOSING_BLOCK,
    BARE_NR,
    _extract_locality,
)
from legalro_processing.extract.md_segmenter import MdActBlock, _md_to_plain

# ── DTO ───────────────────────────────────────────────────────────────────────

_DOC_TYPE_SET = {
    "LEGE", "HG", "OUG", "ORDONANȚĂ", "DECRET", "DECRET_LEGE", "DCC",
    "DECIZIE", "ORDIN", "COMUNICAT", "RAPORT", "ANUNT", "RECTIFICARE", "UNKNOWN",
}

# Romanian act-type words → canonical enum tag.
# Applied BEFORE Pydantic validation so a stray "HOTĂRÂRE" doesn't trigger
# a silent regex fallback (the validator raises on any non-enum value).
_DOC_TYPE_ALIASES: dict[str, str] = {
    "HOTĂRÂRE": "HG",
    "HOTARARE": "HG",
    "HOTĂRÂREA": "HG",
    "HOTĂRÂRE DE GUVERN": "HG",
    "ORDONANȚĂ DE URGENȚĂ": "OUG",
    "ORDONANTA DE URGENTA": "OUG",
    "ORDONANTA": "ORDONANȚĂ",
    "DECRET-LEGE": "DECRET_LEGE",
    "DECRET LEGE": "DECRET_LEGE",
    "DECIZIE A CURȚII CONSTITUȚIONALE": "DCC",
    "DECIZIA CURȚII CONSTITUȚIONALE": "DCC",
    "DECIZIE CURTEA CONSTITUȚIONALĂ": "DCC",
    "RECTIFICĂRI": "RECTIFICARE",
    "RECTIFICARI": "RECTIFICARE",
    "ANUNȚ": "ANUNT",
    "ANUNT": "ANUNT",
}


class ActExtractionLLM(BaseModel):
    """Full act extraction DTO — metadata + corrected body text."""
    doc_type: str
    act_number: Optional[str] = ""
    act_year: Optional[int] = None
    issuing_authority: Optional[str] = ""
    title: Optional[str] = ""
    locality: Optional[str] = None
    full_text_corrected: Optional[str] = ""   # OCR-corrected body text; empty = use source

    @field_validator("doc_type", mode="before")
    @classmethod
    def normalize_doc_type(cls, v: object) -> str:
        v = str(v).strip().upper()
        return _DOC_TYPE_ALIASES.get(v, v)

    @field_validator("doc_type")
    @classmethod
    def validate_doc_type(cls, v: str) -> str:
        if v not in _DOC_TYPE_SET:
            raise ValueError(f"doc_type {v!r} not in allowed set")
        return v

    @field_validator("act_number", "issuing_authority", "title", "full_text_corrected", mode="before")
    @classmethod
    def coerce_none_to_empty(cls, v: object) -> str:
        """NuExtract 3 returns null for optional fields — coerce to empty string."""
        return "" if v is None else str(v)

    @field_validator("act_number")
    @classmethod
    def clean_number(cls, v: str) -> str:
        # Strip trailing dots; strip "Nr." prefix NuExtract sometimes adds
        v = re.sub(r'^[Nn][rR]\.?\s*', '', v).strip().rstrip(".")
        return v

    @field_validator("title")
    @classmethod
    def strip_dot_leaders(cls, v: str) -> str:
        clean = re.sub(r'\.{3,}.*$', '', v).strip()
        return clean if clean else v.strip()


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
Ești un specialist în analiza actelor normative din Monitorul Oficial al României.

Primești un bloc Markdown extras prin OCR dintr-un act normativ.
Trebuie să returnezi un obiect JSON cu câmpurile cerute.

Reguli stricte:
1. doc_type: EXACT unul din: LEGE, HG, OUG, ORDONANȚĂ, DECRET, DECRET_LEGE,
   DCC, DECIZIE, ORDIN, COMUNICAT, RAPORT, ANUNT, RECTIFICARE, UNKNOWN.
   HG = Hotărâre de Guvern. DCC = Decizie Curtea Constituțională.
2. act_number: numărul din blocul „Nr. NNN." de la final. Doar cifre și puncte.
3. act_year: anul din „București, ZZ LUNA AAAA." (sau null).
4. issuing_authority: instituția emitentă (antet sau semnătură). Denumire oficială completă.
5. title: titlul descriptiv din corpul actului, NU din cuprins (fără „......").
   Începe cu „privind", „pentru", „referitoare la" etc.
6. locality: județul menționat explicit, sau null.
7. full_text_corrected: corpul complet al actului cu CORECȚII MINIME:
   - Caractere lipsă/greșite din OCR: ă/â/î/ș/ț (ex: „c'deri"→„căderi")
   - Mojibake specific românesc: „'"->" ă", „,"→„ă", „ã"→„ă" etc.
   - Litere cu spații: „D E C R E T"→„DECRET", „H O T Ă R Â R E"→„HOTĂRÂRE"
   - NU reformula, NU rezuma, NU adăuga informații noi.
   - Dacă textul este deja corect, copiază-l identic.
   - Păstrează toate articolele, alineatele, tabelele și semnăturile.
   - Lungimea full_text_corrected trebuie să fie apropiată de lungimea sursei.

Returnează EXCLUSIV JSON valid, fără text suplimentar, fără markdown.
Schema exactă:
{
  "doc_type": "...",
  "act_number": "...",
  "act_year": <int|null>,
  "issuing_authority": "...",
  "title": "...",
  "locality": <"..."|null>,
  "full_text_corrected": "..."
}
"""

_USER_TMPL = """\
Gazette year: {gazette_year}
Hint din sumar (poate conține puncte de ghidaj — NU folosi ca titlu): {sumar_hint}

MARKDOWN ACT (primele ~2500 + ultimele ~800 caractere):
=== INCEPUT ===
{head}
--- ... ---
{tail}
=== SFARSIT ===

Extrage metadatele și returnează JSON.
"""

# Used when act is too long for full_text_corrected — metadata only
_SYSTEM_META_ONLY = """\
Ești un specialist în analiza actelor normative din Monitorul Oficial al României.

Primești un bloc Markdown extras dintr-un act normativ.
Trebuie să returnezi un obiect JSON DOAR cu câmpurile de metadate (fără full_text_corrected).

Reguli stricte:
1. doc_type: EXACT unul din: LEGE, HG, OUG, ORDONANȚĂ, DECRET, DECRET_LEGE,
   DCC, DECIZIE, ORDIN, COMUNICAT, RAPORT, ANUNT, RECTIFICARE, UNKNOWN.
   HG = Hotărâre de Guvern. DCC = Decizie Curtea Constituțională.
2. act_number: numărul din blocul „Nr. NNN." de la final. Doar cifre și puncte.
   Pentru ORDIN caută și în antet „ORDIN nr. NNN din". Fără text suplimentar.
3. act_year: anul din „București, ZZ LUNA AAAA." (sau null).
4. issuing_authority: instituția emitentă (antet sau semnătură). Denumire oficială completă.
5. title: titlul descriptiv din corpul actului, NU din cuprins (fără „......").
   Începe cu „privind", „pentru", „referitoare la" etc.
6. locality: județul menționat explicit, sau null.

Returnează EXCLUSIV JSON valid, fără text suplimentar, fără markdown.
Schema exactă:
{
  "doc_type": "...",
  "act_number": "...",
  "act_year": <int|null>,
  "issuing_authority": "...",
  "title": "...",
  "locality": <"..."|null>
}
"""

# Threshold: above this many chars, skip full_text_corrected to avoid token truncation
_LONG_ACT_THRESHOLD = 4000


# ── NuExtract 3 template (used when model name contains "nuextract3") ─────────
# NuExtract 3 uses a JSON schema template passed via extra_body.chat_template_kwargs
# instead of a system prompt. Works with vLLM; mlx_lm.server uses standard prompts.

_NUEXTRACT3_TEMPLATE = {
    "doc_type": ["LEGE", "HG", "OUG", "ORDONANȚĂ", "DECRET", "DECRET_LEGE",
                 "DCC", "DECIZIE", "ORDIN", "COMUNICAT", "RAPORT", "ANUNT",
                 "RECTIFICARE", "UNKNOWN"],
    "act_number": "verbatim-string",
    "act_year": "integer",
    "issuing_authority": "string",
    "title": "string",
    "locality": "string",
    "full_text_corrected": "verbatim-string",
}

_NUEXTRACT3_INSTRUCTIONS = (
    "Extrage câmpurile din actul normativ românesc. "
    "doc_type: EXACT din lista — HG=Hotărâre de Guvern, DCC=Decizie Curtea Constituțională, "
    "OUG=Ordonanță de Urgență; folosește DOAR valorile enumerate. "
    "act_number: numărul actului — caută 'Nr. NNN.' în blocul de la final AL actului; "
    "pentru ORDIN caută și în antet ('ORDIN nr. NNN din'); returnează DOAR cifre, fără punct. "
    "act_year: anul din 'București, ZZ LUNA AAAA.' sau din antet. "
    "issuing_authority: denumirea oficială completă a instituției emitente. "
    "title: titlul descriptiv din corpul actului (nu din cuprins, fără '......'); "
    "începe cu 'privind', 'pentru', 'referitoare la' etc. "
    "locality: județul menționat explicit în text, sau null. "
    "full_text_corrected: corpul COMPLET al actului cu corecții minime OCR "
    "(diacritice lipsă, mojibake românesc); NU reformula, NU rezuma, NU omite paragrafe."
)


def _is_nuextract3(model: str) -> bool:
    return "nuextract3" in model.lower() or "nuextract-3" in model.lower()


def _nuextract3_call(block: MdActBlock, gazette_year: int, cfg: dict) -> str:
    """Call NuExtract 3 via vLLM using its native template format."""
    text = block.markdown
    head = text[:2500] if len(text) > 2500 else text
    tail = text[-800:] if len(text) > 3300 else ""
    doc_text = f"Gazette year: {gazette_year}\n\n{head}"
    if tail:
        doc_text += f"\n--- ... ---\n{tail}"

    return call_llm(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": doc_text}],
            }
        ],
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_key=cfg["api_key"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        json_mode=False,
        timeout=120.0,
        max_retries=cfg["max_retries"],
        extra_body={
            "chat_template_kwargs": {
                "template": json.dumps(_NUEXTRACT3_TEMPLATE),
                "instructions": _NUEXTRACT3_INSTRUCTIONS,
                "enable_thinking": False,
            }
        },
    )


# ── Public API ────────────────────────────────────────────────────────────────

def structure_act(
    block: MdActBlock,
    gazette_year: int,
    settings,
    edit_distance_threshold: float = 0.15,
) -> dict:
    """Extract structured metadata + corrected full_text from a MdActBlock.

    Returns a dict with the same keys as extract_metadata() plus '_via'.
    Falls back to the regex path on any LLM/validation failure.
    """
    from legalro_processing.extract.metadata import extract_metadata
    from legalro_processing.extract.segment import RawAct

    # Build a RawAct from plain text for the regex fallback
    raw_act = RawAct(
        text=block.plain_text,
        title=block.title_hint,
        page_range=[],
        position_in_gazette=0,
    )

    ecfg = getattr(settings, "extraction_llm", None) if settings else None
    if not ecfg or not ecfg.enabled:
        meta = extract_metadata(raw_act, gazette_year)
        meta["full_text_corrected"] = block.plain_text
        return meta

    # Prepare prompt — long acts skip full_text_corrected to avoid token truncation.
    # Acts over _LONG_ACT_THRESHOLD chars would require the LLM to reproduce thousands
    # of tokens verbatim, risking truncated JSON. Use source text directly instead.
    text = block.markdown
    is_long_act = len(text) > _LONG_ACT_THRESHOLD
    head = text[:2500] if len(text) > 2500 else text
    tail = text[-800:] if len(text) > 3300 else ""

    user_msg = _USER_TMPL.format(
        gazette_year=gazette_year,
        sumar_hint=block.title_hint or "(fără hint)",
        head=head,
        tail=tail,
    )

    cfg = _effective_config(settings)
    try:
        if _is_nuextract3(cfg["model"]):
            raw_json = _nuextract3_call(block, gazette_year, cfg)
        else:
            system_prompt = _SYSTEM_META_ONLY if is_long_act else _SYSTEM
            raw_json = call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
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
        data = loads_lenient(raw_json)
        dto = ActExtractionLLM(**data)
    except (Exception, ValidationError) as exc:
        meta = extract_metadata(raw_act, gazette_year)
        meta["full_text_corrected"] = block.plain_text
        meta["_via"] = f"regex_fallback(llm_failed:{type(exc).__name__})"
        meta.setdefault("extraction_warnings", []).append(
            f"LLM structuring failed: {str(exc)[:120]}"
        )
        return meta

    # Validate full_text_corrected with edit-distance guard.
    # For long acts we used metadata-only mode — full_text_corrected is empty by design.
    corrected = (dto.full_text_corrected or "").strip()
    source = block.plain_text.strip()
    if is_long_act or not corrected:
        accepted_text = source
        text_via = "source_plain(long_act)" if is_long_act else "source_plain(empty)"
    elif _edit_ratio(source, corrected) <= edit_distance_threshold:
        accepted_text = corrected
        text_via = "llm_corrected"
    else:
        accepted_text = source
        text_via = "source_plain(hallucination_rejected)"

    meta = _dto_to_meta(dto, gazette_year, accepted_text)
    meta["_via"] = f"llm_option_c+{text_via}"
    meta.setdefault("extraction_warnings", []).append(f"structured via Option C LLM ({text_via})")

    # Field-level fallback for act_number
    if not meta.get("act_number") or meta["act_number"] == "0":
        m_close = list(CLOSING_BLOCK.finditer(source))
        if m_close:
            meta["act_number"] = m_close[-1].group(2).replace(".", "")
            meta["act_year"] = int(m_close[-1].group(1))
        else:
            # End-of-line Nr. fallback: matches "Nr. 22." on its own line.
            # More precise than BARE_NR which can hit legal references like
            # "Hotărârea CSM nr. 5 din...".
            m_eol = re.search(r'(?:^|\n)\s*Nr\.\s*([\d.]+)\.\s*$', source, re.MULTILINE)
            if m_eol:
                meta["act_number"] = m_eol.group(1)
            else:
                bare = list(BARE_NR.finditer(source))
                if bare:
                    meta["act_number"] = bare[-1].group(1).replace(".", "")

    if not meta.get("locality"):
        meta["locality"] = _extract_locality(source) or None

    return meta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _effective_config(settings) -> dict[str, Any]:
    ecfg = settings.extraction_llm
    return {
        "base_url":    ecfg.base_url or settings.llm.base_url,
        "model":       ecfg.model    or settings.llm.model,
        "api_key":     ecfg.api_key  or settings.llm.api_key,
        "temperature": ecfg.temperature,
        "max_tokens":  ecfg.max_tokens,
        "max_retries": ecfg.max_retries,
    }


def _derive_authority_tag(name: str) -> str:
    if not name:
        return ""
    for _n, tag, pattern in AUTHORITY_PATTERNS:
        if pattern.search(name):
            return tag
    return re.sub(r'[^a-z]', '', name.lower())[:6]


_CURTEA_RE = re.compile(r'CURTEA\s+CONSTITU[TȚ]IONAL[AĂ]', re.IGNORECASE)


def _dto_to_meta(dto: ActExtractionLLM, gazette_year: int, full_text: str) -> dict:
    # Post-process: upgrade DECIZIE → DCC when Curtea Constituțională is the authority.
    # Llama reliably identifies the authority but sometimes returns DECIZIE instead of DCC.
    doc_type = dto.doc_type
    if doc_type == "DECIZIE" and _CURTEA_RE.search(dto.issuing_authority or ""):
        doc_type = "DCC"

    authority_tag = _derive_authority_tag(dto.issuing_authority)
    act_year = dto.act_year or gazette_year
    type_slug = doc_type.lower()
    law_id = (
        f"{type_slug}_{authority_tag}_{dto.act_number}_{act_year}_v1"
        if authority_tag
        else f"{type_slug}_{dto.act_number}_{act_year}_v1"
    )
    return {
        "doc_type":            doc_type,
        "act_number":          dto.act_number,
        "act_year":            act_year,
        "issuing_authority":   dto.issuing_authority,
        "authority_tag":       authority_tag,
        "locality":            dto.locality,
        "title":               dto.title,
        "law_id":              law_id,
        "full_text_corrected": full_text,
        "extraction_warnings": [],
    }


def _edit_ratio(a: str, b: str) -> float:
    """Approximate edit distance ratio using difflib (fast, no extra deps)."""
    import difflib
    if not a:
        return 0.0
    # SequenceMatcher.ratio() = 2*M / T where M=matches, T=total elements
    # We convert to edit_distance_ratio ≈ 1 - ratio
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return 1.0 - sm.ratio()
