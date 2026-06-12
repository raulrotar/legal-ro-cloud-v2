"""Summary-Augmented Chunking (Phase 2 of docs/EMBEDDINGS_PLAN.md).

Prepending a short GENERIC act summary to every chunk roughly halves
wrong-act retrieval in legal corpora (arXiv 2510.06999) — the dominant
failure mode for structurally similar gazette acts.  One local-LLM call per
act (not per chunk), disk-cached by content hash so re-builds are free.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

SAC_CACHE_DIR = Path("db/sac_cache")
_MAX_INPUT_CHARS = 4000

_PROMPT = (
    "Rezumă actul normativ de mai jos în 1-2 propoziții simple în limba "
    "română: ce tip de act este, cine îl emite, despre ce este și ce "
    "entități/persoane/instituții vizează. Fără introducere, doar rezumatul.\n\n"
    "TITLU: {title}\n\nTEXT:\n{text}"
)


def act_summary(act, settings) -> str:
    """Return a 1–2 sentence generic summary for the act (cached)."""
    text = (getattr(act, "full_text", "") or "")[:_MAX_INPUT_CHARS]
    title = getattr(act, "title", "") or ""
    if len(text) < 200:
        return ""  # tiny acts: the title/metadata prefix already says it all

    key = hashlib.sha1((title + text).encode()).hexdigest()
    cache = SAC_CACHE_DIR / f"{key}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8").strip()

    # candidate models: configured one first unless it's a vision model
    # (mllama no longer loads in current Ollama), then known-good text models
    cfg_model = getattr(settings.llm, "model", "") or ""
    candidates = [m for m in (cfg_model, "gemma4:12b-nvfp4", "qwen3.5:9b")
                  if m and "vision" not in m]
    summary = ""
    import ollama
    client = ollama.Client()
    for model in candidates:
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user",
                           "content": _PROMPT.format(title=title, text=text)}],
                options={"temperature": 0, "num_predict": 300},
                think=False,  # gemma4/qwen3.5 are thinking models; without this content is empty
            )
            summary = (resp.message.content or "").strip()
            if summary:
                break
        except Exception as exc:
            print(f"[sac] {model} failed ({exc}) — trying next", file=sys.stderr)
    if not summary:
        return ""

    # sanity: strip chain-of-thought tags, newlines, hard length cap
    summary = re.sub(r"(?s)<think>.*?</think>", "", summary)
    summary = re.sub(r"\s+", " ", summary).strip()[:500]
    if len(summary.split()) < 4:
        return ""
    SAC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(summary, encoding="utf-8")
    return summary
