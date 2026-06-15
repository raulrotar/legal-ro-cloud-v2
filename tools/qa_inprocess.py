#!/usr/bin/env python3
"""In-process QA harness: retrieval (Ollama bge-m3 + Atlas) + DIRECT Gemini
generation, so the SYSTEM_PROMPT can be swapped WITHOUT redeploying the HF Space.

Why: the deployed HF Space runs the deployed serving code; to test a local
SYSTEM_PROMPT change against the (already-updated) staging data, we bypass the
Space — retrieve in-process, then call Gemini directly with whatever prompt we
pass. Embeddings stay on Ollama bge-m3 to match the int8-Binary vectors loaded
into staging.

Usage:
    set -a; . ./.env; set +a            # MONGODB_URI (Atlas) + GEMINI_API_KEY
    unset LEGALRO_API_URL               # force in-process
    uv run python tools/qa_inprocess.py [--prompt new|current] [--set qt|all]

Prints, per question: retrieved chunk types, whether <table>/colspan reached the
context, the answer, and a crude expected-token check.
"""
from __future__ import annotations
import os, sys, httpx, time, unicodedata, re


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def run(questions, system_prompt, label=""):
    from legalro_core.config import load_settings
    from legalro_core.retrieval.search import hybrid_search
    from legalro_serving.generation import assemble_context
    s = load_settings()  # ollama embeddings + Atlas mongo via MONGODB_URI
    key = os.environ["GEMINI_API_KEY"]
    print(f"\n{'='*70}\nIN-PROCESS QA — {label}\n{'='*70}")
    passed = 0
    for qid, q, exp in questions:
        t = time.time()
        res = hybrid_search(q, s)
        ctx = assemble_context(res, s)
        has_tbl = "<table" in ctx
        try:
            r = httpx.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "gemini-3.1-flash-lite",
                      "messages": [{"role": "system", "content": system_prompt},
                                   {"role": "user", "content": f"Documente relevante:\n{ctx}\n\nÎntrebare: {q}"}],
                      "max_tokens": 1024, "temperature": 0.0},
                timeout=120)
            ans = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            ans = f"ERR {type(e).__name__}: {e}"
        exp_first = _norm(exp).split(";")[0].strip()
        hit = bool(exp_first) and (exp_first in _norm(ans) or any(
            re.search(r"\b" + re.escape(w) + r"\b", _norm(ans)) for w in exp_first.split() if len(w) > 2))
        passed += hit
        print(f"\n## {qid}  ({time.time()-t:.0f}s)  table_in_ctx={has_tbl}  hit={hit}")
        print(f"   types: {[r.get('chunk_type') for r in res[:6]]}")
        print(f"   EXPECTED: {exp}")
        print(f"   ANSWER: {ans[:380]}")
    print(f"\n→ {label}: {passed}/{len(questions)} expected-token hits")
    return passed


if __name__ == "__main__":
    from legalro_serving.generation import SYSTEM_PROMPT
    from tools.test_questions_tables import QUESTIONS as QT
    which = "qt"
    if "--set" in sys.argv:
        which = sys.argv[sys.argv.index("--set") + 1]
    qs = list(QT)
    if which == "all":
        from tools.test_questions_cloud import QUESTIONS as Q51
        qs = list(Q51) + list(QT)
    run(qs, SYSTEM_PROMPT, label=f"current prompt / {which}")
