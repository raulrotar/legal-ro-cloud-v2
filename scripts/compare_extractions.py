"""
Compare extraction quality across three pipelines:
  A) extracted/          — regex baseline (current production)
  B) extracted_llm/      — Option B (LLM metadata only, full_text verbatim)
  C) extracted_option_c/ — Option C (Docling→MD→LLM→JSON)

Accuracy scoring per act (4 metadata fields, 1 point each):
  1. doc_type     known (not UNKNOWN)
  2. issuing_authority present
  3. act_number   present and not "0"
  4. title        present and not a dot-leader TOC fragment (no "......")

Additionally for Option C:
  5. full_text mojibake ratio (lower = better OCR correction)
     counts chars in {',ã,ş,ţ,ą,ę} that indicate broken Romanian encoding

Usage:
    uv run python scripts/compare_extractions.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
BASELINE   = Path("extracted")
OPTION_B   = Path("extracted_llm")
OPTION_C   = Path("extracted_option_c")

# ── Mojibake signatures for broken Romanian encoding ──────────────────────────
# Characters that indicate broken diacritics in BROKEN_2007 / BROKEN_2002 eras
_MOJIBAKE_RE = re.compile(r"[',ãşţąę\x82\x92\x93\x94]")

# ── Helpers ───────────────────────────────────────────────────────────────────

def score_act(act: dict) -> tuple[int, list[str]]:
    """Return (score 0-4, list of failed fields)."""
    dt     = act.get("doc_type", "")
    auth   = act.get("issuing_authority", "")
    num    = str(act.get("act_number", ""))
    title  = act.get("title", "")

    issues = []
    if dt == "UNKNOWN" or not dt:
        issues.append("doc_type=UNKNOWN")
    if not auth:
        issues.append("no_authority")
    if not num or num in ("0", ""):
        issues.append("no_number")
    if not title or "......" in title or len(title) <= 5:
        issues.append("bad_title")

    return 4 - len(issues), issues


def mojibake_ratio(text: str) -> float:
    if not text:
        return 0.0
    hits = len(_MOJIBAKE_RE.findall(text))
    return hits / len(text)


def load_json_tree(root: Path) -> dict[str, dict]:
    """Return {filename: gazette_dict} for all JSONs under root."""
    result = {}
    for p in root.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            result[d.get("filename", p.name)] = d
        except Exception:
            pass
    return result


def era_of(gazette: dict) -> str:
    return gazette.get("era", "unknown")


# ── Main comparison ───────────────────────────────────────────────────────────

def compare():
    base  = load_json_tree(BASELINE)
    opt_b = load_json_tree(OPTION_B)
    opt_c = load_json_tree(OPTION_C)

    common = sorted(set(base) & set(opt_c))
    print(f"Gazettes in baseline: {len(base)}")
    print(f"Gazettes in Option C: {len(opt_c)}")
    print(f"Common (both present): {len(common)}\n")

    # Per-era accumulators
    stats: dict[str, dict] = defaultdict(lambda: {
        "n_acts": 0, "n_acts_c": 0,
        "score_base": 0, "score_b": 0, "score_c": 0,
        "perfect_base": 0, "perfect_b": 0, "perfect_c": 0,
        "mojibake_base": 0.0, "mojibake_c": 0.0, "n_moji": 0,
        "seg_match": 0, "seg_total": 0,
    })

    rows = []

    for fname in common:
        g_base = base[fname]
        g_c    = opt_c[fname]
        g_b    = opt_b.get(fname)

        era = era_of(g_base)
        acts_base = g_base.get("acts", [])
        acts_c    = g_c.get("acts", [])
        acts_b    = g_b.get("acts", []) if g_b else []

        s = stats[era]
        s["n_acts"]   += len(acts_base)
        s["n_acts_c"] += len(acts_c)
        s["seg_total"] += 1
        if len(acts_base) == len(acts_c):
            s["seg_match"] += 1

        for act in acts_base:
            sc, _ = score_act(act)
            s["score_base"] += sc
            if sc == 4:
                s["perfect_base"] += 1

        for act in acts_b:
            sc, _ = score_act(act)
            s["score_b"] += sc
            if sc == 4:
                s["perfect_b"] += 1

        for act in acts_c:
            sc, _ = score_act(act)
            s["score_c"] += sc
            if sc == 4:
                s["perfect_c"] += 1
            mj = mojibake_ratio(act.get("full_text", ""))
            s["mojibake_c"] += mj
            s["n_moji"] += 1

        for act in acts_base:
            mj = mojibake_ratio(act.get("full_text", ""))
            s["mojibake_base"] += mj

        # Per-gazette row
        sc_base = sum(score_act(a)[0] for a in acts_base)
        sc_c    = sum(score_act(a)[0] for a in acts_c)
        max_base = len(acts_base) * 4
        max_c    = len(acts_c) * 4
        rows.append({
            "file": fname,
            "era": era,
            "acts_base": len(acts_base),
            "acts_c": len(acts_c),
            "acc_base": sc_base / max_base if max_base else 0,
            "acc_c":    sc_c    / max_c    if max_c    else 0,
        })

    # ── Per-gazette table ──────────────────────────────────────────────────────
    print("=" * 105)
    print(f"{'Gazette':<35} {'Era':<12} {'Acts base/C':<12} {'Regex %':<9} {'Opt-C %':<9} {'Δ':>6}")
    print("=" * 105)
    for r in rows:
        delta = (r["acc_c"] - r["acc_base"]) * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {r['file']:<33} {r['era']:<12} {r['acts_base']:>3}/{r['acts_c']:<3}      "
              f"  {r['acc_base']*100:>6.1f}%  {r['acc_c']*100:>6.1f}%  {sign}{delta:>5.1f}pp")

    # ── Era summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"{'ERA':<14} {'Acts':<8} {'Regex':>7} {'Opt-B':>7} {'Opt-C':>7} {'Δ(C-regex)':>11} {'Seg match':>10}")
    print("=" * 85)

    totals = defaultdict(int)
    totals["n_base"] = totals["n_c"] = totals["n_b"] = 0

    for era in ["scanned", "broken_2007", "modern", "unknown"]:
        s = stats.get(era)
        if not s or s["n_acts"] == 0:
            continue

        n    = s["n_acts"]
        n_c  = s["n_acts_c"]
        pct_base = s["score_base"] / (n * 4) * 100
        pct_c    = s["score_c"]    / (n_c * 4) * 100 if n_c else 0
        delta    = pct_c - pct_base
        seg      = f"{s['seg_match']}/{s['seg_total']}"
        mj_base  = s["mojibake_base"] / n * 100
        mj_c     = s["mojibake_c"] / s["n_moji"] * 100 if s["n_moji"] else 0

        totals["n_base"]  += n
        totals["n_c"]     += n_c
        totals["sb"]      += s["score_base"]
        totals["sc"]      += s["score_c"]
        totals["sb_b"]    += s["score_b"]
        totals["pb"]      += s["perfect_base"]
        totals["pc"]      += s["perfect_c"]
        totals["sm"]      += s["seg_match"]
        totals["st"]      += s["seg_total"]
        totals["mj_base"] += s["mojibake_base"]
        totals["mj_c"]    += s["mojibake_c"]
        totals["n_moji"]  += s["n_moji"]

        sign = "+" if delta >= 0 else ""
        print(f"  {era:<12}  {n:>4} acts  {pct_base:>6.1f}%  {'N/A':>6}  {pct_c:>6.1f}%  "
              f"  {sign}{delta:>7.1f}pp  {seg:>8}")
        print(f"              mojibake: regex={mj_base:.3f}%  opt-c={mj_c:.3f}%")

    # ── Grand total ─────────────────────────────────────────────────────────────
    n     = totals["n_base"]
    n_c   = totals["n_c"]
    sb    = totals["sb"]
    sc    = totals["sc"]
    pb    = totals["pb"]
    pc    = totals["pc"]
    pct_base = sb / (n * 4) * 100   if n   else 0
    pct_c    = sc / (n_c * 4) * 100 if n_c else 0
    delta    = pct_c - pct_base
    seg      = f"{totals['sm']}/{totals['st']}"
    mj_base  = totals["mj_base"] / n * 100   if n   else 0
    mj_c     = totals["mj_c"] / totals["n_moji"] * 100 if totals["n_moji"] else 0
    sign = "+" if delta >= 0 else ""

    print("=" * 85)
    print(f"  {'TOTAL':<12}  {n:>4} acts  {pct_base:>6.1f}%   N/A  {pct_c:>6.1f}%  "
          f"  {sign}{delta:>7.1f}pp  {seg:>8}")
    print(f"  Perfect acts (4/4):  regex={pb}/{n} ({100*pb/n:.1f}%)   "
          f"opt-c={pc}/{n_c} ({100*pc/n_c:.1f}%)")
    print(f"  Mojibake chars:      regex={mj_base:.4f}%   opt-c={mj_c:.4f}%")
    print(f"  Segmentation match:  {seg} gazettes produced same act count as regex baseline")

    # ── Field-level breakdown ──────────────────────────────────────────────────
    print("\n── Field breakdown (all eras combined) ──")
    def field_stats(tree: dict[str, dict], fnames: list[str]):
        unk = miss_auth = miss_num = bad_title = total = 0
        for fn in fnames:
            if fn not in tree:
                continue
            for a in tree[fn].get("acts", []):
                total += 1
                if a.get("doc_type", "") in ("UNKNOWN", ""):
                    unk += 1
                if not a.get("issuing_authority", ""):
                    miss_auth += 1
                if not str(a.get("act_number", "")) or str(a.get("act_number", "")) == "0":
                    miss_num += 1
                t = a.get("title", "")
                if not t or "......" in t or len(t) <= 5:
                    bad_title += 1
        return total, unk, miss_auth, miss_num, bad_title

    t_b, unk_b, auth_b, num_b, tit_b = field_stats(base,  common)
    t_c, unk_c, auth_c, num_c, tit_c = field_stats(opt_c, common)

    print(f"{'Field':<22} {'Regex':>10} {'Opt-C':>10} {'Δ':>8}")
    print(f"  {'doc_type=UNKNOWN':<20} {unk_b:>4}/{t_b} {100*unk_b/t_b:>5.1f}%  "
          f"{unk_c:>4}/{t_c} {100*unk_c/t_c:>5.1f}%  "
          f"{(100*unk_c/t_c - 100*unk_b/t_b):>+7.1f}pp")
    print(f"  {'missing authority':<20} {auth_b:>4}/{t_b} {100*auth_b/t_b:>5.1f}%  "
          f"{auth_c:>4}/{t_c} {100*auth_c/t_c:>5.1f}%  "
          f"{(100*auth_c/t_c - 100*auth_b/t_b):>+7.1f}pp")
    print(f"  {'missing act_number':<20} {num_b:>4}/{t_b} {100*num_b/t_b:>5.1f}%  "
          f"{num_c:>4}/{t_c} {100*num_c/t_c:>5.1f}%  "
          f"{(100*num_c/t_c - 100*num_b/t_b):>+7.1f}pp")
    print(f"  {'bad title':<20} {tit_b:>4}/{t_b} {100*tit_b/t_b:>5.1f}%  "
          f"{tit_c:>4}/{t_c} {100*tit_c/t_c:>5.1f}%  "
          f"{(100*tit_c/t_c - 100*tit_b/t_b):>+7.1f}pp")


if __name__ == "__main__":
    compare()
