"""Head-to-head benchmark: NuExtract3 vs Qwen3-14B vs Gemini baseline.

Compares extracted JSON dirs on the same small PDF set.
Also audits _via field to detect silent regex fallbacks.

Usage:
    uv run python scripts/benchmark_local_llm.py \
        --gemini   extracted_option_c/ \
        --nuextract extracted_nuextract/ \
        --qwen14b  extracted_qwen14b/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


def score_act(a: dict) -> tuple[int, list[str]]:
    issues = []
    if a.get("doc_type", "") in ("UNKNOWN", "", None):
        issues.append("doc_type=UNKNOWN")
    if not a.get("issuing_authority", ""):
        issues.append("no_authority")
    if not str(a.get("act_number", "")) or str(a.get("act_number", "")) == "0":
        issues.append("no_number")
    t = a.get("title", "")
    if not t or "......" in t or len(t) <= 5:
        issues.append("bad_title")
    return 4 - len(issues), issues


def via_label(a: dict) -> str:
    """Classify how the act was actually extracted.

    _via is stored in extraction_warnings as '_via:<value>' (pipeline.py)
    or directly as a.get('_via') for older outputs.
    """
    warns = a.get("extraction_warnings", [])
    warns_str = " ".join(warns)

    # New format: first warning is '_via:llm_option_c+...'
    for w in warns:
        if w.startswith("_via:llm_option_c"):
            return "LLM"
        if w.startswith("_via:regex_fallback"):
            return "REGEX_FALLBACK"
        if w.startswith("_via:"):
            return w[5:]  # whatever the label is

    # Old format / direct field
    via = a.get("_via", "")
    if "llm_option_c" in via:
        return "LLM"
    if "regex_fallback" in warns_str or "llm_failed" in warns_str:
        return "REGEX_FALLBACK"
    if "Extra data" in warns_str or "LLM structuring failed" in warns_str:
        return "REGEX_FALLBACK"
    return "?" if not via else via


def load_dir(root: Path) -> dict[str, dict]:
    result = {}
    if not root.exists():
        return result
    for p in root.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            result[d.get("filename", p.name)] = d
        except Exception:
            pass
    return result


def report_dir(label: str, tree: dict[str, dict], common: list[str]) -> None:
    if not tree:
        print(f"  {label}: (no data)")
        return

    total_acts = total_score = perfect = llm_count = fallback_count = unknown_count = 0
    field_fails: dict[str, int] = defaultdict(int)

    for fn in common:
        if fn not in tree:
            continue
        for a in tree[fn].get("acts", []):
            sc, issues = score_act(a)
            total_acts += 1
            total_score += sc
            if sc == 4:
                perfect += 1
            for iss in issues:
                field_fails[iss] += 1
            via = via_label(a)
            if via == "LLM":
                llm_count += 1
            elif via == "REGEX_FALLBACK":
                fallback_count += 1
            else:
                unknown_count += 1

    pct = 100 * total_score / (total_acts * 4) if total_acts else 0
    perf_pct = 100 * perfect / total_acts if total_acts else 0
    fallback_pct = 100 * fallback_count / total_acts if total_acts else 0

    # Warn loudly if most acts are silent fallbacks
    fallback_warn = " ⚠️  MOSTLY REGEX FALLBACKS" if fallback_pct > 50 else ""
    print(f"  {label:<22} {total_acts:>3} acts  acc={pct:5.1f}%  perfect={perf_pct:5.1f}%  "
          f"LLM={llm_count}/{total_acts}  fallback={fallback_count}/{total_acts}{fallback_warn}")
    for field, n in sorted(field_fails.items(), key=lambda x: -x[1]):
        print(f"      {field:<20} failed: {n}/{total_acts} ({100*n/total_acts:.1f}%)")


def per_act_diff(dirs: dict[str, tuple[str, dict[str, dict]]], common: list[str]) -> None:
    """Show per-act comparison for every act where results differ."""
    print("\n── Per-act diff (acts where any pipeline differs) ──")

    # Collect all filenames present in at least one dir
    all_files = set(fn for _, tree in dirs.values() for fn in tree)
    shown = 0

    for fn in sorted(all_files & set(common)):
        rows = {}
        for label, (_, tree) in dirs.items():
            if fn in tree:
                rows[label] = tree[fn].get("acts", [])

        max_len = max(len(v) for v in rows.values()) if rows else 0
        file_header_printed = False

        for i in range(max_len):
            line_parts = {}
            for label, acts in rows.items():
                if i < len(acts):
                    a = acts[i]
                    sc, issues = score_act(a)
                    via = via_label(a)
                    line_parts[label] = (a["doc_type"], a.get("act_number", "?"), sc, via)
                else:
                    line_parts[label] = ("—", "—", 0, "—")

            # Only print if any model differs in doc_type or act_number
            types = {v[0] for v in line_parts.values() if v[0] != "—"}
            nums  = {v[1] for v in line_parts.values() if v[1] != "—"}
            vias  = {v[3] for v in line_parts.values()}
            has_diff = len(types) > 1 or len(nums) > 1 or "REGEX_FALLBACK" in vias

            if has_diff:
                if not file_header_printed:
                    print(f"\n  {fn}")
                    file_header_printed = True
                    shown += 1
                parts_str = "  |  ".join(
                    f"{lbl}: {p[0]}/{p[1]} [{p[2]}/4] via={p[3]}"
                    for lbl, p in line_parts.items()
                )
                print(f"    act[{i}]  {parts_str}")

    if shown == 0:
        print("  (all pipelines agree on every act)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini",    default="extracted_option_c/",  help="Gemini Option C dir")
    ap.add_argument("--nuextract", default="extracted_nuextract/",  help="NuExtract3 dir")
    ap.add_argument("--qwen14b",   default="extracted_qwen14b/",   help="Qwen3-14B dir")
    ap.add_argument("--baseline",  default="extracted/",           help="Regex baseline dir")
    args = ap.parse_args()

    dirs = {
        "Gemini":      (args.gemini,    load_dir(Path(args.gemini))),
        "NuExtract3":  (args.nuextract, load_dir(Path(args.nuextract))),
        "Qwen3-14B":   (args.qwen14b,   load_dir(Path(args.qwen14b))),
        "Regex":       (args.baseline,  load_dir(Path(args.baseline))),
    }

    # Only score files present in Gemini (ground truth reference)
    gemini_files = set(dirs["Gemini"][1].keys())
    common = sorted(gemini_files)

    print(f"Files in Gemini baseline: {len(gemini_files)}")
    for label, (path, tree) in dirs.items():
        present = len(set(tree) & gemini_files)
        print(f"  {label:<12}: {len(tree)} total, {present} matching Gemini")

    print("\n── Accuracy summary ──")
    for label, (_, tree) in dirs.items():
        report_dir(label, tree, common)

    # Per-act diff (only if at least 2 non-empty non-baseline dirs)
    llm_dirs = {k: v for k, v in dirs.items() if k != "Regex" and v[1]}
    if len(llm_dirs) >= 2:
        per_act_diff(llm_dirs, common)


if __name__ == "__main__":
    main()
