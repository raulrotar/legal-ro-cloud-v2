"""Batch extraction-quality audit: md_cache + extracted JSONs vs source PDFs.

Per gazette, reports:
  - MD volume and OCR verification gate results (scanned era sidecars)
  - residual intra-word mojibake rate (broken-era regression check)
  - acts vs sumar reconciliation (missing acts / phantom candidates)
  - generic-title count (title backfill regression check)
  - recovered-acts and gate warnings from extraction_warnings

Usage:
    uv run python tools/ops/audit_extraction.py [--json-dir db/extracted] [--out reports/extraction_audit.md]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "processing" / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "core" / "src"))

ROOT = Path(__file__).parents[2]

_MOJIBAKE_RE = re.compile(r"[a-zA-ZăâîșțĂÂÎȘȚ][',][a-zA-ZăâîșțĂÂÎȘȚ]")
_GENERIC_TITLES = {"", "DECRET", "DECRET-LEGE", "LEGE", "HOTĂRÂRE", "ORDIN",
                   "DECIZIE", "COMUNICAT", "UNKNOWN"}


def audit_one(json_path: Path, md_cache_dir: Path) -> dict:
    d = json.loads(json_path.read_text(encoding="utf-8"))
    gid = d.get("gazette_id", json_path.stem)
    date = d.get("issue_date", "")
    y, m, day = (date.split("-") + ["", "", ""])[:3]
    md_path = md_cache_dir / y / m / day / f"{json_path.stem}.md"
    md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    verify_path = md_path.with_suffix(".verify.json")
    pages_failed = 0
    pages_total = 0
    if verify_path.exists():
        vs = json.loads(verify_path.read_text(encoding="utf-8"))
        pages_total = len(vs)
        pages_failed = sum(1 for v in vs if not v.get("passed"))

    words = len(md.split()) or 1
    moji = len(_MOJIBAKE_RE.findall(md))
    acts = d.get("acts", [])
    sumar = d.get("sumar", [])
    warnings = d.get("extraction_warnings", [])
    generic = sum(
        1 for a in acts
        if (a.get("title") or "").strip().upper() in _GENERIC_TITLES
    )
    missing = sum(1 for w in warnings if "MISSING act" in w)
    phantom = sum(1 for w in warnings if "phantom candidate" in w)
    recovered = sum(
        int(mm.group(1)) for w in warnings
        for mm in [re.search(r"md_act_recovery: (\d+) act", w)] if mm
    )
    return {
        "gazette": gid,
        "era": d.get("era", "?"),
        "md_chars": len(md),
        "ocr_pages": f"{pages_total - pages_failed}/{pages_total}" if pages_total else "—",
        "mojibake_pct": round(moji / words * 100, 2),
        "sumar": len(sumar),
        "acts": len(acts),
        "missing": missing,
        "phantom": phantom,
        "recovered": recovered,
        "generic_titles": generic,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", default=str(ROOT / "db" / "extracted"))
    p.add_argument("--md-cache", default=str(ROOT / "db" / "md_cache"))
    p.add_argument("--out", default=str(ROOT / "reports" / "extraction_audit.md"))
    args = p.parse_args()

    rows = [audit_one(j, Path(args.md_cache))
            for j in sorted(Path(args.json_dir).rglob("*.json"))]

    cols = ["gazette", "era", "md_chars", "ocr_pages", "mojibake_pct",
            "sumar", "acts", "missing", "phantom", "recovered", "generic_titles"]
    header = f"{'gazette':22}{'era':13}{'md_chars':>9}{'ocr':>7}{'moji%':>7}" \
             f"{'sumar':>6}{'acts':>6}{'miss':>5}{'phan':>5}{'recov':>6}{'gen.title':>10}"
    print(header)
    for r in rows:
        print(f"{r['gazette']:22}{r['era']:13}{r['md_chars']:>9}{r['ocr_pages']:>7}"
              f"{r['mojibake_pct']:>7}{r['sumar']:>6}{r['acts']:>6}{r['missing']:>5}"
              f"{r['phantom']:>5}{r['recovered']:>6}{r['generic_titles']:>10}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Extraction audit\n", "| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
