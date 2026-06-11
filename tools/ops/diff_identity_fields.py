#!/usr/bin/env python3
"""Regression diff: compare identity fields (doc_type/act_number/act_year/title)
between two extraction runs of the same issues.

Usage: python tools/ops/diff_identity_fields.py <old_dir> <new_root>
  old_dir  — flat dir of pre-change JSONs (e.g. /tmp/pre_regression)
  new_root — extracted tree root (e.g. db/extracted)

Body text is intentionally NOT compared — the fixes only touch identity
fields; any body change would itself be a regression worth seeing, but the
extractor is deterministic from cached MD, so identity fields are the signal.
"""
import json
import sys
from pathlib import Path


def act_key(a: dict) -> tuple:
    return (a.get("doc_type"), str(a.get("act_number")), a.get("act_year"),
            (a.get("title") or "")[:80])


def main(old_dir: str, new_root: str) -> int:
    old_files = sorted(Path(old_dir).glob("*.json"))
    n_changed_files = 0
    for old_path in old_files:
        hits = list(Path(new_root).rglob(old_path.name))
        if not hits:
            print(f"!! {old_path.name}: missing from new run")
            n_changed_files += 1
            continue
        old = json.loads(old_path.read_text())
        new = json.loads(hits[0].read_text())
        old_acts = [act_key(a) for a in old.get("acts", [])]
        new_acts = [act_key(a) for a in new.get("acts", [])]
        if old_acts == new_acts:
            print(f"OK {old_path.name}: {len(new_acts)} acts identical")
            continue
        n_changed_files += 1
        print(f"\n== {old_path.name}: {len(old_acts)} -> {len(new_acts)} acts ==")
        removed = [k for k in old_acts if k not in new_acts]
        added = [k for k in new_acts if k not in old_acts]
        for k in removed:
            print(f"  - {k}")
        for k in added:
            print(f"  + {k}")
    print(f"\n{n_changed_files}/{len(old_files)} files changed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
