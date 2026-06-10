#!/usr/bin/env python3
"""
Extract one or more gazette PDFs to structured JSON.

Usage:
    uv run python scripts/extract_gazette.py laws/2017/01/30/MO_PI_76_2017-01-30.pdf
    uv run python scripts/extract_gazette.py laws/2017/      # all PDFs under a directory
    uv run python scripts/extract_gazette.py --validate-only db/extracted/2017/01/30/MO_PI_76_2017-01-30.json

Options:
    --out DIR       Output directory (default: db/extracted/)
    --validate      Run validation after extraction and print report
    --validate-only Load existing JSON and validate without re-extracting
    --strict        Exit with code 1 if any ERROR is found
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legalro_processing.extract.gazette_extractor import extract_gazette, save_gazette, load_gazette
from legalro_processing.gazette_validator import validate_gazette, format_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract gazette PDFs to structured JSON")
    parser.add_argument("paths", nargs="+", help="PDF file(s) or directories to extract")
    parser.add_argument("--out", default="db/extracted", help="Output directory (default: db/extracted/)")
    parser.add_argument("--validate", action="store_true", help="Validate after extraction")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing JSON, skip extraction")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any ERROR found")
    args = parser.parse_args()

    has_error = False

    for raw_path in args.paths:
        path = Path(raw_path)

        # Collect files
        if path.is_dir():
            files = sorted(path.rglob("*.pdf")) if not args.validate_only else sorted(path.rglob("*.json"))
        elif path.is_file():
            files = [path]
        else:
            print(f"[SKIP] {path} — not found", file=sys.stderr)
            continue

        for fpath in files:
            if args.validate_only:
                print(f"Loading  {fpath}")
                gazette = load_gazette(fpath)
                issues = validate_gazette(gazette)
                print(format_report(issues, gazette.gazette_id))
                if any(i.severity == "ERROR" for i in issues):
                    has_error = True
                continue

            print(f"Extracting {fpath} …", end=" ", flush=True)
            try:
                gazette = extract_gazette(fpath)
            except Exception as exc:
                print(f"FAILED: {exc}")
                has_error = True
                continue

            out_path = save_gazette(gazette, args.out)
            n_acts = len(gazette.acts)
            n_sumar = len(gazette.sumar)
            n_warn = len(gazette.extraction_warnings)
            print(f"→ {out_path}  ({n_acts} acts, {n_sumar} sumar entries, {n_warn} warnings)")

            if args.validate:
                issues = validate_gazette(gazette)
                print(format_report(issues, gazette.gazette_id))
                if any(i.severity == "ERROR" for i in issues):
                    has_error = True

    return 1 if (args.strict and has_error) else 0


if __name__ == "__main__":
    sys.exit(main())
