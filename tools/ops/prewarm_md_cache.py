"""Pre-warm the Docling MD cache for all PDFs under laws/.

Run this BEFORE extraction when the LLM model is large (e.g. gemma4:12b-mlx)
so that Docling OCR and the LLM never run in the same process simultaneously.

After this script finishes, run the normal extraction:
    uv run legalro-process extract --root laws/ --out db/bundle_bge-m3/ --extracted-dir db/extracted/
The extraction will find all MD files cached and skip Docling entirely.

Usage:
    uv run python tools/ops/prewarm_md_cache.py [--root laws/] [--md-cache-dir db/md_cache/]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "processing" / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "core" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-warm Docling MD cache for all PDFs")
    parser.add_argument("--root", default="laws/", help="Root directory containing PDFs")
    parser.add_argument("--md-cache-dir", default="db/md_cache/", help="MD cache directory")
    parser.add_argument("--config", default="config/local.yaml", help="Config file")
    args = parser.parse_args()

    from legalro_core.config import load_settings
    from legalro_processing.extract import md_cache, md_extractor
    from legalro_processing.extract.era import detect_era

    settings = load_settings(args.config)
    pdfs = sorted(Path(args.root).rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs under {args.root}")

    done = skipped = failed = 0
    for i, pdf in enumerate(pdfs, 1):
        # Check if already cached
        if md_cache.load(pdf, args.md_cache_dir) is not None:
            print(f"[{i:2}/{len(pdfs)}] SKIP (cached)  {pdf.name}")
            skipped += 1
            continue

        era = detect_era(pdf)
        print(f"[{i:2}/{len(pdfs)}] Docling → {pdf.name}  era={era.value}", flush=True)
        t0 = time.time()
        try:
            md = md_extractor.extract_markdown(str(pdf), era, settings)
            md_cache.save(pdf, md, era.value, args.md_cache_dir)
            elapsed = time.time() - t0
            print(f"         done {len(md):,} chars  +{elapsed:.1f}s", flush=True)
            done += 1
        except Exception as exc:
            print(f"         FAILED: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {done} extracted, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
