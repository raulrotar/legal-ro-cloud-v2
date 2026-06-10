"""
Compare GLM-OCR vs Docling on the 1989 scanned-era PDFs.

For each PDF:
  1. Render each page to PNG at 200 DPI
  2. Send to glm-ocr:latest via Ollama vision API
  3. Concatenate page texts → glm_ocr output
  4. Load existing Docling MD from md_cache
  5. Print side-by-side stats and save both to /tmp/glm_compare/
"""

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import fitz  # pymupdf
import ollama


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[bytes]:
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        images.append(pix.tobytes("png"))
    return images


def glm_ocr_page(png_bytes: bytes, client: ollama.Client) -> str:
    b64 = base64.b64encode(png_bytes).decode()
    resp = client.chat(
        model="glm-ocr:latest",
        messages=[{
            "role": "user",
            "content": "Please OCR this image and output the text exactly as it appears.",
            "images": [b64],
        }],
        options={"temperature": 0},
    )
    return resp.message.content or ""


def char_overlap(a: str, b: str) -> float:
    """Simple character-level Jaccard on trigrams."""
    def trigrams(s):
        s = re.sub(r'\s+', ' ', s).strip()
        return set(s[i:i+3] for i in range(len(s) - 2))
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_docling_md(pdf_path: Path, md_cache_dir: Path) -> str | None:
    stem = pdf_path.stem  # e.g. MO_PI_1_1989-12-22
    # cache layout: md_cache/YYYY/MM/DD/<stem>.md
    parts = pdf_path.parts
    # find year/month/day from path
    matches = list(md_cache_dir.rglob(f"{stem}.md"))
    if matches:
        return matches[0].read_text(encoding="utf-8")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="laws/1989", help="Root dir of scanned PDFs")
    ap.add_argument("--md-cache-dir", default="db/md_cache")
    ap.add_argument("--out-dir", default="/tmp/glm_compare")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pdf", help="Single PDF to process (optional)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_cache_dir = Path(args.md_cache_dir)

    if args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        pdfs = sorted(Path(args.root).rglob("*.pdf"))

    client = ollama.Client()
    results = []

    for pdf_path in pdfs:
        name = pdf_path.stem
        print(f"\n{'='*60}")
        print(f"PDF: {name}  ({pdf_path.stat().st_size // 1024} KB)")

        # --- GLM-OCR ---
        t0 = time.time()
        images = pdf_to_images(pdf_path, dpi=args.dpi)
        print(f"  Rendered {len(images)} page(s) at {args.dpi} DPI")
        glm_pages = []
        for i, img in enumerate(images):
            print(f"  GLM-OCR page {i+1}/{len(images)}...", end=" ", flush=True)
            pt = time.time()
            text = glm_ocr_page(img, client)
            print(f"{len(text)} chars in {time.time()-pt:.1f}s")
            glm_pages.append(text)
        glm_text = "\n\n---PAGE---\n\n".join(glm_pages)
        glm_elapsed = time.time() - t0

        glm_out = out_dir / f"{name}_glm.txt"
        glm_out.write_text(glm_text, encoding="utf-8")

        # --- Docling MD ---
        docling_md = load_docling_md(pdf_path, md_cache_dir)
        if docling_md:
            docling_out = out_dir / f"{name}_docling.md"
            docling_out.write_text(docling_md, encoding="utf-8")
            similarity = char_overlap(glm_text, docling_md)
            docling_chars = len(docling_md)
            print(f"  Docling MD: {docling_chars} chars (cached)")
        else:
            similarity = None
            docling_chars = 0
            print(f"  Docling MD: NOT FOUND in cache")

        glm_chars = len(glm_text)
        print(f"  GLM-OCR:    {glm_chars} chars in {glm_elapsed:.1f}s")
        if similarity is not None:
            print(f"  Trigram similarity (GLM vs Docling): {similarity:.3f}")

        results.append({
            "pdf": name,
            "pages": len(images),
            "glm_chars": glm_chars,
            "glm_elapsed_s": round(glm_elapsed, 1),
            "docling_chars": docling_chars,
            "trigram_sim": round(similarity, 3) if similarity is not None else None,
        })

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'PDF':<30} {'pages':>5} {'GLM chars':>10} {'Docling chars':>13} {'sim':>6} {'time(s)':>8}")
    for r in results:
        sim = f"{r['trigram_sim']:.3f}" if r["trigram_sim"] is not None else "  N/A"
        print(f"{r['pdf']:<30} {r['pages']:>5} {r['glm_chars']:>10} {r['docling_chars']:>13} {sim:>6} {r['glm_elapsed_s']:>8.1f}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()
