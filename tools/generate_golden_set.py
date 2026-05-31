#!/usr/bin/env python3
"""Generate a golden evaluation set for retrieval quality benchmarking.

Pulls random chunks from the DB, generates a question for each using Gemini,
and saves (question, source_chunk_id, metadata) pairs to tests/golden_set.json.

Usage:
    uv run python scripts/generate_golden_set.py --n 15
    LEGALRO_ENV=staging uv run python scripts/generate_golden_set.py --n 15
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legalro_core.config import load_settings
from legalro_core.store import get_db


_PROMPT_TEMPLATE = """Ești un asistent juridic specializat în dreptul românesc.

Citește fragmentul de lege de mai jos și formulează O SINGURĂ întrebare specifică
la care răspunsul se găsește exclusiv în acest fragment. Întrebarea trebuie să fie
în română, să fie naturală (cum ar pune-o un cetățean sau un jurist), și să nu
conțină cuvinte din titlul actului normativ.

Fragment:
{text}

Răspunde DOAR cu întrebarea, fără explicații suplimentare."""


def generate_question(text: str, settings) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url)
    response = client.chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(text=text[:2000])}],
        max_tokens=256,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15, help="Number of golden pairs to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="tests/golden_set.json")
    args = parser.parse_args()

    settings = load_settings()
    db = get_db(settings)

    random.seed(args.seed)

    total = db.chunks.count_documents({"tokens": {"$gte": 100}})
    if total == 0:
        print("No chunks found. Ingest some documents first.")
        sys.exit(1)

    print(f"Sampling {args.n} chunks from {total} candidates...")

    # Sample by skipping random offsets — avoids loading all IDs into memory
    sample_chunks = []
    skip_offsets = sorted(random.sample(range(total), min(args.n * 3, total)))
    seen_laws: set[str] = set()

    for offset in skip_offsets:
        if len(sample_chunks) >= args.n:
            break
        doc = db.chunks.find_one(
            {"tokens": {"$gte": 100}},
            {"_id": 1, "text": 1, "law_id": 1, "document_type": 1,
             "act_number": 1, "act_year": 1, "issuing_authority": 1,
             "source_issue_id": 1, "full_path": 1},
            skip=offset,
        )
        if doc and doc["law_id"] not in seen_laws:
            seen_laws.add(doc["law_id"])
            sample_chunks.append(doc)

    golden = []
    for i, chunk in enumerate(sample_chunks):
        print(f"  Generating question {i + 1}/{len(sample_chunks)}: {chunk['law_id']}...")
        try:
            question = generate_question(chunk["text"], settings)
            golden.append({
                "question": question,
                "source_chunk_id": str(chunk["_id"]),
                "law_id": chunk["law_id"],
                "document_type": chunk.get("document_type"),
                "act_number": chunk.get("act_number"),
                "act_year": chunk.get("act_year"),
                "issuing_authority": chunk.get("issuing_authority"),
                "source_issue_id": chunk.get("source_issue_id"),
                "full_path": chunk.get("full_path"),
                "text_preview": chunk["text"][:200],
            })
        except Exception as e:
            print(f"    WARNING: failed to generate question: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(golden)} golden pairs to {out_path}")


if __name__ == "__main__":
    main()
