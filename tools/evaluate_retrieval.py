#!/usr/bin/env python3
"""Evaluate retrieval quality against a golden set.

Metrics: Hit@3, Hit@10, MRR@10

Usage:
    uv run python scripts/evaluate_retrieval.py
    uv run python scripts/evaluate_retrieval.py --golden tests/golden_set.json
    LEGALRO_ENV=staging uv run python scripts/evaluate_retrieval.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legalro_core.config import load_settings
from legalro_core.retrieval.search import hybrid_search


def evaluate(golden: list[dict], settings, k_values: list[int] = [3, 10]) -> dict:
    hits = {k: 0 for k in k_values}
    reciprocal_ranks = []
    failures = []

    for i, item in enumerate(golden):
        question = item["question"]
        source_chunk_id = item["source_chunk_id"]
        print(f"  [{i + 1}/{len(golden)}] {question[:80]}...")

        try:
            results = hybrid_search(question, settings)
            result_ids = [str(r["_id"]) for r in results]

            # Hit@K
            for k in k_values:
                if source_chunk_id in result_ids[:k]:
                    hits[k] += 1

            # MRR@10
            if source_chunk_id in result_ids[:10]:
                rank = result_ids.index(source_chunk_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
                failures.append({
                    "question": question,
                    "expected": source_chunk_id,
                    "law_id": item.get("law_id"),
                    "top_3_retrieved": result_ids[:3],
                })

        except Exception as e:
            print(f"    ERROR: {e}")
            reciprocal_ranks.append(0.0)

    n = len(golden)
    metrics = {f"Hit@{k}": round(hits[k] / n, 4) for k in k_values}
    metrics["MRR@10"] = round(sum(reciprocal_ranks) / n, 4)
    metrics["n"] = n
    metrics["failures"] = failures
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="tests/golden_set.json")
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"Golden set not found at {golden_path}. Run generate_golden_set.py first.")
        sys.exit(1)

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    import os
    settings = load_settings()
    env = os.getenv("LEGALRO_ENV", "cloud")
    print(f"Evaluating retrieval — env={env}, model={settings.embeddings.model}, n={len(golden)}\n")

    metrics = evaluate(golden, settings)

    print(f"\n{'=' * 40}")
    print(f"  Hit@3:  {metrics['Hit@3']:.1%}")
    print(f"  Hit@10: {metrics['Hit@10']:.1%}")
    print(f"  MRR@10: {metrics['MRR@10']:.4f}")
    print(f"  n={metrics['n']}")
    print(f"{'=' * 40}")

    if metrics["failures"]:
        print(f"\nFailed to retrieve ({len(metrics['failures'])} queries):")
        for f in metrics["failures"]:
            print(f"  - [{f['law_id']}] {f['question'][:70]}")
            print(f"    top-3 retrieved: {f['top_3_retrieved']}")

    # Save results
    out = {
        "env": env,
        "model": settings.embeddings.model,
        "dimensions": settings.embeddings.dimensions,
        **{k: v for k, v in metrics.items() if k != "failures"},
        "failure_count": len(metrics["failures"]),
    }
    results_path = Path(f"tests/eval_results_{env}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
