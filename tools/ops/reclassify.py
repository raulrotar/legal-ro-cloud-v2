#!/usr/bin/env python3
"""Backfill issuing_authority, act_number, locality, law_id into existing chunks.

Groups chunks by (source_issue_id, act_index_in_issue), reassembles act text from
chunk texts ordered by position_in_law, runs extract_metadata, and updates the DB.
This fixes Q11 and Q29 without requiring a full re-ingestion.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from pymongo import UpdateMany, UpdateOne

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legalro_core.config import load_settings
from legalro_core.store import get_db
from legalro_processing.extract.metadata import extract_metadata
from legalro_processing.extract.segment import RawAct


def main():
    settings = load_settings()
    db = get_db(settings)

    print("Grouping chunks by (source_issue_id, act_index_in_issue)...")

    # Build groups: {(source_issue_id, act_index): [chunk_doc, ...]}
    groups: dict[tuple, list[dict]] = {}
    for chunk in db.chunks.find({}, {
        "_id": 1, "source_issue_id": 1, "act_index_in_issue": 1,
        "text": 1, "position_in_law": 1, "document_type": 1,
    }):
        key = (chunk.get("source_issue_id", ""), chunk.get("act_index_in_issue", 0))
        groups.setdefault(key, []).append(chunk)

    print(f"Found {len(groups)} unique acts across {db.chunks.count_documents({})} chunks.")

    # Year lookup per source_issue_id
    def year_from_issue(issue_id: str) -> int:
        parts = issue_id.split("_")
        for p in reversed(parts):
            if p.isdigit() and len(p) == 4:
                return int(p)
        return 2024

    ops = []
    for (issue_id, act_idx), chunks in sorted(groups.items()):
        gazette_year = year_from_issue(issue_id)
        # Reassemble act text in order of position_in_law
        ordered = sorted(chunks, key=lambda c: c.get("position_in_law", 0))
        full_text = "\n\n".join(c["text"] for c in ordered)

        raw_act = RawAct(text=full_text, title="", page_range=[], position_in_gazette=act_idx)
        meta = extract_metadata(raw_act, gazette_year)

        ids = [c["_id"] for c in chunks]
        # Do NOT update law_id — there is a unique index on {law_id, position_in_law}
        # and multiple acts falling back to the same "unknown_*" law_id would collide.
        # Instead, add new disambiguation fields that context.py already reads.
        set_doc: dict = {
            "issuing_authority": meta["issuing_authority"],
            "act_number": meta["act_number"],
            "locality": meta["locality"],
        }
        if meta["doc_type"] != "UNKNOWN":
            set_doc["document_type"] = meta["doc_type"]

        ops.append(UpdateMany({"_id": {"$in": ids}}, {"$set": set_doc}))
        if not meta["issuing_authority"]:
            print(f"  WARN {issue_id}/act{act_idx}: no authority found (doc={meta['doc_type']} num={meta['act_number']})")

    print(f"Writing {len(ops)} bulk updates...")
    result = db.chunks.bulk_write(ops, ordered=False)
    print(f"Matched {result.matched_count}, modified {result.modified_count} chunks.")

    # Spot-check the failing gazettes
    print("\n=== PI_76_2017 acts after update ===")
    seen = set()
    for c in db.chunks.find({"source_issue_id": "PI_76_2017"}, {
        "act_index_in_issue": 1, "law_id": 1, "issuing_authority": 1, "act_number": 1, "locality": 1, "title": 1,
    }):
        k = (c["act_index_in_issue"], c.get("law_id"))
        if k not in seen:
            seen.add(k)
            print(f"  act{c['act_index_in_issue']}: {c.get('law_id')} | "
                  f"auth={c.get('issuing_authority','')[:30]} | loc={c.get('locality','')} | "
                  f"title={str(c.get('title',''))[:50]}")

    print("\n=== PI_820_2007 acts after update ===")
    seen2 = set()
    for c in db.chunks.find({"source_issue_id": "PI_820_2007"}, {
        "act_index_in_issue": 1, "law_id": 1, "issuing_authority": 1, "act_number": 1, "locality": 1, "title": 1,
    }):
        k = (c["act_index_in_issue"], c.get("law_id"))
        if k not in seen2:
            seen2.add(k)
            print(f"  act{c['act_index_in_issue']}: {c.get('law_id')} | "
                  f"auth={c.get('issuing_authority','')[:30]} | loc={c.get('locality','')} | "
                  f"num={c.get('act_number','')} | title={str(c.get('title',''))[:50]}")


if __name__ == "__main__":
    main()
