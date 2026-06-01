"""Re-embed all MongoDB chunks with BAAI/bge-m3 (1024d).

Run this once to migrate existing 768d nomic embeddings to 1024d BGE-M3
so the cloud serving package can use vector search correctly.

Usage:
    MONGODB_URI="mongodb+srv://..." uv run python scripts/reembed_bge_m3.py

Optional env vars:
    BATCH_SIZE   chunks per embedding batch (default: 16)
    DRY_RUN      set to "1" to print counts without writing
"""
from __future__ import annotations

import os
import sys
import time

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
MONGODB_URI = os.getenv("MONGODB_URI", "")
DB_NAME = os.getenv("DB_NAME", "legalro")

if not MONGODB_URI:
    sys.exit("ERROR: MONGODB_URI env var is required")

print(f"Connecting to MongoDB... (db={DB_NAME})")
from pymongo import MongoClient, UpdateOne

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

total = db.chunks.count_documents({})
print(f"Total chunks: {total}")

if DRY_RUN:
    print("DRY_RUN=1 — no writes will be performed")

print("Loading BAAI/bge-m3 model...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
model.max_seq_length = 8192

cursor = db.chunks.find({}, {"_id": 1, "text": 1}, batch_size=BATCH_SIZE)
batch_ids: list = []
batch_texts: list[str] = []
updated = 0
errors = 0
t0 = time.time()

def flush(ids, texts):
    global updated, errors
    if not ids:
        return
    try:
        embeddings = model.encode(
            [t[:30000] for t in texts],
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        if not DRY_RUN:
            ops = [
                UpdateOne({"_id": _id}, {"$set": {"embedding": emb}})
                for _id, emb in zip(ids, embeddings)
            ]
            db.chunks.bulk_write(ops, ordered=False)
        updated += len(ids)
        elapsed = time.time() - t0
        rate = updated / elapsed if elapsed > 0 else 0
        eta = (total - updated) / rate if rate > 0 else 0
        print(
            f"  {updated}/{total} chunks  "
            f"({rate:.1f}/s  ETA {eta:.0f}s)",
            end="\r",
            flush=True,
        )
    except Exception as exc:
        errors += len(ids)
        print(f"\nERROR on batch: {exc}", flush=True)

for doc in cursor:
    batch_ids.append(doc["_id"])
    batch_texts.append(doc.get("text") or "")
    if len(batch_ids) >= BATCH_SIZE:
        flush(batch_ids, batch_texts)
        batch_ids, batch_texts = [], []

flush(batch_ids, batch_texts)

print(f"\nDone. Updated={updated}  Errors={errors}  Time={time.time()-t0:.1f}s")
if DRY_RUN:
    print("DRY_RUN — no changes written to MongoDB")
