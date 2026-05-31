#!/usr/bin/env python3
"""Re-embed all chunks using the configured model and update the vector search index.

Reconstructs text_embedded from metadata for chunks that predate the contextual
prepending change, then embeds from text_embedded (not raw text).
"""
import sys
from pathlib import Path
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legalro_core.config import load_settings
from legalro_core.store import get_db
from legalro_core.embeddings import embed_texts


def _build_text_embedded(chunk: dict) -> str:
    doc_type = chunk.get("document_type") or "ACT"
    act_number = chunk.get("act_number")
    act_year = chunk.get("act_year")
    issuing_authority = chunk.get("issuing_authority")
    source_issue_id = chunk.get("source_issue_id")

    label = doc_type
    if act_number and act_year:
        label += f" {act_number}/{act_year}"
    elif act_number:
        label += f" {act_number}"

    parts = [label]
    if issuing_authority:
        parts.append(issuing_authority)
    if source_issue_id:
        parts.append(f"MO {source_issue_id}")
    if act_year:
        parts.append(str(act_year))

    prefix = "[" + " | ".join(parts) + "]"
    return f"{prefix} {chunk['text']}"


def reembed_all():
    settings = load_settings()
    db = get_db(settings)

    total = db.chunks.count_documents({})
    print(f"Re-embedding {total} chunks with model={settings.embeddings.model} "
          f"(dims={settings.embeddings.dimensions})")

    batch_size = settings.embeddings.batch_size
    cursor = db.chunks.find({}, {
        "_id": 1, "text": 1, "text_embedded": 1,
        "document_type": 1, "act_number": 1, "act_year": 1,
        "issuing_authority": 1, "source_issue_id": 1,
    })
    chunks = list(cursor)

    ops = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts_embedded = []
        for c in batch:
            if c.get("text_embedded"):
                texts_embedded.append(c["text_embedded"])
            else:
                texts_embedded.append(_build_text_embedded(c))

        embeddings = embed_texts(texts_embedded, settings)

        for chunk, text_embedded, emb in zip(batch, texts_embedded, embeddings):
            ops.append(UpdateOne(
                {"_id": chunk["_id"]},
                {"$set": {
                    "text_embedded": text_embedded,
                    "embedding": emb,
                    "embedding_dim": len(emb),
                    "embedding_model": settings.embeddings.model,
                }}
            ))

        done = min(i + batch_size, total)
        print(f"  {done}/{total}", end="\r", flush=True)

    print(f"\nWriting {len(ops)} updates...")
    result = db.chunks.bulk_write(ops, ordered=False)
    print(f"Updated {result.modified_count} chunks.")


def update_vector_index():
    settings = load_settings()
    db = get_db(settings)
    dims = settings.embeddings.dimensions

    print(f"Updating vector search index to {dims} dimensions...")
    existing = {idx["name"]: idx for idx in db.chunks.list_search_indexes()}

    fields = [
        {"type": "vector", "path": "embedding", "numDimensions": dims, "similarity": "cosine"},
        {"type": "filter", "path": "law_id"},
        {"type": "filter", "path": "chunk_type"},
        {"type": "filter", "path": "document_type"},
    ]

    if "chunks_vector" in existing:
        db.chunks.update_search_index("chunks_vector", {"fields": fields})
        print("Updated existing 'chunks_vector' index.")
    else:
        db.chunks.create_search_index({
            "name": "chunks_vector",
            "type": "vectorSearch",
            "definition": {"fields": fields},
        })
        print("Created new 'chunks_vector' index.")


if __name__ == "__main__":
    reembed_all()
    update_vector_index()
    print("Done.")
