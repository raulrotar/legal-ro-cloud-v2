"""Create/update MongoDB indexes for LegalRo.

Run standalone:  uv run python scripts/setup_indexes.py
Also called by reingest.py after every full reingest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def ensure_indexes(db, settings) -> None:
    """Idempotent: create or update all B-tree and Atlas Search indexes."""
    dims = settings.embeddings.dimensions

    # ── B-tree indexes ────────────────────────────────────────────────
    db.gazettes.create_index("filename", unique=True, background=True)
    db.gazettes.create_index("year", background=True)
    db.gazettes.create_index("sha256", background=True)

    db.chunks.create_index([("source_issue_id", 1), ("act_index_in_issue", 1)], background=True)
    db.chunks.create_index("law_id", background=True)
    db.chunks.create_index("document_type", background=True)

    # ── Atlas Search indexes ──────────────────────────────────────────
    existing = {idx["name"]: idx for idx in db.chunks.list_search_indexes()}

    vector_fields = [
        {"type": "vector", "path": "embedding", "numDimensions": dims, "similarity": "cosine"},
        {"type": "filter", "path": "law_id"},
        {"type": "filter", "path": "chunk_type"},
        {"type": "filter", "path": "document_type"},
    ]
    if "chunks_vector" in existing:
        db.chunks.update_search_index("chunks_vector", {"fields": vector_fields})
        print("Updated chunks_vector index.")
    else:
        db.chunks.create_search_index({
            "name": "chunks_vector",
            "type": "vectorSearch",
            "definition": {"fields": vector_fields},
        })
        print("Created chunks_vector index.")

    text_definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "text": {"type": "string", "analyzer": "ro_analyzer"},
                "text_normalized": {"type": "string", "analyzer": "lucene.romanian"},
                "title": {"type": "string", "analyzer": "ro_analyzer"},
            },
        },
        "analyzers": [{
            "name": "ro_analyzer",
            "charFilters": [],
            "tokenizer": {"type": "standard"},
            "tokenFilters": [
                {"type": "lowercase"},
                {"type": "icuFolding"},
                {
                    "type": "stopword",
                    "tokens": [
                        "și", "de", "la", "în", "cu", "pe", "pentru",
                        "din", "care", "este", "sau", "nu", "se", "prin",
                        "acest", "această", "ale", "cel", "cea",
                        "si", "in", "pentru", "din",
                    ],
                },
                {"type": "snowballStemming", "stemmerName": "romanian"},
            ],
        }],
    }
    if "chunks_search_ro" in existing:
        db.chunks.update_search_index("chunks_search_ro", text_definition)
        print("Updated chunks_search_ro index.")
    else:
        db.chunks.create_search_index({
            "name": "chunks_search_ro",
            "type": "search",
            "definition": text_definition,
        })
        print("Created chunks_search_ro index.")

    print("Indexes ensured.")


if __name__ == "__main__":
    from legalro_core.config import load_settings
    from legalro_core.store import get_db

    settings = load_settings()
    db = get_db(settings)
    ensure_indexes(db, settings)
