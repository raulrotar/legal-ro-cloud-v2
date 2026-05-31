"""Storage operations — MongoDB (metadata + vectors via $vectorSearch)."""
from pymongo import MongoClient
from legalro_core.config import Settings
from legalro_core.models import Gazette

_mongo_client = None
_mongo_db = None


# ── MongoDB ───────────────────────────────────────────────────────────────────

def get_db(settings: Settings):
    global _mongo_client, _mongo_db
    if _mongo_db is None:
        _mongo_client = MongoClient(settings.mongodb.uri)
        _mongo_db = _mongo_client[settings.mongodb.database]
    return _mongo_db


def store_gazette(gazette: Gazette, settings: Settings) -> str:
    db = get_db(settings)
    doc = {
        "issue_number": gazette.issue_number,
        "part": gazette.part,
        "date": gazette.date.isoformat(),
        "year": gazette.year,
        "era": gazette.era.value,
        "filename": gazette.filename,
        "sha256": gazette.sha256,
        "page_count": gazette.page_count,
        "act_count": gazette.act_count,
        "status": gazette.status,
    }
    result = db.gazettes.insert_one(doc)
    return str(result.inserted_id)


def store_acts(acts: list[dict], settings: Settings) -> list[str]:
    db = get_db(settings)
    result = db.acts.insert_many(acts)
    return [str(oid) for oid in result.inserted_ids]


def store_chunks(chunks: list[dict], settings: Settings) -> list[str]:
    db = get_db(settings)
    result = db.chunks.insert_many(chunks)
    return [str(oid) for oid in result.inserted_ids]


def gazette_exists(filename: str, settings: Settings) -> bool:
    db = get_db(settings)
    return db.gazettes.find_one({"filename": filename}) is not None

