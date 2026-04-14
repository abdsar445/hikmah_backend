"""
scripts/ingest.py
------------------
One-time (or re-runnable) script that:
  1. Reads your Hadith dataset (JSON or CSV).
  2. Builds the text to embed — by default: "collection: matn" (adjust to taste).
  3. Embeds all records in batches using the same sentence-transformer.
  4. Upserts the vectors + metadata into Pinecone.

Run from the project root:
    python scripts/ingest.py --dataset data/hadiths.json --format json
    python scripts/ingest.py --dataset data/hadiths.csv  --format csv

Prerequisites:
  • .env is populated with PINECONE_API_KEY etc.
  • `pip install -r requirements.txt` has been run.
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
import os
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.core.logger import logger
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


# ── Helpers ────────────────────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both a top-level list and {"hadiths": [...]}
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("hadiths", "data", "results"):
            if key in data:
                return data[key]
    raise ValueError("Unrecognised JSON structure. Expected a list or {'hadiths': [...]}")


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_embed_text(record: dict) -> str:
    """
    Construct the string that will be embedded.
    Combining collection name + matn gives the model rich context.
    Adjust this to match your actual column names.
    """
    parts = []
    if record.get("collection"):
        parts.append(record["collection"])
    if record.get("chapter"):
        parts.append(record["chapter"])
    matn = record.get("English") or record.get("text") or record.get("body") or ""
    parts.append(matn)
    return " | ".join(p for p in parts if p)


def build_vector_record(record: dict, vector: list[float], idx: int) -> dict:
    """
    Map a raw dataset record to the Pinecone upsert format.
    Adjust field names to match your CSV/JSON columns.
    """
    matn = record.get("English") or record.get("text") or record.get("body") or ""
    arabic = record.get("arabic_text") or record.get("Arabic") or ""
    
    # ── Pinecone 40KB limits safe-guard ──────────────────────────────────
    # Truncate extremely massive rows to fit inside Pinecone limit
    if len(matn.encode('utf-8')) > 15000:
        matn = matn[:14997] + "..."
    if len(arabic.encode('utf-8')) > 15000:
        arabic = arabic[:14997] + "..."
        
    raw_id = (
        record.get("id")
        or record.get("hadith_id")
        or record.get("_id")
        or record.get("")
    )
    
    if raw_id is not None and str(raw_id).strip() != "":
        hadith_id = str(raw_id)
    else:
        if matn.strip():
            hadith_id = str(hash(matn))
        else:
            hadith_id = f"row_{idx}"
            
    if idx < 10:
        logger.info(f"Row {idx} -> Generated ID '{hadith_id}' | matn preview: {matn[:30].strip()}...")

    return {
        "id": hadith_id,
        "values": vector,
        "metadata": {
            # ── Required fields ──────────────────────────────────────────
            "matn":          matn,
            # ── Optional fields (include what you have) ───────────────────
            "isnad":         record.get("isnad", ""),
            "narrator":      record.get("narrator") or record.get("Narrated") or "",
            "collection":    record.get("collection", ""),
            "book":          record.get("book") or record.get("Book") or "",
            "chapter":       record.get("chapter") or record.get("Reference") or "",
            "hadith_number": str(record.get("hadith_number", "")),
            "grade":         record.get("grade", ""),
            "arabic_text":   arabic,
            "translation_en": (record.get("translation_en") or record.get("translation", ""))[:10000],
            "translation_ur":record.get("translation_ur", ""),
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest Hadith dataset into Pinecone.")
    parser.add_argument("--dataset", required=True, help="Path to your JSON or CSV file.")
    parser.add_argument("--format",  choices=["json", "csv"], default="json")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Records per embedding batch (default 64).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Embed but do NOT upsert — useful for testing.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of records to ingest for testing.")
    args = parser.parse_args()

    # ── Load dataset ───────────────────────────────────────────────────────
    logger.info(f"Loading dataset from: {args.dataset}")
    if args.format == "json":
        records = load_json(args.dataset)
    else:
        records = load_csv(args.dataset)

    if args.limit:
        records = records[:args.limit]
        logger.info(f"Limiting to first {args.limit} records for testing...")

    logger.info(f"Loaded {len(records)} records.")

    # ── Connect to services ────────────────────────────────────────────────
    embedder = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    embedder.load()

    store = VectorStore(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX_NAME,
    )
    if not args.dry_run:
        store.connect()

    # ── Embed ALL in single go ──────────────────────────────────────────────
    texts = [build_embed_text(r) for r in records]
    logger.info(f"Embedding ALL {len(texts)} texts in batches of {args.batch_size}...")
    all_vectors = embedder.embed_batch(texts, batch_size=args.batch_size)
    
    # ── Build Pinecone records ─────────────────────────────────────────────
    pinecone_records = [
        build_vector_record(record, vector, idx)
        for idx, (record, vector) in enumerate(zip(records, all_vectors))
    ]
    
    if args.dry_run:
        logger.info(f"Dry run — sample record:\n{json.dumps(pinecone_records[0], indent=2, ensure_ascii=False)}")
        logger.info("Dry run complete. No data was written to Pinecone.")
        return
        
    # ── Upsert into Pinecone ───────────────────────────────────────────────
    try:
        total = store.upsert(pinecone_records, batch_size=100)
        logger.info(f"[SUCCESS] Ingestion complete — {total} vectors upserted to '{settings.PINECONE_INDEX_NAME}'.")
    except Exception as e:
        logger.error(f"[FATAL] Failed to upsert: {e}")


if __name__ == "__main__":
    main()