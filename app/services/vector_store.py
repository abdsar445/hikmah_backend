"""
app/services/vector_store.py
-----------------------------
Wraps the Pinecone client.

Responsibilities:
  • connect()  — called once at startup; initialises the Pinecone client and
                 fetches a handle to the target index.
  • query()    — called per search request; runs a nearest-neighbour search
                 and returns raw Pinecone matches.
  • upsert()   — called by the ingestion script; bulk-inserts vectors.

The rest of the app never imports pinecone directly — all Pinecone concerns
are isolated here, so swapping to a different vector DB later touches only
this one file.
"""

from __future__ import annotations

from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import VectorStoreError


class VectorStore:
    def __init__(self, api_key: str, index_name: str):
        self._api_key = api_key
        self._index_name = index_name
        self._client: Pinecone | None = None
        self._index = None          # pinecone.Index handle

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Initialise the Pinecone client and bind to the target index.
        If the index does not exist yet it will be created automatically
        (useful for first-time setup in a fresh environment).
        Call once at application startup.
        """
        try:
            self._client = Pinecone(api_key=self._api_key)

            existing = [idx.name for idx in self._client.list_indexes()]
            if self._index_name not in existing:
                logger.warning(
                    f"VectorStore: index '{self._index_name}' not found — creating it. "
                    f"Dimension={settings.EMBEDDING_DIMENSION}, metric=cosine"
                )
                self._client.create_index(
                    name=self._index_name,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            self._index = self._client.Index(self._index_name)
            stats = self._index.describe_index_stats()
            logger.info(
                f"VectorStore: connected to '{self._index_name}'. "
                f"Total vectors: {stats.total_vector_count}"
            )
        except Exception as exc:
            logger.exception(f"VectorStore: connection failed — {exc}")
            raise VectorStoreError(f"Cannot connect to Pinecone index '{self._index_name}'.") from exc

    # ── Search ─────────────────────────────────────────────────────────────

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str | None = None,
        filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run a nearest-neighbour search against the index.

        Returns a list of match dicts, each containing:
          {
            "id":       str,
            "score":    float,       # cosine similarity 0-1
            "metadata": dict,        # all fields stored at upsert time
          }

        Raises VectorStoreError on failure.
        """
        self._ensure_connected()
        ns = namespace or settings.PINECONE_NAMESPACE
        try:
            response = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=ns,
                include_metadata=True,
                filter=filter,          # e.g. {"grade": {"$eq": "Sahih"}}
            )
            return [
                {
                    "id":       match.id,
                    "score":    match.score,
                    "metadata": match.metadata or {},
                }
                for match in response.matches
            ]
        except Exception as exc:
            logger.exception(f"VectorStore: query failed — {exc}")
            raise VectorStoreError("Pinecone query failed.") from exc

    # ── Ingestion ──────────────────────────────────────────────────────────

    def upsert(
        self,
        vectors: list[dict],        # [{"id": str, "values": [...], "metadata": {...}}, …]
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """
        Upsert vectors in batches.  Called by scripts/ingest.py.
        Returns total number of vectors upserted.
        """
        self._ensure_connected()
        ns = namespace or settings.PINECONE_NAMESPACE
        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            try:
                self._index.upsert(vectors=batch, namespace=ns)
                total += len(batch)
                logger.info(f"VectorStore: upserted batch {i // batch_size + 1} ({total} total)")
            except Exception as exc:
                logger.warning(f"Batch {i // batch_size + 1} failed. Falling back to item-by-item to isolate massive rows...")
                for item in batch:
                    try:
                        self._index.upsert(vectors=[item], namespace=ns)
                        total += 1
                    except Exception as item_exc:
                        logger.error(f"Failed item limit. Aggressively truncating metadata...")
                        for k in list(item['metadata'].keys()):
                            if isinstance(item['metadata'][k], str) and len(item['metadata'][k].encode('utf-8')) > 2000:
                                item['metadata'][k] = item['metadata'][k][:1997] + "..."
                        try:
                            self._index.upsert(vectors=[item], namespace=ns)
                            total += 1
                        except Exception as e3:
                            logger.error(f"Item {item['id']} irrecoverable. Skipping. {e3}")
        return total

    def delete(self, ids: list[str], namespace: str | None = None) -> None:
        """Delete specific vectors by ID."""
        self._ensure_connected()
        ns = namespace or settings.PINECONE_NAMESPACE
        self._index.delete(ids=ids, namespace=ns)

    def describe(self) -> dict:
        """Return index statistics (useful for health checks)."""
        self._ensure_connected()
        return self._index.describe_index_stats().to_dict()

    # ── Private ────────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._index is None:
            raise VectorStoreError(
                "VectorStore is not connected. Call connect() at startup."
            )