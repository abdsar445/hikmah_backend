"""
app/services/search_service.py
--------------------------------
The business-logic brain of the search pipeline.

Flow:
  1. Receive raw query string
  2. Embed via EmbeddingService (run in thread-pool to avoid blocking the loop)
  3. Query VectorStore for top-k matches
  4. Filter out low-confidence results (below min_score)
  5. Map raw Pinecone matches → HadithResult Pydantic models
  6. Return SearchResponse

This layer has no HTTP concerns — it is unit-testable in isolation.
"""

from __future__ import annotations

import asyncio
from functools import partial

from app.core.logger import logger
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.schemas.search import HadithResult, SearchResponse


class SearchService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self._embedder = embedding_service
        self._store = vector_store

    # ── Public API ─────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.10,
    ) -> SearchResponse:
        """
        Full semantic search pipeline.
        Returns a SearchResponse ready to be returned from the route handler.
        """
        logger.info(f"SearchService: query='{query[:80]}' limit={limit} min_score={min_score}")

        # ── Step 1: Embed the query ────────────────────────────────────────
        # sentence-transformers is a synchronous, CPU/GPU-bound operation.
        # We run it in the default thread-pool executor so it does not block
        # the asyncio event loop and other requests can be served concurrently.
        loop = asyncio.get_event_loop()
        embed_fn = partial(self._embedder.embed_query, query)
        query_vector = await loop.run_in_executor(None, embed_fn)

        # ── Step 2: Nearest-neighbour search in Pinecone ───────────────────
        # We fetch a few extra results so we still have `limit` after filtering.
        fetch_k = min(limit + 5, 20)
        raw_matches = self._store.query(vector=query_vector, top_k=limit,)

        # ── Step 3: Filter + rank ──────────────────────────────────────────
        filtered = [m for m in raw_matches if m["score"] >= min_score]
        filtered = filtered[:limit]          # honour the requested limit

        # ── Step 4: Map to response schema ────────────────────────────────
        results = [self._map_match(rank=i + 1, match=m) for i, m in enumerate(filtered)]

        logger.info(
            f"SearchService: returned {len(results)} results "
            f"(raw={len(raw_matches)}, after_filter={len(filtered)})"
        )

        return SearchResponse(
            query=query,
            results=results
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _map_match(rank: int, match: dict) -> HadithResult:
        """
        Convert a raw Pinecone match dict into a typed HadithResult.

        All Hadith metadata fields were stored in Pinecone at ingestion time.
        We use .get() with None defaults so missing fields never crash the API.
        """
        meta = match.get("metadata", {})
        return HadithResult(
            rank=rank,
            score=round(match["score"], 4),
            id=match["id"],                            # Changed from hadith_id
            text=meta.get("matn") or meta.get("text") or "No text Found", # Changed from matn
            metadata=meta,                             # Added this field
            isnad=meta.get("isnad", ""),
            narrator=meta.get("narrator"),
            collection=meta.get("collection"),
            book=meta.get("book"),
            chapter=meta.get("chapter"),
            hadith_number=meta.get("hadith_number"),
            grade=meta.get("grade"),
            arabic_text=meta.get("arabic_text"),
            translation_en=meta.get("translation_en"),
            translation_ur=meta.get("translation_ur")
        )
        