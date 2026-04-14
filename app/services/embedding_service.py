"""
app/services/embedding_service.py
-----------------------------------
Wraps sentence-transformers so the rest of the app never imports it directly.

Design decisions:
  • Model is loaded ONCE in the lifespan hook (main.py) and stored on
    app.state.embedding_service — never re-loaded per request.
  • encode() is synchronous because sentence-transformers uses PyTorch under
    the hood.  We run it inside FastAPI's thread-pool via
    asyncio.get_event_loop().run_in_executor() in the route handler, so it
    never blocks the async event loop.
  • Normalisation (normalize_embeddings=True) turns the raw vector into a
    unit vector, making Pinecone's dot-product metric equivalent to cosine
    similarity — the most meaningful distance for semantic search.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.logger import logger
from app.core.exceptions import EmbeddingError


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Download (first run) or load from local cache the sentence-transformer
        model.  Call once at startup.
        """
        try:
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                f"EmbeddingService: model '{self.model_name}' ready. "
                f"Dimension={self._model.get_sentence_embedding_dimension()}"
            )
        except Exception as exc:
            logger.exception(f"EmbeddingService: failed to load model — {exc}")
            raise EmbeddingError(f"Could not load embedding model '{self.model_name}'.") from exc

    # ── Public API ─────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.
        Returns a normalised float list ready to send to Pinecone.
        Raises EmbeddingError on any failure.
        """
        self._ensure_loaded()
        try:
            vector: np.ndarray = self._model.encode(
                text,
                normalize_embeddings=True,   # unit-vector → cosine via dot-product
                convert_to_numpy=True,
            )
            return vector.tolist()
        except Exception as exc:
            logger.exception(f"EmbeddingService: embed_query failed — {exc}")
            raise EmbeddingError("Failed to generate embedding for query.") from exc

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Embed a list of texts in batches.
        Used by the ingestion script (scripts/ingest.py), not the search endpoint.
        """
        self._ensure_loaded()
        try:
            vectors: np.ndarray = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            return vectors.tolist()
        except Exception as exc:
            logger.exception(f"EmbeddingService: embed_batch failed — {exc}")
            raise EmbeddingError("Failed to generate batch embeddings.") from exc

    @property
    def dimension(self) -> int:
        """Returns the vector dimension of the loaded model."""
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    # ── Private helpers ────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise EmbeddingError(
                "EmbeddingService is not initialised. Call load() at startup."
            )