"""
app/api/v1/endpoints/search.py
--------------------------------
Defines the POST /api/v1/search endpoint.

This layer has ONE job: translate HTTP ↔ service layer.
No business logic lives here — that belongs in SearchService.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.schemas.search import SearchRequest, SearchResponse, ErrorResponse
from app.services.search_service import SearchService
from app.core.logger import logger
from app.core.exceptions import EmbeddingError, VectorStoreError

router = APIRouter()


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Hadith Search",
    description=(
        "Embed the user's natural-language query and return the top-k most "
        "semantically similar Hadiths from the Pinecone vector database."
    ),
    responses={
        200: {"model": SearchResponse, "description": "Successful search response."},
        422: {"model": ErrorResponse, "description": "Invalid request payload."},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
        503: {"model": ErrorResponse, "description": "AI or database service unavailable."},
    },
    tags=["Search"],
)
async def search_hadiths(
    request: Request,
    body: SearchRequest,
) -> SearchResponse:
    """
    **Core semantic search endpoint.**

    The mobile app sends a JSON body:
    ```json
    {
      "query": "What does Islam say about honesty?",
      "limit": 5,
      "min_score": 0.30
    }
    ```

    The backend:
    1. Validates the payload (Pydantic).
    2. Embeds the query using the pre-loaded sentence-transformer model.
    3. Queries Pinecone for the nearest neighbours by cosine similarity.
    4. Filters results below `min_score`.
    5. Returns a ranked list of Hadiths with metadata.
    """
    # Pull services from app.state (loaded once at startup)
    embedding_service = request.app.state.embedding_service
    vector_store = request.app.state.vector_store

    # Instantiate the service (lightweight — no I/O in __init__)
    search_service = SearchService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # Delegate all work to the service layer
    # EmbeddingError / VectorStoreError bubble up to the registered handlers
    result = await search_service.search(
        query=body.query,
        limit=body.limit,
        min_score=body.min_score,
    )

    return result