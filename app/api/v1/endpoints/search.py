"""
app/api/v1/endpoints/search.py
--------------------------------
Defines the POST /api/v1/search endpoint.

This layer has ONE job: translate HTTP ↔ service layer.
No business logic lives here — that belongs in SearchService.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request, status

from app.schemas.search import SearchRequest, SearchResponse, ErrorResponse
from app.core.logger import logger
from app.core.exceptions import EmbeddingError, VectorStoreError

if TYPE_CHECKING:
    from app.services.search_service import SearchService

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
    if not hasattr(request.app.state, "search_service"):
        raise RuntimeError("SearchService is not configured on the application state.")

    search_service = request.app.state.search_service

    result = await search_service.search(
        query=body.query,
        limit=body.limit,
        min_score=body.min_score,
    )

    return result