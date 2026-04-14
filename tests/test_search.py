"""
tests/test_search.py
---------------------
Unit and integration tests for the search pipeline.

Run with:
    pytest tests/ -v

Requires:
    pip install pytest pytest-asyncio httpx
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from main import app
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.search_service import SearchService
from app.schemas.search import SearchRequest


# ── Fixtures ───────────────────────────────────────────────────────────────

MOCK_VECTOR = [0.1] * 384       # 384-dim unit vector (paraphrase-multilingual-MiniLM)

MOCK_PINECONE_MATCHES = [
    {
        "id": "bukhari-1",
        "score": 0.91,
        "metadata": {
            "matn": "Actions are judged by intentions.",
            "collection": "Sahih Bukhari",
            "book": "Revelation",
            "hadith_number": "1",
            "grade": "Sahih",
            "narrator": "Umar ibn al-Khattab",
            "arabic_text": "إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ",
            "translation_en": "Actions are judged by intentions.",
        },
    },
    {
        "id": "muslim-23",
        "score": 0.85,
        "metadata": {
            "matn": "None of you truly believes until he loves for his brother what he loves for himself.",
            "collection": "Sahih Muslim",
            "hadith_number": "45",
            "grade": "Sahih",
        },
    },
]


@pytest.fixture
def mock_embedding_service():
    svc = MagicMock(spec=EmbeddingService)
    svc.embed_query.return_value = MOCK_VECTOR
    return svc


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.query.return_value = MOCK_PINECONE_MATCHES
    return store


# ── Unit tests: SearchService ──────────────────────────────────────────────

class TestSearchService:
    @pytest.mark.asyncio
    async def test_returns_correct_number_of_results(
        self, mock_embedding_service, mock_vector_store
    ):
        svc = SearchService(mock_embedding_service, mock_vector_store)
        response = await svc.search("test query", limit=5)
        assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_results_are_ranked_by_score(
        self, mock_embedding_service, mock_vector_store
    ):
        svc = SearchService(mock_embedding_service, mock_vector_store)
        response = await svc.search("test query")
        assert response.results[0].score >= response.results[1].score

    @pytest.mark.asyncio
    async def test_min_score_filter_excludes_low_results(
        self, mock_embedding_service, mock_vector_store
    ):
        # Set a high threshold that excludes the second result (score=0.85)
        # Wait — 0.85 > 0.90 threshold means zero results; use 0.90 threshold
        mock_vector_store.query.return_value = [
            {"id": "h1", "score": 0.92, "metadata": {"matn": "High score hadith"}},
            {"id": "h2", "score": 0.40, "metadata": {"matn": "Low score hadith"}},
        ]
        svc = SearchService(mock_embedding_service, mock_vector_store)
        response = await svc.search("query", min_score=0.90)
        assert len(response.results) == 1
        assert response.results[0].score == 0.92

    @pytest.mark.asyncio
    async def test_rank_starts_at_one(self, mock_embedding_service, mock_vector_store):
        svc = SearchService(mock_embedding_service, mock_vector_store)
        response = await svc.search("query")
        assert response.results[0].rank == 1
        assert response.results[1].rank == 2

    @pytest.mark.asyncio
    async def test_metadata_fields_mapped_correctly(
        self, mock_embedding_service, mock_vector_store
    ):
        svc = SearchService(mock_embedding_service, mock_vector_store)
        response = await svc.search("query")
        first = response.results[0]
        assert first.id == "bukhari-1"
        assert first.collection == "Sahih Bukhari"
        assert first.grade == "Sahih"
        assert first.arabic_text == "إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ"

    @pytest.mark.asyncio
    async def test_missing_metadata_fields_are_none(
        self, mock_embedding_service, mock_vector_store
    ):
        mock_vector_store.query.return_value = [
            {"id": "sparse-1", "score": 0.75, "metadata": {"matn": "Sparse record"}},
        ]
        svc = SearchService(mock_embedding_service, mock_vector_store)
        response = await svc.search("query")
        result = response.results[0]
        assert result.collection is None
        assert result.grade is None
        assert result.arabic_text is None


# ── Integration tests: HTTP endpoint ──────────────────────────────────────

@pytest.fixture
def app_with_mocks(mock_embedding_service, mock_vector_store):
    """Inject mock services into app.state for HTTP-level tests."""
    app.state.embedding_service = mock_embedding_service
    app.state.vector_store = mock_vector_store
    return app


@pytest.mark.asyncio
async def test_search_endpoint_200(app_with_mocks):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mocks), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/v1/search",
            json={"query": "What does Islam say about honesty?"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["rank"] == 1


@pytest.mark.asyncio
async def test_search_endpoint_422_on_short_query(app_with_mocks):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mocks), base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/search", json={"query": "hi"})
    assert resp.status_code == 422      # Pydantic min_length=3 validation


@pytest.mark.asyncio
async def test_search_endpoint_422_on_missing_query(app_with_mocks):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mocks), base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/search", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_endpoint_respects_top_k(app_with_mocks, mock_vector_store):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mocks), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/v1/search",
            json={"query": "patience in Islam", "limit": 1},
        )
    assert resp.status_code == 200
    # With top_k=1 only the first match should be returned
    assert len(resp.json()["results"]) == 1


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"