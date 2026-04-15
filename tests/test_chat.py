"""
tests/test_chat.py
------------------
Integration tests for the chat endpoint response shape.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport

from main import app
from app.api.v1.endpoints import chat as chat_module
from app.schemas.search import SearchResponse, HadithResult


@pytest.mark.asyncio
async def test_chat_endpoint_returns_structured_sources(monkeypatch):
    dummy_hadith = HadithResult(
        id="1123",
        text="Actions are judged by intentions.",
        score=0.95,
        metadata={
            "matn": "Actions are judged by intentions.",
            "collection": "Sahih Bukhari",
            "book": "Faith",
            "hadith_number": "1123",
            "grade": "Sahih",
            "narrator": "Aisha",
            "arabic_text": "الأعمال بالنيات",
            "translation_en": "Actions are judged by intentions.",
        },
        rank=1,
        collection="Sahih Bukhari",
        grade="Sahih",
        arabic_text="الأعمال بالنيات",
        narrator="Aisha",
        book="Faith",
        chapter="1",
        hadith_number="1123",
        translation_en="Actions are judged by intentions.",
    )

    mock_search = AsyncMock(return_value=SearchResponse(query="importance of prayer", results=[dummy_hadith]))
    mock_llm = AsyncMock(return_value="Prayer is important according to authentic Hadith.")

    monkeypatch.setattr(chat_module.search_service, "search", mock_search)
    monkeypatch.setattr(chat_module.llm_service, "get_chat_response", mock_llm)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/ask",
            json={"question": "importance of prayer?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Prayer is important according to authentic Hadith."
    assert isinstance(data["sources"], list)
    assert data["sources"][0]["hadith_number"] == "1123"
    assert data["sources"][0]["narrator"] == "Aisha"
    assert data["sources"][0]["arabic_text"] == "الأعمال بالنيات"
    assert data["sources"][0]["translation_en"] == "Actions are judged by intentions."
    assert data["sources"][0]["book"] == "Faith"
    assert data["sources"][0]["collection"] == "Sahih Bukhari"
