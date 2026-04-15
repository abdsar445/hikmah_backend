"""
tests/test_chat.py
------------------
Integration tests for the chat endpoint response shape.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

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

    state = SimpleNamespace(
        search_service=SimpleNamespace(search=mock_search),
        llm_service=SimpleNamespace(get_chat_response=mock_llm),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    response = await chat_module.ask_himak(
        request,
        chat_module.ChatRequest(question="importance of prayer?"),
    )

    assert response.answer == "Prayer is important according to authentic Hadith."
    assert isinstance(response.sources, list)
    assert response.sources[0].hadith_number == "1123"
    assert response.sources[0].narrator == "Aisha"
    assert response.sources[0].arabic_text == "الأعمال بالنيات"
    assert response.sources[0].translation_en == "Actions are judged by intentions."
    assert response.sources[0].book == "Faith"
    assert response.sources[0].collection == "Sahih Bukhari"
