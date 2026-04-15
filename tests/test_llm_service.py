"""
tests/test_llm_service.py
-------------------------
Unit tests for the LLMService greeting fallback logic.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_greeting_returns_fallback_without_llm_call(monkeypatch):
    mock_generate = AsyncMock()
    mock_model = MagicMock()
    mock_model.generate_content_async = mock_generate

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("app.services.llm_service.genai.configure", MagicMock())
    monkeypatch.setattr(
        "app.services.llm_service.genai.GenerativeModel",
        MagicMock(return_value=mock_model),
    )

    service = LLMService()
    result = await service.get_chat_response("Hi", [])

    assert "Himak" in result
    assert "Hadith" in result or "Hadith assistant" in result
    mock_generate.assert_not_called()


@pytest.mark.parametrize(
    "greeting",
    ["hello", "hey", "assalamualaikum", "good morning", "salam"],
)
@pytest.mark.asyncio
async def test_greeting_variants_are_handled(monkeypatch, greeting):
    mock_generate = AsyncMock()
    mock_model = MagicMock()
    mock_model.generate_content_async = mock_generate

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("app.services.llm_service.genai.configure", MagicMock())
    monkeypatch.setattr(
        "app.services.llm_service.genai.GenerativeModel",
        MagicMock(return_value=mock_model),
    )

    service = LLMService()
    result = await service.get_chat_response(greeting, [])

    assert "Himak" in result
    mock_generate.assert_not_called()
