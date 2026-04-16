from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

if TYPE_CHECKING:
    from app.services.search_service import SearchService
    from app.services.llm_service import LLMService

router = APIRouter()

# This tells FastAPI to expect a JSON object like {"question": "your question"}
class ChatRequest(BaseModel):
    question: str

class ChatSource(BaseModel):
    hadith_number: Optional[str]
    narrator: Optional[str]
    book: Optional[str]
    collection: Optional[str]
    arabic_text: Optional[str]
    translation_en: Optional[str]
    text: Optional[str]
    grade: Optional[str]

class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]

@router.post("/ask", response_model=ChatResponse)
async def ask_himak(request: Request, body: ChatRequest):
    if not hasattr(request.app.state, "search_service"):
        raise RuntimeError("SearchService is not configured on the application state.")
    if not hasattr(request.app.state, "llm_service"):
        raise RuntimeError("LLMService is not configured on the application state.")

    search_service = request.app.state.search_service
    llm_service = request.app.state.llm_service

    related_hadiths = await search_service.search(query=body.question, limit=10)
    ai_answer = await llm_service.get_chat_response(body.question, related_hadiths.results)

    return ChatResponse(
        answer=ai_answer,
        sources=[
            ChatSource(
                hadith_number=hadith.hadith_number,
                narrator=hadith.narrator,
                book=hadith.book,
                collection=hadith.collection,
                arabic_text=hadith.arabic_text,
                translation_en=hadith.translation_en or hadith.text,
                text=hadith.text,
                grade=hadith.grade,
            )
            for hadith in related_hadiths.results
        ],
    )
