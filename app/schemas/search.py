from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HadithResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    rank: Optional[int] = None
    collection: Optional[str] = None
    grade: Optional[str] = None
    arabic_text: Optional[str] = None
    narrator: Optional[str] = None
    book: Optional[str] = None
    chapter: Optional[str] = None
    hadith_number: Optional[str] = None
    isnad: Optional[str] = None
    translation_en: Optional[str] = None
    translation_ur: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[HadithResult]

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    limit: int = Field(10, ge=1,le=100)
    min_score: float = Field(0.10,ge=0.0,le=1.0)

class ErrorResponse(BaseModel):
    detail: str