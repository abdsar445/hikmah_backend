from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from fastapi import APIRouter
from app.services.search_service import SearchService
from app.services.llm_service import LLMService # Make sure this file name is correct!
from pydantic import BaseModel


router = APIRouter()

# 1. Create instances of the required services
from app.core.config import settings

embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
embedding_service.load()
# Fix: Give VectorStore the API key and Index Name from settings
vector_store = VectorStore(
    api_key=settings.PINECONE_API_KEY,
    index_name=settings.PINECONE_INDEX_NAME
)
vector_store.connect()
# 2. Pass them INTO SearchService
search_service = SearchService(
    embedding_service=embedding_service, 
    vector_store=vector_store
)

llm_service = LLMService()

# This tells FastAPI to expect a JSON object like {"question": "your question"}
class ChatRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_himak(request: ChatRequest):
    # 1. Search the Vector DB (Pinecone) for relevant Hadiths
    # We use request.question now
    related_hadiths = await search_service.search(query=request.question, limit=10)
    
    # 2. Pass those Hadiths to the LLM (Gemini)
    ai_answer = await llm_service.get_chat_response(request.question, related_hadiths)
    
    return {
        "answer": ai_answer,
        "sources": related_hadiths
    }