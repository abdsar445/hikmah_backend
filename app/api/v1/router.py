from fastapi import APIRouter
from app.api.v1.endpoints.search import router as search_router
# 1. ADD THIS LINE:
from app.api.v1.endpoints.chat import router as chat_router 

api_router = APIRouter()

api_router.include_router(search_router, prefix="/search", tags=["Search"])
# 2. ADD THIS LINE:
api_router.include_router(chat_router, prefix="/chat", tags=["Chat"])