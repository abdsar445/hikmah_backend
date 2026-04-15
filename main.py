"""
Himak Backend — AI-Powered Islamic Hadith Chatbot API
Entry point: starts the FastAPI application.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logger import logger


# ---------------------------------------------------------------------------
# Lifespan: runs once on startup and once on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy resources (embedding model, Pinecone client) once at startup
    and store them in app.state so every request handler can reach them via
    `request.app.state`.
    """
    logger.info("[START] Himak backend starting up...")

    # 1. Load the sentence-transformer model into memory
    from app.services.embedding_service import EmbeddingService
    from app.services.vector_store import VectorStore
    from app.services.search_service import SearchService
    from app.services.llm_service import LLMService

    embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    embedding_service.load()
    app.state.embedding_service = embedding_service
    logger.info(f"[SUCCESS] Embedding model loaded: {settings.EMBEDDING_MODEL}")

    # 2. Connect to Pinecone and obtain the target index
    vector_store = VectorStore(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX_NAME,
    )
    vector_store.connect()
    app.state.vector_store = vector_store
    logger.info(f"[SUCCESS] Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")

    # 3. Create high-level services and store them on application state
    app.state.search_service = SearchService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    app.state.llm_service = LLMService()

    yield  # application runs here

    # Cleanup (models release memory automatically; log for visibility)
    logger.info("[STOP] Himak backend shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Semantic Hadith search API powering the Himak mobile application.",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        lifespan=lifespan,
    )

    # CORS — tighten `allow_origins` before going to production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount all versioned routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "version": settings.VERSION}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )