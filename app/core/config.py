"""
app/core/config.py
------------------
All configuration is read from environment variables (or a .env file).
Never hard-code secrets here — use .env.example as a template.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Project metadata ──────────────────────────────────────────────────
    PROJECT_NAME: str = "Hikmah API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # ── Security ──────────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production"          # JWT signing key
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24           # 24 hours

    # ── CORS ──────────────────────────────────────────────────────────────
    # Comma-separated list in .env, e.g. CORS_ORIGINS=http://localhost:3000,https://myapp.com
    CORS_ORIGINS: List[str] = ["*"]

    # ── Pinecone ──────────────────────────────────────────────────────────
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "himak-hadiths"
    PINECONE_NAMESPACE: str = ""  
    GOOGLE_API_KEY: str

    # ── Embedding model (Hugging Face sentence-transformers) ──────────────
    # "all-MiniLM-L6-v2"  →  fast, 384-dim, great for English
    # "paraphrase-multilingual-MiniLM-L12-v2"  →  multilingual (Arabic + English)
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384    # must match what Pinecone index was created with

    # ── Search defaults ────────────────────────────────────────────────────
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    MIN_SCORE_THRESHOLD: float = 0.30   # discard results below this cosine similarity

    # ── Rate limiting (requests per minute per IP) ─────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


# Single import-time instance used everywhere
settings = Settings()