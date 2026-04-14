# Himak Backend API
### AI-Powered Islamic Hadith Semantic Search — FastAPI + Pinecone + sentence-transformers

---

## Folder Structure

```
himak-backend/
│
├── main.py                          # FastAPI app factory + lifespan startup
│
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── router.py            # Aggregates all v1 endpoint routers
│   │       └── endpoints/
│   │           └── search.py        # POST /api/v1/search  ← core endpoint
│   │
│   ├── core/
│   │   ├── config.py                # All settings from .env (pydantic-settings)
│   │   ├── exceptions.py            # Custom exceptions + global handlers
│   │   └── logger.py                # Shared structured logger
│   │
│   ├── schemas/
│   │   └── search.py                # Pydantic request/response models
│   │
│   └── services/
│       ├── embedding_service.py     # sentence-transformers wrapper
│       ├── vector_store.py          # Pinecone wrapper (connect, query, upsert)
│       └── search_service.py        # Orchestrates embed → search → map
│
├── scripts/
│   └── ingest.py                    # One-time dataset → Pinecone ingestion
│
├── tests/
│   └── test_search.py               # Unit + integration tests (pytest)
│
├── data/                            # Put your hadiths.json / hadiths.csv here
│
├── .env.example                     # Copy to .env and fill in secrets
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <your-repo>
cd himak-backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in PINECONE_API_KEY and other values
```

### 3. Ingest your Hadith dataset into Pinecone

```bash
# JSON format
python scripts/ingest.py --dataset data/hadiths.json --format json

# CSV format
python scripts/ingest.py --dataset data/hadiths.csv --format csv

# Dry run — embeds but does NOT write to Pinecone (safe for testing)
python scripts/ingest.py --dataset data/hadiths.json --format json --dry-run
```

**Expected JSON structure** (adjust `ingest.py` if your columns differ):
```json
[
  {
    "id": "bukhari-1",
    "matn": "Actions are judged by intentions.",
    "arabic_text": "إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ",
    "translation_en": "Actions are judged by intentions.",
    "collection": "Sahih Bukhari",
    "book": "Revelation",
    "chapter": "How the Divine Revelation started",
    "hadith_number": "1",
    "grade": "Sahih",
    "narrator": "Umar ibn al-Khattab",
    "isnad": "...",
    "translation_ur": "..."
  }
]
```

### 4. Start the API server

```bash
uvicorn main:app --reload          # development
uvicorn main:app --workers 2       # production (or use Docker)
```

Server starts at **http://localhost:8000**

---

## API Reference

### `POST /api/v1/search`

Semantic search for Hadiths matching a natural-language query.

**Request body**
```json
{
  "query": "What does Islam say about honesty in business?",
  "top_k": 5,
  "min_score": 0.30
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | — | User's natural-language query (3–500 chars) |
| `top_k` | integer | ❌ | 5 | Number of results to return (1–20) |
| `min_score` | float | ❌ | 0.30 | Minimum cosine similarity threshold (0–1) |

**Response 200**
```json
{
  "success": true,
  "query": "What does Islam say about honesty in business?",
  "total_results": 3,
  "results": [
    {
      "rank": 1,
      "score": 0.8932,
      "hadith_id": "bukhari-2082",
      "matn": "The truthful and honest merchant will be with the Prophets...",
      "arabic_text": "التَّاجِرُ الصَّدُوقُ الأَمِينُ مَعَ النَّبِيِّينَ...",
      "translation_en": "The truthful and honest merchant will be with the Prophets...",
      "collection": "Jami al-Tirmidhi",
      "hadith_number": "1209",
      "grade": "Sahih",
      "narrator": "Abu Sa'id al-Khudri",
      "isnad": "...",
      "book": "Business",
      "chapter": "Honest Merchants"
    }
  ]
}
```

**Error responses**
| Status | When |
|--------|------|
| 422 | Invalid payload (query too short, top_k out of range, etc.) |
| 429 | Rate limit exceeded |
| 503 | Embedding model or Pinecone unavailable |

### `GET /health`
```json
{ "status": "ok", "version": "1.0.0" }
```

### Interactive docs
- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc:       `http://localhost:8000/api/v1/redoc`

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker Deployment

```bash
# Build
docker build -t himak-backend .

# Run (inject secrets via environment variables — never bake them into the image)
docker run -p 8000:8000 \
  -e PINECONE_API_KEY=your-key \
  -e PINECONE_INDEX_NAME=himak-hadiths \
  himak-backend
```

---

## Architecture: How a Search Request Flows

```
Mobile App
    │
    │  POST /api/v1/search  {"query": "..."}
    ▼
FastAPI Router  (app/api/v1/endpoints/search.py)
    │  validates payload with Pydantic
    ▼
SearchService  (app/services/search_service.py)
    │
    ├─► EmbeddingService.embed_query(query)
    │       sentence-transformers → 384-dim unit vector
    │       (runs in thread-pool executor — never blocks event loop)
    │
    ├─► VectorStore.query(vector, top_k)
    │       Pinecone cosine similarity search
    │       returns top-k (id, score, metadata) matches
    │
    └─► filter by min_score → map to HadithResult → SearchResponse
    │
    ▼
FastAPI serialises SearchResponse → JSON → Mobile App
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Services loaded once in lifespan** | Avoids re-downloading the ~120MB model on every request. App startup takes ~5s; requests take ~300ms. |
| **Thread-pool executor for embedding** | `sentence-transformers` is synchronous. Running it directly in an `async` route would block the event loop and kill concurrency. |
| **`min_score` filtering** | Cosine similarity never returns "no result" — without a threshold the API would return irrelevant Hadiths with low scores on every query. |
| **Metadata stored in Pinecone** | Avoids a separate database round-trip per search. All Hadith fields needed for display are stored alongside the vector. |
| **Multilingual model** | `paraphrase-multilingual-MiniLM-L12-v2` handles both Arabic and English queries natively — essential for the Himak use case. |
| **Pydantic v2 schemas** | Single source of truth for validation, serialisation, and OpenAPI documentation. No separate OpenAPI YAML needed. |