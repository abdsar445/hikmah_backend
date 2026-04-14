import os
from dotenv import load_dotenv
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.core.config import settings

def ingest():
    load_dotenv()
    
    # 1. Initialize Services
    embedder = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    embedder.load()
    
    vector_store = VectorStore(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX_NAME
    )
    vector_store.connect()

    # 2. Your Data (Example)
    hadiths = [
        {"id": "h1", "text": "Actions are but by intentions...", "source": "Bukhari"},
        {"id": "h2", "text": "The best among you is he who learns the Quran...", "source": "Bukhari"},
    ]

    # 3. Process and Upload
    vectors_to_upsert = []
    for item in hadiths:
        # Generate embedding for the text
        vector = embedder.embed_query(item["text"])
        
        # Prepare the Pinecone record
        vectors_to_upsert.append({
            "id": item["id"],
            "values": vector,
            "metadata": {"text": item["text"], "source": item["source"]}
        })

    # 4. Upsert to Pinecone
    vector_store.upsert(vectors=vectors_to_upsert)
    print(f"Successfully uploaded {len(vectors_to_upsert)} Hadiths to Pinecone!")

if __name__ == "__main__":
    ingest()