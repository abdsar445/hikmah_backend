# ============================================================
#  Himak Backend — Dockerfile
#  Multi-stage build: keeps the final image lean.
# ============================================================

# ── Stage 1: Build dependencies ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime image ────────────────────────────────────
FROM python:3.11-slim AS runtime

# Hugging Face Spaces strictly requires running as User 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source with appropriate permissions
COPY --chown=user . $HOME/app

# Pre-download the embedding model so the container starts instantly
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

EXPOSE 7860

# We no longer need restrictive memory limits because Hugging Face gives us 16 GB of RAM!
ENV MALLOC_ARENA_MAX=2

# Spin up Uvicorn on Hugging Face's required native Port 7860
CMD uvicorn main:app --host 0.0.0.0 --port 7860 --workers 2 --log-level info