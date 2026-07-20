# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# AI-Powered Customer Support Platform — Single-container Dockerfile
#
# Build stages:
#   1. python:3.11-slim base
#   2. Install dependencies
#   3. Copy source code
#   4. Train the classifier and bake the .pkl into the image
#   5. Expose port 8000 and start uvicorn
#
# Deploy on Render:
#   - Set Docker environment in the Render service settings
#   - Add GROQ_API_KEY as an environment variable in Render's dashboard
#   - Port: 8000
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Prevents Python from writing .pyc files; keeps stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (Docker layer cache: only invalidated when
# requirements.txt changes, not on every source code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Train the classifier and save the .pkl
# (Runs at build time so cold starts are instant)
RUN python -m app.ml.train

# Render uses the PORT env var; default to 8000
EXPOSE 8000

# Health check — Render also monitors /health via HTTP
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the server
# Use $PORT for Render compatibility (Render sets PORT automatically)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
