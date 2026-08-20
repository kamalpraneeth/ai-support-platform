# AI Support Platform — Deployment

The platform is containerized using Docker and is ready for deployment to any modern cloud environment (AWS ECS, Google Cloud Run, Azure Container Apps, or Kubernetes).

## 1. Dockerization

The application is built using a minimal `python:3.11-slim` base image.

### Building the Image
```bash
docker build -t ai-support-platform:latest .
```

### Running Locally
To run the container locally, you must provide your Groq API key:
```bash
docker run -d \
  -p 8000:8000 \
  -e GROQ_API_KEY="your-api-key-here" \
  --name ai-support \
  ai-support-platform:latest
```

## 2. Environment Variables

The application behavior is controlled via environment variables. In production, these should be securely injected via a secret manager.

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | Your API key for Groq's LLaMA 3.1 LLM | Yes (for AI) | None (falls back to canned replies) |
| `DATABASE_URL` | SQLAlchemy connection string | No | `sqlite:///./support.db` |
| `PORT` | The port the FastAPI server runs on inside the container | No | `8000` |
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins | No | `*` |

## 3. CI/CD Pipeline

We utilize GitHub Actions (`.github/workflows/ci.yml`) for Continuous Integration.

**Trigger:** The workflow runs on every `push` and `pull_request` to any branch.

**Jobs:**
1. **Linting**: Runs `flake8` to enforce Python PEP8 style guidelines.
2. **ML Training & Validation**: Runs `python -m app.ml.train` to ensure the ML pipeline can train successfully on the latest data and evaluates performance.
3. **Automated Tests**: Runs the `pytest` test suite, executing >160 tests covering the API, RAG pipeline, ML models, evaluation heuristics, and Responsible AI constraints.
4. **Coverage Reporting**: Generates a `coverage.xml` report.

If any of these steps fail, the CI pipeline fails, preventing broken code from merging.

## 4. Production Considerations

When moving to a production environment:
1. **Database**: Replace the default SQLite database with a production-grade PostgreSQL instance by updating the `DATABASE_URL`.
2. **Gunicorn Workers**: The Dockerfile currently runs `uvicorn` directly. For high-throughput production, consider wrapping `uvicorn` in `gunicorn` with multiple worker processes.
3. **CORS Security**: Ensure `CORS_ORIGINS` is strictly limited to your actual frontend domain names to prevent unauthorized API access.
