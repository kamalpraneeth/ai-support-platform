# AI Support Platform — Monitoring & Analytics

Observability is a critical requirement for production GenAI applications. The platform implements comprehensive tracking of both application health and AI performance.

## 1. Monitoring (`app/monitoring.py`)

The monitoring module focuses on real-time operational metrics and logging.

### Structured Logging
All logs are emitted using structured JSON formatting. This allows log aggregation systems (e.g., Datadog, ELK, CloudWatch) to easily parse and query the logs. Critical events like model loading, LLM latency, and fallback activations are logged with execution context.

### In-Memory Counters
We track vital operational statistics using thread-safe, in-memory counters (similar to Prometheus metrics format).
- `llm_calls_total`, `llm_calls_success`, `llm_calls_failed`
- `llm_total_latency_ms` (for calculating average latency)
- `rag_retrievals_total`
- `prompt_injections_detected`
- `pii_detected`

These metrics are exposed via the `GET /metrics` endpoint for scraping by monitoring tools.

## 2. Analytics (`app/analytics.py`)

The analytics module focuses on business and AI performance metrics derived from the persisted database.

### Ticket Analytics
Exposed via `GET /analytics`, this aggregates data directly from the SQLite database to provide:
- **Category Distribution**: Breakdown of ticket volume by ML-predicted category.
- **Sentiment Distribution**: Breakdown by sentiment (Positive, Neutral, Negative).
- **Urgency Distribution**: Breakdown by urgency.

### AI Session Analytics
Exposed via `GET /analytics/ai`, this provides insights into the GenAI generation phase:
- Total LLM calls.
- Total Tokens Used (estimated for cost tracking).
- Average LLM latency.
- Error rates.

These endpoints are designed to back a frontend administration dashboard for customer support managers to monitor system health and ticket trends.
