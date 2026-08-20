# AI Support Platform — API Documentation

The platform provides a RESTful API built with FastAPI. Base URL when running locally: `http://localhost:8000`

## 1. Ticket Management Endpoints

### `POST /ticket`
Submit a new support ticket. The system automatically classifies the ticket using the ML model.

**Request Body (JSON):**
```json
{
  "text": "I was charged twice for my subscription this month."
}
```

**Response (200 OK):**
```json
{
  "id": 1,
  "text": "I was charged twice for my subscription this month.",
  "category": "Billing",
  "urgency": "High",
  "sentiment": "Negative",
  "ml_confidence": 0.92,
  "escalated": false,
  "created_at": "2026-08-20T10:00:00Z"
}
```

---

### `POST /ticket/reply`
Generate an AI response for an existing ticket using RAG and the LLM.

**Request Body (JSON):**
```json
{
  "ticket_id": 1
}
```

**Response (200 OK):**
```json
{
  "reply": "Thank you for reaching out. To request a refund for the duplicate charge, please contact our billing team within 30 days...",
  "is_ai_generated": true,
  "rag_chunks_used": 2,
  "llm_latency_ms": 1450.5,
  "prompt_template": "support_reply_v1",
  "escalated": false
}
```

## 2. Analytics & Monitoring Endpoints

### `GET /analytics`
Retrieve aggregated analytics for tickets.

**Response (200 OK):**
```json
{
  "category_distribution": {
    "Billing": 45,
    "Technical": 30,
    "Account": 15,
    "General": 10
  },
  "sentiment_distribution": {
    "Positive": 15,
    "Neutral": 45,
    "Negative": 40
  },
  "urgency_distribution": {
    "High": 20,
    "Medium": 50,
    "Low": 30
  }
}
```

---

### `GET /analytics/tickets`
Retrieve recent tickets for dashboard display.

**Response (200 OK):**
```json
{
  "total_tickets": 100,
  "recent_tickets": [
    {
      "id": 100,
      "category": "Billing",
      "urgency": "High",
      "escalated": false
    }
  ]
}
```

---

### `GET /analytics/ai`
Retrieve AI session usage statistics and monitoring metrics.

**Response (200 OK):**
```json
{
  "session_stats": {
    "total_llm_calls": 50,
    "successful_llm_calls": 49,
    "failed_llm_calls": 1,
    "total_tokens_used": 15000
  }
}
```

## 3. System & Health Endpoints

### `GET /health`
Check the health of the application and its dependencies.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "ml_classifier": "ready",
    "rag_retriever": "ready",
    "database": "connected"
  }
}
```

### `GET /metrics`
Retrieve raw internal counters for monitoring (Prometheus-style key-value format for internal logging).

**Response (200 OK):**
```json
{
  "llm_calls_total": 50,
  "llm_calls_success": 49,
  "llm_calls_failed": 1,
  "rag_retrievals_total": 45,
  "prompt_injections_detected": 2,
  "pii_detected": 5
}
```
