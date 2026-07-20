"""
Integration tests for the FastAPI endpoints.

Uses FastAPI's TestClient via the session-scoped `client` fixture defined in
conftest.py. That fixture:
  - Points SQLAlchemy at an in-memory SQLite DB (isolated, no file I/O)
  - Triggers the FastAPI lifespan so the ML classifier is loaded
  - Never touches the production support.db file
"""

import os
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def submit_ticket(client, text: str):
    """Submit a ticket and return the response."""
    return client.post("/ticket", json={"text": text})


# ── Health Check Tests ────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """GET /health should always return HTTP 200."""
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_response_body(self, client):
        """GET /health should return status=ok."""
        res = client.get("/health")
        data = res.json()
        assert data["status"] == "ok"


# ── Frontend Endpoint ─────────────────────────────────────────────────────────

class TestFrontendEndpoint:
    def test_root_returns_html(self, client):
        """GET / should serve the HTML frontend."""
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]

    def test_root_contains_form(self, client):
        """Frontend HTML should contain the ticket submission form."""
        res = client.get("/")
        assert "ticket-form" in res.text


# ── POST /ticket Tests ────────────────────────────────────────────────────────

class TestSubmitTicketEndpoint:
    def test_submit_valid_ticket_status(self, client):
        """POST /ticket with valid text returns HTTP 200."""
        res = submit_ticket(client, "I was charged twice for my subscription this month")
        assert res.status_code == 200

    def test_submit_returns_all_fields(self, client):
        """POST /ticket response must include id, category, urgency, sentiment."""
        res = submit_ticket(client, "The app crashes every time I try to upload a file")
        data = res.json()
        assert "id" in data
        assert "category" in data
        assert "urgency" in data
        assert "sentiment" in data

    def test_submit_category_is_valid(self, client):
        """Category in response must be one of the four valid values."""
        res = submit_ticket(client, "I cannot log into my account after the password reset")
        data = res.json()
        assert data["category"] in ("Billing", "Technical", "Account", "General")

    def test_submit_urgency_is_valid(self, client):
        """Urgency in response must be High, Medium, or Low."""
        res = submit_ticket(client, "Do you offer a free trial for enterprise customers?")
        data = res.json()
        assert data["urgency"] in ("High", "Medium", "Low")

    def test_submit_sentiment_is_valid(self, client):
        """Sentiment in response must be Positive, Neutral, or Negative."""
        res = submit_ticket(client, "I love the platform but there is a small issue")
        data = res.json()
        assert data["sentiment"] in ("Positive", "Neutral", "Negative")

    def test_submit_empty_text_returns_422(self, client):
        """POST /ticket with empty text should return HTTP 422 (validation error)."""
        res = client.post("/ticket", json={"text": ""})
        assert res.status_code == 422

    def test_submit_short_text_returns_422(self, client):
        """POST /ticket with text shorter than 5 chars returns 422."""
        res = client.post("/ticket", json={"text": "hi"})
        assert res.status_code == 422

    def test_submit_missing_body_returns_422(self, client):
        """POST /ticket with no body returns 422."""
        res = client.post("/ticket", json={})
        assert res.status_code == 422

    def test_ticket_id_increments(self, client):
        """Consecutive ticket submissions should have incrementing IDs."""
        res1 = submit_ticket(client, "First ticket about billing issue with my invoice")
        res2 = submit_ticket(client, "Second ticket about technical problem with the app")
        assert res2.json()["id"] > res1.json()["id"]


# ── POST /ticket/reply Tests ──────────────────────────────────────────────────

class TestTicketReplyEndpoint:
    def test_reply_returns_200_for_existing_ticket(self, client):
        """POST /ticket/reply with valid ticket_id returns HTTP 200."""
        ticket_res = submit_ticket(client, "I cannot connect to the API and need help urgently")
        ticket_id = ticket_res.json()["id"]

        res = client.post("/ticket/reply", json={"ticket_id": ticket_id})
        assert res.status_code == 200

    def test_reply_response_contains_reply_text(self, client):
        """Reply response must contain a non-empty reply string."""
        ticket_res = submit_ticket(client, "My account has been suspended without warning")
        ticket_id = ticket_res.json()["id"]

        res = client.post("/ticket/reply", json={"ticket_id": ticket_id})
        data = res.json()
        assert "reply" in data
        assert len(data["reply"]) > 10

    def test_reply_contains_is_ai_generated_flag(self, client):
        """Reply response must include the is_ai_generated boolean flag."""
        ticket_res = submit_ticket(client, "I would like to know more about your pricing plans")
        ticket_id = ticket_res.json()["id"]

        res = client.post("/ticket/reply", json={"ticket_id": ticket_id})
        data = res.json()
        assert "is_ai_generated" in data
        assert isinstance(data["is_ai_generated"], bool)

    def test_reply_for_nonexistent_ticket_returns_404(self, client):
        """POST /ticket/reply with a bogus ticket ID should return 404."""
        res = client.post("/ticket/reply", json={"ticket_id": 999999})
        assert res.status_code == 404

    def test_reply_fallback_when_no_api_key(self, client):
        """
        Without GROQ_API_KEY, the reply should return a non-empty
        fallback string and is_ai_generated should be False.
        """
        os.environ.pop("GROQ_API_KEY", None)

        ticket_res = submit_ticket(client, "I have a general question about your service availability")
        ticket_id = ticket_res.json()["id"]

        res = client.post("/ticket/reply", json={"ticket_id": ticket_id})
        data = res.json()
        assert res.status_code == 200
        assert data["is_ai_generated"] is False
        assert len(data["reply"]) > 20
