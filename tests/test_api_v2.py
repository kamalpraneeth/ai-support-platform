"""
v2.0.0 API tests: new fields and analytics endpoints.
Uses the same client fixture from conftest.py.
"""


def submit_ticket(client, text: str):
    return client.post("/ticket", json={"text": text})


# -- v2.0.0 New Field Tests ---------------------------------------------------

class TestTicketV2Fields:
    """Verify new fields added in v2.0.0 are present in responses."""

    def test_submit_ticket_returns_ml_confidence(self, client):
        res = submit_ticket(
            client, "I was charged twice for my subscription this month")
        data = res.json()
        assert "ml_confidence" in data
        assert 0.0 <= data["ml_confidence"] <= 1.0

    def test_submit_ticket_returns_escalated_flag(self, client):
        res = submit_ticket(
            client, "General question about pricing plans and features")
        data = res.json()
        assert "escalated" in data
        assert isinstance(data["escalated"], bool)

    def test_reply_returns_rag_chunks_used(self, client):
        ticket_res = submit_ticket(
            client, "I need help with a billing refund for duplicate charge")
        ticket_id = ticket_res.json()["id"]
        res = client.post("/ticket/reply", json={"ticket_id": ticket_id})
        data = res.json()
        assert "rag_chunks_used" in data
        assert data["rag_chunks_used"] >= 0

    def test_reply_returns_llm_latency_ms(self, client):
        ticket_res = submit_ticket(
            client, "How do I update my billing information?")
        ticket_id = ticket_res.json()["id"]
        res = client.post("/ticket/reply", json={"ticket_id": ticket_id})
        data = res.json()
        assert "llm_latency_ms" in data
        assert data["llm_latency_ms"] >= 0.0

    def test_reply_returns_escalated_flag(self, client):
        ticket_res = submit_ticket(
            client, "I want to know about your pricing plans")
        ticket_id = ticket_res.json()["id"]
        res = client.post("/ticket/reply", json={"ticket_id": ticket_id})
        data = res.json()
        assert "escalated" in data
        assert isinstance(data["escalated"], bool)

    def test_reply_returns_prompt_template(self, client):
        ticket_res = submit_ticket(
            client, "The application keeps crashing on startup")
        ticket_id = ticket_res.json()["id"]
        res = client.post("/ticket/reply", json={"ticket_id": ticket_id})
        data = res.json()
        assert "prompt_template" in data
        assert isinstance(data["prompt_template"], str)


# -- Analytics Endpoint Tests -------------------------------------------------

class TestAnalyticsEndpoints:
    def test_analytics_returns_200(self, client):
        res = client.get("/analytics")
        assert res.status_code == 200

    def test_analytics_tickets_returns_200(self, client):
        res = client.get("/analytics/tickets")
        assert res.status_code == 200

    def test_analytics_ai_returns_200(self, client):
        res = client.get("/analytics/ai")
        assert res.status_code == 200

    def test_metrics_returns_200(self, client):
        res = client.get("/metrics")
        assert res.status_code == 200

    def test_analytics_tickets_has_total(self, client):
        res = client.get("/analytics/tickets")
        data = res.json()
        assert "total_tickets" in data

    def test_analytics_ai_has_session_stats(self, client):
        res = client.get("/analytics/ai")
        data = res.json()
        assert "session_stats" in data

    def test_health_returns_components_status(self, client):
        res = client.get("/health")
        data = res.json()
        assert "components" in data
        assert "ml_classifier" in data["components"]
