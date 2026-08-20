from fastapi.testclient import TestClient


def test_create_chat_session(client: TestClient):
    response = client.post("/chat/session")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["message"] == "Chat session created successfully."


def test_send_chat_message_invalid_session(client: TestClient):
    response = client.post(
        "/chat/message",
        json={
            "session_id": 999999,
            "message": "Hello"})
    assert response.status_code == 404


def test_send_chat_message_success(client: TestClient, monkeypatch):
    # Mock LLM call to avoid hitting real API
    def fake_call_llm(messages):
        return "mock reply", 10.0, True

    import app.ai_orchestrator
    monkeypatch.setattr(app.ai_orchestrator, "_call_llm", fake_call_llm)

    # 1. Create session
    res = client.post("/chat/session")
    session_id = res.json()["session_id"]

    # 2. Send message
    msg_payload = {"session_id": session_id, "message": "My router is broken"}
    res2 = client.post("/chat/message", json=msg_payload)
    assert res2.status_code == 200
    data = res2.json()
    assert data["session_id"] == session_id
    assert "reply" in data
    assert data["is_ai_generated"] is True
    assert data["reply"] == "mock reply"


def test_chat_maintains_context(client: TestClient, monkeypatch):
    # Mock LLM to return a deterministic string based on history
    call_history = []

    def fake_call_llm(messages):
        call_history.append(messages)
        return "mock reply", 10.0, True

    import app.ai_orchestrator
    monkeypatch.setattr(app.ai_orchestrator, "_call_llm", fake_call_llm)

    res = client.post("/chat/session")
    session_id = res.json()["session_id"]

    # Turn 1
    client.post(
        "/chat/message",
        json={
            "session_id": session_id,
            "message": "First message"})

    # Turn 2
    client.post(
        "/chat/message",
        json={
            "session_id": session_id,
            "message": "Second message"})

    # Check that the final call to LLM contained the first message as history
    # (Note: it may be called multiple times due to evaluation retries, so we check the last call)
    second_call_messages = call_history[-1]

    # Structure of messages in second call:
    # [system, user(Turn 1), assistant(Turn 1 reply), user(Turn 2 context+query)]

    assert len(second_call_messages) == 4
    assert second_call_messages[0]["role"] == "system"
    assert second_call_messages[1]["role"] == "user"
    assert second_call_messages[1]["content"] == "First message"
    assert second_call_messages[2]["role"] == "assistant"
    assert second_call_messages[2]["content"] == "mock reply"
    assert second_call_messages[3]["role"] == "user"
    assert "[CUSTOMER QUERY]\nSecond message" in second_call_messages[3]["content"]
