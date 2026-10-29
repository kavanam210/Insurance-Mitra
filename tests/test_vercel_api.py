import logging

from fastapi import HTTPException
from google.genai.errors import APIError

from api.app import NO_CONTEXT_ANSWER, QuestionRequest, answer_question, retrieve_evidence
from api.app import app
from fastapi.testclient import TestClient

client = TestClient(app, raise_server_exceptions=False)


def test_vercel_retrieval_finds_waiting_period_evidence():
    sources = retrieve_evidence("What does a waiting period mean for health insurance?")

    assert sources
    assert sources[0]["source"] == "insurance_knowledge.md"
    assert "waiting period" in sources[0]["text"].lower()


def test_vercel_retrieval_rejects_unrelated_questions():
    assert retrieve_evidence("Explain quantum entanglement in distant galaxies") == []


def test_vercel_no_evidence_does_not_need_gemini(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = answer_question("Explain quantum entanglement in distant galaxies")

    assert result == {"answer": NO_CONTEXT_ANSWER, "sources": [], "grounded": False}


def test_question_input_is_trimmed_and_bounded():
    assert QuestionRequest(question="  Explain premiums.  ").question == "Explain premiums."


def test_vercel_home_and_health_routes():
    home = client.get("/")
    health = client.get("/api/health")

    assert home.status_code == 200
    assert "Insurance Mitra" in home.text
    assert health.json() == {"ok": True, "data": {"status": "ok"}}


def test_ask_returns_success_envelope(monkeypatch):
    result = {"answer": "A premium is a price.", "sources": [], "grounded": True}
    monkeypatch.setattr("api.app.answer_question", lambda question: result)

    response = client.post("/api/ask", json={"question": "What is a premium?"})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "data": result}


def test_validation_error_uses_json_envelope_and_422():
    response = client.post("/api/ask", json={"question": "   "})

    assert response.status_code == 422
    assert response.json() == {
        "ok": False,
        "error": {"code": "invalid_request", "message": "Enter a question under 1,200 characters."},
    }


def test_http_error_preserves_status_and_uses_json_envelope(monkeypatch):
    def fail(_question):
        raise HTTPException(status_code=503, detail="The answer service is not configured.")

    monkeypatch.setattr("api.app.answer_question", fail)

    response = client.post("/api/ask", json={"question": "What is a premium?"})

    assert response.status_code == 503
    assert response.json() == {
        "ok": False,
        "error": {"code": "request_failed", "message": "The answer service is not configured."},
    }


def test_gemini_failure_logs_safe_details_and_closes_client(monkeypatch, caplog):
    api_key = "synthetic-test-api-key"

    class FailingClient:
        def __init__(self):
            self.models = self
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def generate_content(self, **_kwargs):
            raise APIError(429, {"error": {"message": f"Quota exceeded for {api_key}", "status": "RESOURCE_EXHAUSTED"}})

    fake_client = FailingClient()
    monkeypatch.setenv("GOOGLE_API_KEY", api_key)
    monkeypatch.setattr("api.app.genai.Client", lambda **_kwargs: fake_client)

    with caplog.at_level(logging.ERROR, logger="api.app"):
        response = client.post(
            "/api/ask",
            json={"question": "What does a waiting period mean for health insurance?"},
        )

    assert response.status_code == 502
    assert response.json() == {
        "ok": False,
        "error": {"code": "request_failed", "message": "Gemini could not complete the request. Please try again."},
    }
    assert "code=429" in caplog.text
    assert "RESOURCE_EXHAUSTED" in caplog.text
    assert api_key not in caplog.text
    assert api_key not in response.text
    assert fake_client.closed


def test_unexpected_error_is_logged_but_not_returned(monkeypatch, caplog):
    api_key = "synthetic-unexpected-api-key"

    def fail(_question):
        raise RuntimeError(f"internal failure with {api_key}")

    monkeypatch.setenv("GOOGLE_API_KEY", api_key)
    monkeypatch.setattr("api.app.answer_question", fail)

    with caplog.at_level(logging.ERROR, logger="api.app"):
        response = client.post("/api/ask", json={"question": "What is a premium?"})

    assert response.status_code == 500
    assert response.json() == {
        "ok": False,
        "error": {"code": "internal_error", "message": "The service encountered an unexpected error. Please try again."},
    }
    assert "RuntimeError" in caplog.text
    assert api_key not in caplog.text
    assert api_key not in response.text
