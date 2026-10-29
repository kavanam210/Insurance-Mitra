from api.app import NO_CONTEXT_ANSWER, QuestionRequest, answer_question, retrieve_evidence
from api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


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
    assert health.json() == {"status": "ok"}
