from fastapi.testclient import TestClient

from serve.api import app


class StubModel:
    def generate(self, question, schema):
        return f"SELECT 1 -- {question}"


def test_health_and_generate():
    app.state.model = StubModel()
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/generate", json={"question": "how many?", "schema": "CREATE TABLE t (a INT)"})
        assert r.status_code == 200
        assert r.json()["sql"].startswith("SELECT 1")


def test_rejects_empty_question():
    app.state.model = StubModel()
    with TestClient(app) as client:
        r = client.post("/generate", json={"question": "", "schema": "CREATE TABLE t (a INT)"})
        assert r.status_code == 422
