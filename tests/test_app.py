from fastapi.testclient import TestClient

from llmgw.app import app


def test_complete_endpoint_caches_the_static_block_on_the_second_call():
    client = TestClient(app)
    req = {
        "team": "test-team-app",
        "backend": "fast-sim",
        "static_block_id": "test-team-app-instructions",
        "static_tokens": 300,
        "variable_tokens": 150,
        "difficulty": "easy",
    }
    first = client.post("/complete", json=req).json()
    second = client.post("/complete", json=req).json()
    assert first["cached_prefix_tokens"] == 0
    assert second["cached_prefix_tokens"] == 300


def test_team_report_reflects_recorded_calls():
    client = TestClient(app)
    req = {
        "team": "test-team-report",
        "backend": "fast-sim",
        "static_block_id": "test-team-report-instructions",
        "static_tokens": 300,
        "variable_tokens": 150,
        "difficulty": "easy",
    }
    client.post("/complete", json=req)
    client.post("/complete", json=req)
    report = client.get("/teams/test-team-report/report").json()
    assert report["n_requests"] == 2
