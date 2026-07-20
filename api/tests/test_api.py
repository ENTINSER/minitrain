import os
import sys

# Demo mode prevents the API from launching real GPU training during tests.
os.environ.setdefault("TRAINING_DEMO_MODE", "true")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"train_jobs_total" in response.content


def test_create_train_job():
    payload = {
        "trigger_reason": "Accuracy dropped below threshold",
        "error_type": "NullPointerException",
        "low_score_samples": 238,
        "date_range": "2024-01-01/2024-01-31",
        "current_model": "Qwen2.5-7B",
        "current_accuracy": 72.1,
        "target_accuracy": 85.0,
    }
    response = client.post("/train-jobs", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] in ("queued", "running")

    get_response = client.get(f"/train-jobs/{data['job_id']}")
    assert get_response.status_code == 200
    assert get_response.json()["status"] in ("queued", "running", "completed")


def test_invalid_train_job_payload():
    response = client.post("/train-jobs", json={"current_accuracy": "not-a-number"})
    assert response.status_code == 422


def test_get_nonexistent_job():
    response = client.get("/train-jobs/does-not-exist")
    assert response.status_code == 404


def test_agent_factory_training_trigger():
    """Agent Factory emits a smaller payload; /training-trigger maps it internally."""
    payload = {
        "error_type": "off_by_one",
        "avg_score": 62.5,
        "sample_count": 10,
        "trigger_time": "2026-07-19T08:11:53+00:00",
    }
    response = client.post("/training-trigger", json=payload)
    assert response.status_code == 202, response.text
    data = response.json()
    assert "job_id" in data
    assert data["request"]["error_type"] == "off_by_one"
    assert data["request"]["current_accuracy"] == 62.5
