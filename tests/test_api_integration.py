"""End-to-end API tests for MiniTrain, including Agent Factory compatibility."""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"train_jobs_total" in response.content


def test_create_train_job() -> None:
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


def test_agent_factory_training_trigger() -> None:
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
    assert data["request"]["trigger_reason"] == "Agent Factory automatic trigger"
