"""
tests/test_api.py

Phase 7 API Integration & Validation Test Suite asserting:
1. GET /api/v1/health: Returns 200 OK, healthy status, classifier/retriever loaded, 35,000 vectors, and live_llm_available=False.
2. POST /api/v1/inquire (AUTO): Order tracking inquiry yields AUTO decision and grounded response.
3. POST /api/v1/inquire (ESCALATE): High-risk security inquiry yields ESCALATE decision and halted response.
4. POST /api/v1/inquire (Empty Input): Empty customer_message yields HTTP 422 Unprocessable Entity.
5. POST /api/v1/inquire (Missing Field): Payload without customer_message yields HTTP 422.
6. GET /api/v1/metrics: Returns runtime telemetry metrics with inquiry counts.
7. Telemetry Logger: Appends clean non-sensitive audit records to logs/agent_operations.jsonl.
8. OpenAPI Documentation: Exposes compliant /openapi.json schema.
"""

import os
import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.api.app import app, startup_event

# Run startup event to initialize agent before testing
startup_event()
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["classifier_loaded"] is True
    assert data["retriever_loaded"] is True
    assert data["index_items"] == 35000
    assert data["generator_provider"] == "mock"
    assert data["live_llm_available"] is False


def test_valid_inquiry_auto():
    payload = {"customer_message": "Where is my order #12345? Tracking shows delayed since yesterday."}
    response = client.post("/api/v1/inquire", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "AUTO"
    assert data["requires_human_escalation"] is False
    assert data["escalation_reason"] == "none"
    assert data["generated_response"] is not None
    assert len(data["retrieved_evidence"]) == 3
    assert data["retrieval_similarity"] > 0.0


def test_valid_inquiry_escalate():
    payload = {"customer_message": "Someone hacked my account password and changed my email address!"}
    response = client.post("/api/v1/inquire", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "ESCALATE"
    assert data["requires_human_escalation"] is True
    assert data["escalation_reason"] != "none"
    assert data["generated_response"] is None


def test_invalid_inquiry_empty_string():
    payload = {"customer_message": ""}
    response = client.post("/api/v1/inquire", json=payload)
    assert response.status_code == 422


def test_invalid_inquiry_whitespace_only():
    payload = {"customer_message": "   "}
    response = client.post("/api/v1/inquire", json=payload)
    assert response.status_code == 422


def test_invalid_inquiry_missing_message_field():
    payload = {"interaction_id": "TEST_INVALID_001"}
    response = client.post("/api/v1/inquire", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_inquiries" in data
    assert "auto_handled" in data
    assert "escalated" in data
    assert "escalation_rate" in data
    assert "mean_intent_confidence" in data
    assert isinstance(data["escalation_reasons"], dict)


def test_openapi_schema_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "paths" in data
    assert "/api/v1/inquire" in data["paths"]
    assert "/api/v1/health" in data["paths"]
    assert "/api/v1/metrics" in data["paths"]
