"""
tests/test_observability_ci.py

Unit tests for Prometheus metrics exporter and CI observability endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.monitoring.prometheus_exporter import LightweightPrometheusExporter


@pytest.fixture
def client():
    return TestClient(app)


def test_prometheus_exporter_unit():
    """Verifies LightweightPrometheusExporter logic and metric formatting."""
    exporter = LightweightPrometheusExporter()
    exporter.record_inquiry(action="AUTO", confidence=0.85, evidence_count=3)
    exporter.record_inquiry(action="ESCALATE", confidence=0.42, evidence_count=0)

    text = exporter.generate_metrics_text()
    assert "# HELP hiver_uptime_seconds" in text
    assert '# TYPE hiver_inquiries_total counter' in text
    assert 'hiver_inquiries_total{action="AUTO"} 1' in text
    assert 'hiver_inquiries_total{action="ESCALATE"} 1' in text
    assert "hiver_rag_evidence_items_total 3" in text
    assert 'hiver_intent_confidence_bucket{le="0.5"} 1' in text
    assert 'hiver_intent_confidence_bucket{le="+Inf"} 2' in text


def test_prometheus_endpoint(client):
    """Verifies GET /metrics endpoint returns 200 OK and text/plain content type."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "hiver_uptime_seconds" in response.text
    assert "hiver_inquiries_total" in response.text


def test_inquiry_updates_prometheus(client):
    """Verifies POST /api/v1/inquire increments Prometheus counter."""
    payload = {
        "customer_message": "Where is my Amazon package delivery?",
        "interaction_id": "test_prom_101"
    }
    resp = client.post("/api/v1/inquire", json=payload)
    assert resp.status_code == 200

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert 'hiver_inquiries_total' in metrics_resp.text
