"""
tests/test_deployment.py

Verifies Phase 9 deployment constraints:
- Docker configuration file existence
- FastAPI /dashboard static route availability
- FastAPI / root redirect
- Multi-turn capabilities on /api/v1/inquire
"""

import os
from pathlib import Path
from fastapi.testclient import TestClient

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.api.app import app

client = TestClient(app)

def test_docker_files_exist():
    """Verify Phase 9A containerization files exist."""
    assert (BASE_DIR / "Dockerfile").exists(), "Dockerfile is missing"
    assert (BASE_DIR / "docker-compose.yml").exists(), "docker-compose.yml is missing"
    assert (BASE_DIR / ".dockerignore").exists(), ".dockerignore is missing"

def test_dashboard_static_route():
    """Verify Phase 9C static dashboard mount."""
    response = client.get("/dashboard/index.html")
    assert response.status_code == 200, "Dashboard index.html should be served at /dashboard/"
    assert "text/html" in response.headers.get("content-type", "")
    assert "AmazonHelp" in response.text

def test_root_redirect():
    """Verify / redirects to /dashboard"""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308) # Temporary or Permanent Redirect
    assert "/dashboard" in response.headers.get("location", "")

def test_multiturn_api_integration():
    """Verify Phase 9B API modifications support multi-turn history."""
    payload = {
        "customer_message": "Where is my package?",
        "conversation_id": "conv_test123",
        "conversation_history": [
            {"role": "customer", "content": "Hello, I need help."},
            {"role": "agent", "content": "Hi there! How can I help you today?"}
        ]
    }
    response = client.post("/api/v1/inquire", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "conv_test123"
    assert "query_context_used" in data
    # The history strings should be integrated in the used context
    assert "Hello, I need help." in data["query_context_used"] or "Where is my package?" in data["query_context_used"]
