"""
src/api/app.py

FastAPI production service wrapping the Hiver Twitter AI Support Agent.
Exposes REST endpoints:
- POST /api/v1/inquire: Processes customer inquiry and returns structured decision & evidence
- GET /api/v1/health: Returns system health, index status, and provider state
- GET /api/v1/metrics: Returns runtime operational telemetry
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress Python 3.14 resource tracker process exit on macOS
try:
    import multiprocessing.resource_tracker
    multiprocessing.resource_tracker._resource_tracker.main = lambda *args, **kwargs: None
except Exception:
    pass

from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure root directory is on Python path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.agent import TwitterSupportAgent
from src.api.schemas import (
    InquiryRequest,
    InquiryResponse,
    EvidenceItem,
    HealthResponse,
    MetricsResponse
)
from src.monitoring.logger import OperationalTelemetryLogger


# Instantiate FastAPI Application
app = FastAPI(
    title="Hiver Twitter AI Customer Support API",
    description="Production-grade AI Support Agent API with RAG retrieval, escalation policy triage, and audit telemetry.",
    version="1.0.0"
)

# Configure CORS Middleware (Allow local development dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Documented for local presentation demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Agent & Telemetry Logger Instances
agent: TwitterSupportAgent = None
telemetry_logger = OperationalTelemetryLogger(log_file_path=str(BASE_DIR / "logs" / "agent_operations.jsonl"))


def get_agent() -> TwitterSupportAgent:
    """Returns singleton TwitterSupportAgent instance, loading from models/ if needed."""
    global agent
    if agent is None:
        models_dir = BASE_DIR / "models"
        try:
            agent = TwitterSupportAgent.load_from_models_dir(models_dir=models_dir, mock_llm=True)
            print(f"[FastAPI Server] TwitterSupportAgent loaded successfully from {models_dir}.")
        except Exception as e:
            print(f"[FastAPI Server Warning] Could not load from {models_dir}: {e}")
            agent = TwitterSupportAgent()
    return agent


@app.on_event("startup")
def startup_event():
    """Initializes models and agent on server startup."""
    get_agent()


@app.get("/api/v1/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    Health check endpoint.
    Reports classifier status, FAISS index items (35,000), generator provider, and live LLM status.
    """
    agent_inst = get_agent()
    classifier_loaded = agent_inst.classifier is not None if agent_inst else False
    retriever_loaded = agent_inst.retriever is not None if agent_inst else False
    index_items = agent_inst.retriever.index.ntotal if (agent_inst and agent_inst.retriever and agent_inst.retriever.index) else 0
    provider_mode = "mock"

    return HealthResponse(
        status="healthy" if classifier_loaded and retriever_loaded else "degraded",
        classifier_loaded=classifier_loaded,
        retriever_loaded=retriever_loaded,
        index_items=index_items,
        generator_provider=provider_mode,
        live_llm_available=False  # Explicitly documents live LLM benchmark testing as pending
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """
    Returns runtime operational telemetry summary.
    Exposes runtime ticket counts, auto vs escalate ratio, and escalation reasons.
    """
    metrics = telemetry_logger.get_metrics_summary()
    return MetricsResponse(**metrics)


@app.post("/api/v1/inquire", response_model=InquiryResponse)
def process_inquiry(payload: InquiryRequest) -> InquiryResponse:
    """
    Processes a customer Twitter inquiry through the TwitterSupportAgent pipeline:
    1. Intent Classification
    2. Top-3 RAG FAISS Vector Retrieval
    3. Multi-Factor Escalation Policy Triage
    4. Response Generation (if AUTO) or Pre-Generation Halt (if ESCALATE)
    5. Telemetry JSONL Audit Logging
    """
    agent_inst = get_agent()
    if not agent_inst:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent model pipeline not initialized."
        )

    customer_msg = payload.customer_message.strip()
    if not customer_msg:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="customer_message must not be empty or whitespace."
        )

    # Process inquiry end-to-end
    result = agent_inst.process_inquiry(
        customer_message=customer_msg,
        interaction_id=payload.interaction_id
    )

    # Log operational telemetry
    telemetry_logger.log_inquiry(result, provider_mode="mock")

    # Format retrieved evidence items
    retrieved_docs = agent_inst.retriever.retrieve(customer_msg, k=3) if agent_inst.retriever else []
    evidence_items = [
        EvidenceItem(
            interaction_id=doc["interaction_id"],
            predicted_intent=doc.get("predicted_intent", result["predicted_intent"]),
            brand_response=doc["brand_response"],
            similarity_score=float(doc["similarity_score"])
        )
        for doc in retrieved_docs
    ]

    return InquiryResponse(
        interaction_id=result["interaction_id"],
        customer_message=result["customer_message"],
        predicted_intent=result["predicted_intent"],
        intent_confidence=float(result["intent_confidence"]),
        retrieval_top_k=int(result["retrieval_top_k"]),
        retrieval_similarity=float(result["retrieval_similarity"]),
        retrieved_evidence=evidence_items,
        requires_human_escalation=bool(result["requires_human_escalation"]),
        escalation_reason=str(result["escalation_reason"]),
        decision=result["decision"],
        generated_response=result.get("generated_response"),
        elapsed_seconds=float(result["elapsed_seconds"]),
        timestamp=result["timestamp"]
    )
