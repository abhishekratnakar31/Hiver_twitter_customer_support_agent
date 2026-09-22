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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress Python 3.14 resource tracker process exit on macOS
try:
    import multiprocessing.resource_tracker
    multiprocessing.resource_tracker._resource_tracker.main = lambda *args, **kwargs: None
except Exception:
    pass

from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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
from src.monitoring.prometheus_exporter import LightweightPrometheusExporter



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

# Mount Dashboard Static Files
web_dir = BASE_DIR / "web"
if web_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(web_dir), html=True), name="dashboard")

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/dashboard")

# Global Agent & Telemetry Logger Instances
agent: TwitterSupportAgent = None
telemetry_logger = OperationalTelemetryLogger(log_file_path=str(BASE_DIR / "logs" / "agent_operations.jsonl"))
prometheus_exporter = LightweightPrometheusExporter()


def get_agent() -> TwitterSupportAgent:
    """Returns singleton TwitterSupportAgent instance, enabling real LLM generation if API keys are present."""
    global agent
    if agent is None:
        models_dir = BASE_DIR / "models"
        has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        force_mock = os.environ.get("MOCK_LLM", "0") == "1"
        use_mock = force_mock or (not has_api_key)

        agent = TwitterSupportAgent.load_from_models_dir(models_dir=models_dir, mock_llm=use_mock)
        mode_str = "Mock Mode" if use_mock else "LIVE LLM API Mode"
        print(f"[FastAPI Server] TwitterSupportAgent loaded successfully ({mode_str}) from {models_dir}.")
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

    has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    is_mock = agent_inst.generator.mock_mode if (agent_inst and agent_inst.generator) else True
    provider_mode = "mock" if is_mock else ("gemini" if os.environ.get("GEMINI_API_KEY") else "openai")

    return HealthResponse(
        status="healthy" if classifier_loaded and retriever_loaded else "degraded",
        classifier_loaded=classifier_loaded,
        retriever_loaded=retriever_loaded,
        index_items=index_items,
        generator_provider=provider_mode,
        live_llm_available=has_api_key
    )



@app.get("/api/v1/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """
    Returns runtime operational telemetry summary.
    Exposes runtime ticket counts, auto vs escalate ratio, and escalation reasons.
    """
    metrics = telemetry_logger.get_metrics_summary()
    return MetricsResponse(**metrics)


@app.get("/metrics")
def get_prometheus_metrics():
    """
    Exposes plain-text Prometheus metrics endpoint.
    Format: text/plain; version=0.0.4
    """
    return Response(
        content=prometheus_exporter.generate_metrics_text(),
        media_type="text/plain; version=0.0.4"
    )


@app.post("/api/v1/inquire", response_model=InquiryResponse)
def process_inquiry(payload: InquiryRequest) -> InquiryResponse:
    """
    Processes a customer Twitter inquiry through the TwitterSupportAgent pipeline:
    1. Intent Classification
    2. Top-3 RAG FAISS Vector Retrieval
    3. Multi-Factor Escalation Policy Triage
    4. Response Generation (if AUTO) or Pre-Generation Halt (if ESCALATE)
    5. Telemetry JSONL Audit Logging & Prometheus Metric Recording
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
        interaction_id=payload.interaction_id,
        conversation_id=payload.conversation_id,
        conversation_history=payload.conversation_history
    )

    # Log operational telemetry
    telemetry_logger.log_inquiry(result, provider_mode="mock")

    # Record Prometheus metric
    prometheus_exporter.record_inquiry(
        action=result.get("decision", "ESCALATE"),
        confidence=float(result.get("intent_confidence", 0.0)),
        evidence_count=int(result.get("retrieval_top_k", 0))
    )


    # Format retrieved evidence items evaluated on query_context_used
    query_context = result.get("query_context_used", customer_msg)
    retrieved_docs = agent_inst.retriever.retrieve(query_context, k=3) if agent_inst.retriever else []
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
        conversation_id=result.get("conversation_id"),
        customer_message=result["customer_message"],
        query_context_used=result.get("query_context_used"),
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
