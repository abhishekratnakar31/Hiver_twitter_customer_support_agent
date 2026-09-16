"""
src/api/schemas.py

Pydantic schemas for Hiver Twitter AI Support Agent FastAPI REST endpoints.
Defines explicit request/response contracts with strict type validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal


class InquiryRequest(BaseModel):
    """Request payload for /api/v1/inquire endpoint."""
    customer_message: str = Field(..., min_length=1, description="Customer Twitter inquiry text")
    interaction_id: Optional[str] = Field(None, description="Optional unique interaction identifier")


class EvidenceItem(BaseModel):
    """Single RAG evidence item retrieved from vector index."""
    interaction_id: str
    predicted_intent: Optional[str] = "historical_case"
    brand_response: str
    similarity_score: float


class InquiryResponse(BaseModel):
    """Response payload for /api/v1/inquire endpoint."""
    interaction_id: str
    customer_message: str
    predicted_intent: str
    intent_confidence: float
    retrieval_top_k: int
    retrieval_similarity: float
    retrieved_evidence: List[EvidenceItem]
    requires_human_escalation: bool
    escalation_reason: str
    decision: Literal["AUTO", "ESCALATE"]
    generated_response: Optional[str] = None
    elapsed_seconds: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response payload for /api/v1/health status endpoint."""
    status: str
    classifier_loaded: bool
    retriever_loaded: bool
    index_items: int
    generator_provider: str
    live_llm_available: bool


class MetricsResponse(BaseModel):
    """Response payload for /api/v1/metrics runtime telemetry endpoint."""
    total_inquiries: int
    auto_handled: int
    escalated: int
    escalation_rate: float
    mean_intent_confidence: float
    escalation_reasons: Dict[str, int]
