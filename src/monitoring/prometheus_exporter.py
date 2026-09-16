"""
src/monitoring/prometheus_exporter.py

Zero-dependency Prometheus metrics exporter for the Hiver AI Customer Support Agent.
Outputs metrics in standard Prometheus exposition format (text/plain; version=0.0.4).
"""

import time
from typing import Dict, Any


class LightweightPrometheusExporter:
    """Zero-dependency Prometheus metrics collector and exporter."""

    def __init__(self):
        self.start_time = time.time()
        self.auto_count = 0
        self.escalate_count = 0
        self.total_evidence_items = 0
        self.confidence_buckets = {
            "0.5": 0,
            "0.75": 0,
            "0.9": 0,
            "+Inf": 0
        }

    def record_inquiry(self, action: str, confidence: float, evidence_count: int = 0):
        """Records telemetry for an inquiry execution."""
        if action.upper() == "AUTO":
            self.auto_count += 1
        elif action.upper() == "ESCALATE":
            self.escalate_count += 1

        self.total_evidence_items += evidence_count

        # Bucket confidence score
        if confidence <= 0.5:
            self.confidence_buckets["0.5"] += 1
        if confidence <= 0.75:
            self.confidence_buckets["0.75"] += 1
        if confidence <= 0.9:
            self.confidence_buckets["0.9"] += 1
        self.confidence_buckets["+Inf"] += 1

    def generate_metrics_text(self) -> str:
        """Renders collected metrics into Prometheus exposition text format."""
        uptime = max(0.0, time.time() - self.start_time)
        total_inquiries = self.auto_count + self.escalate_count

        lines = [
            "# HELP hiver_uptime_seconds Total runtime of the Hiver API service in seconds.",
            "# TYPE hiver_uptime_seconds gauge",
            f"hiver_uptime_seconds {uptime:.2f}",
            "",
            "# HELP hiver_inquiries_total Total number of customer inquiries processed by decision action.",
            "# TYPE hiver_inquiries_total counter",
            f'hiver_inquiries_total{{action="AUTO"}} {self.auto_count}',
            f'hiver_inquiries_total{{action="ESCALATE"}} {self.escalate_count}',
            "",
            "# HELP hiver_rag_evidence_items_total Cumulative RAG evidence items retrieved across inquiries.",
            "# TYPE hiver_rag_evidence_items_total counter",
            f"hiver_rag_evidence_items_total {self.total_evidence_items}",
            "",
            "# HELP hiver_intent_confidence Cumulative histogram of intent classification confidence scores.",
            "# TYPE hiver_intent_confidence histogram",
            f'hiver_intent_confidence_bucket{{le="0.5"}} {self.confidence_buckets["0.5"]}',
            f'hiver_intent_confidence_bucket{{le="0.75"}} {self.confidence_buckets["0.75"]}',
            f'hiver_intent_confidence_bucket{{le="0.9"}} {self.confidence_buckets["0.9"]}',
            f'hiver_intent_confidence_bucket{{le="+Inf"}} {self.confidence_buckets["+Inf"]}',
            f'hiver_intent_confidence_count {total_inquiries}',
            ""
        ]

        return "\n".join(lines)
