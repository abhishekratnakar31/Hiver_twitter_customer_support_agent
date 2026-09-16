"""
src/monitoring/logger.py

Structured JSONL operational logger and runtime metrics collector for Hiver Twitter AI Support Agent.
Appends non-sensitive telemetry records to logs/agent_operations.jsonl and tracks in-memory metrics.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any


class OperationalTelemetryLogger:
    """
    Structured non-sensitive JSONL audit logger and runtime telemetry tracker.
    """

    def __init__(self, log_file_path: str = "logs/agent_operations.jsonl"):
        self.log_path = Path(log_file_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory runtime telemetry counters
        self.total_inquiries = 0
        self.auto_handled = 0
        self.escalated = 0
        self.total_confidence_sum = 0.0
        self.escalation_reasons: Dict[str, int] = {}

    def log_inquiry(self, inquiry_record: Dict[str, Any], provider_mode: str = "mock") -> Dict[str, Any]:
        """
        Logs a single processed inquiry record to JSONL and updates runtime telemetry counters.
        Excludes sensitive authentication tokens or credentials.
        """
        self.total_inquiries += 1
        decision = inquiry_record.get("decision", "AUTO")
        conf = float(inquiry_record.get("intent_confidence", 0.0))
        reason = str(inquiry_record.get("escalation_reason", "none"))

        self.total_confidence_sum += conf

        if decision == "AUTO":
            self.auto_handled += 1
        else:
            self.escalated += 1
            self.escalation_reasons[reason] = self.escalation_reasons.get(reason, 0) + 1

        # Build clean non-sensitive log record
        telemetry_record = {
            "interaction_id": inquiry_record.get("interaction_id"),
            "timestamp": inquiry_record.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            "decision": decision,
            "escalation_reason": reason,
            "predicted_intent": inquiry_record.get("predicted_intent"),
            "intent_confidence": conf,
            "retrieval_similarity": float(inquiry_record.get("retrieval_similarity", 0.0)),
            "latency_ms": round(float(inquiry_record.get("elapsed_seconds", 0.0)) * 1000.0, 2),
            "provider_mode": provider_mode
        }

        # Append to JSONL log file safely
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(telemetry_record) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write telemetry record to {self.log_path}: {e}")

        return telemetry_record

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Returns runtime operational telemetry summary.
        Separated explicitly from offline benchmark metrics (Golden Set F1).
        """
        total = self.total_inquiries
        esc_rate = round(self.escalated / total, 4) if total > 0 else 0.0
        mean_conf = round(self.total_confidence_sum / total, 4) if total > 0 else 0.0

        return {
            "total_inquiries": total,
            "auto_handled": self.auto_handled,
            "escalated": self.escalated,
            "escalation_rate": esc_rate,
            "mean_intent_confidence": mean_conf,
            "escalation_reasons": self.escalation_reasons
        }
