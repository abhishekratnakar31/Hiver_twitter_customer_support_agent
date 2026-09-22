"""
src/models/escalation_policy.py

Phase 5 Baseline Escalation Triage Policy:
Enforces a strict 6-step deterministic precedence order:
1. High-Risk Intent: intent == "account_security_access" -> HUMAN ("high_risk")
2. Sensitive / Policy-Exception Rule:
   - payment_billing_issues -> HUMAN ("sensitive_account_issue")
   - package_missing_damaged -> HUMAN ("potential_policy_exception")
   - feedback_general_complaint -> HUMAN ("complex_unresolved_issue")
3. Reserve Bucket Intent: intent == "other_unclear" -> HUMAN ("ambiguous_request")
4. Low Classifier Confidence: intent_confidence < tau_conf (fixed starting heuristic tau_conf = 0.60) -> HUMAN ("insufficient_context")
5. Low Retrieval Similarity: retrieval_similarity < tau_sim (fixed starting heuristic tau_sim = 0.50) -> HUMAN ("insufficient_context")
6. Otherwise: -> AUTO ("none")

Outputs exactly one deterministic escalation_reason enum matching the frozen Phase 4 enum.
Fixed starting thresholds (tau_conf = 0.60, tau_sim = 0.50) are NOT tuned on the Golden Set.
"""

from typing import Dict, Any, List

VALID_ESCALATION_REASONS = {
    "high_risk",
    "ambiguous_request",
    "insufficient_context",
    "sensitive_account_issue",
    "potential_policy_exception",
    "complex_unresolved_issue",
    "none"
}


class BaselineEscalationPolicy:
    """Multi-factor baseline escalation policy engine."""

    def __init__(self, tau_conf: float = 0.60, tau_sim: float = 0.50):
        self.tau_conf = tau_conf
        self.tau_sim = tau_sim

    def evaluate_triage(
        self,
        predicted_intent: str,
        intent_confidence: float = 1.0,
        retrieval_similarity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluates triage decision for a single customer query using deterministic precedence order.
        """
        intent = str(predicted_intent).strip()
        conf = float(intent_confidence)
        sim = float(retrieval_similarity)

        # Precedence Step 1: High-Risk Intent
        if intent == "account_security_access":
            return {
                "requires_human_escalation": True,
                "escalation_reason": "high_risk"
            }

        # Precedence Step 2: Sensitive / Policy-Exception Rule
        if intent == "payment_billing_issues":
            return {
                "requires_human_escalation": True,
                "escalation_reason": "sensitive_account_issue"
            }
        if intent == "package_missing_damaged":
            return {
                "requires_human_escalation": True,
                "escalation_reason": "potential_policy_exception"
            }
        if intent == "feedback_general_complaint":
            return {
                "requires_human_escalation": True,
                "escalation_reason": "complex_unresolved_issue"
            }

        # Precedence Step 3: Reserve Bucket Intent
        if intent == "other_unclear":
            return {
                "requires_human_escalation": True,
                "escalation_reason": "ambiguous_request"
            }

        # Precedence Step 4: Low Classifier Confidence
        if conf < self.tau_conf:
            return {
                "requires_human_escalation": True,
                "escalation_reason": "insufficient_context"
            }

        # Precedence Step 5: Low Retrieval Similarity
        if sim < self.tau_sim:
            return {
                "requires_human_escalation": True,
                "escalation_reason": "insufficient_context"
            }

        # Precedence Step 6: Otherwise -> AUTO
        return {
            "requires_human_escalation": False,
            "escalation_reason": "none"
        }

    def evaluate_triage_batch(
        self,
        predicted_intents: List[str],
        intent_confidences: List[float] = None,
        retrieval_similarities: List[float] = None
    ) -> List[Dict[str, Any]]:
        """Evaluates triage decision for a batch of queries."""
        n = len(predicted_intents)
        if intent_confidences is None:
            intent_confidences = [1.0] * n
        if retrieval_similarities is None:
            retrieval_similarities = [1.0] * n

        results = []
        for i in range(n):
            res = self.evaluate_triage(
                predicted_intent=predicted_intents[i],
                intent_confidence=intent_confidences[i],
                retrieval_similarity=retrieval_similarities[i]
            )
            results.append(res)
        return results
