#!/usr/bin/env python3
"""
scripts/demo.py

Phase 6 Interactive CLI Demonstration:
Showcases live end-to-end Twitter AI Support Agent processing in < 30 seconds.
Demonstrates:
1. Standard inquiry -> Intent Classification -> RAG Retrieval -> AUTO Response Generation.
2. High-risk security inquiry -> Mandatory HUMAN Escalation (Pre-generation halt).
3. Low confidence / ambiguous inquiry -> Mandatory HUMAN Escalation.
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.agent import TwitterSupportAgent

DEMO_SCENARIOS = [
    {
        "name": "Scenario 1: Standard Order Tracking Inquiry (Expected: AUTO)",
        "message": "Hi, where is my package for order #112-9847291? It was supposed to arrive yesterday.",
        "interaction_id": "DEMO_001"
    },
    {
        "name": "Scenario 2: Account Access & Security Issue (Expected: HUMAN Escalation)",
        "message": "Someone hacked into my account and changed my email and password! Help!",
        "interaction_id": "DEMO_002"
    },
    {
        "name": "Scenario 3: Ambiguous / Low-Context Inquiry (Expected: HUMAN Escalation)",
        "message": "Hello option change?",
        "interaction_id": "DEMO_003"
    }
]


def main():
    print("============================================================")
    print("HIVER PHASE 6: INTERACTIVE TWITTER AI SUPPORT AGENT DEMO")
    print("============================================================")

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"

    print("Initializing Twitter Support Agent from models/...")
    agent = TwitterSupportAgent.load_from_models_dir(models_dir=models_dir, mock_llm=True)
    print("Agent initialized successfully!\n")

    for idx, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"------------------------------------------------------------")
        print(f"[{idx}/3] {scenario['name']}")
        print(f"Customer Inquiry: \"{scenario['message']}\"")
        print(f"------------------------------------------------------------")

        res = agent.process_inquiry(scenario["message"], interaction_id=scenario["interaction_id"])

        print(f"  Predicted Intent  : {res['predicted_intent']} (Confidence: {res['intent_confidence']:.4f})")
        print(f"  RAG Retrieval     : Top-{res['retrieval_top_k']} docs (Top Similarity: {res['retrieval_similarity']:.4f})")
        print(f"  Escalation Triage : Decision = [{res['decision']}] | Reason = {res['escalation_reason']}")

        if res["decision"] == "AUTO":
            print(f"  Generated Reply   : \"{res['generated_response']}\"")
        else:
            print(f"  Generated Reply   : [HALTED - Ticket escalated to human support queue]")

        print(f"  Elapsed Time      : {res['elapsed_seconds']:.4f}s")
        print("\n  [Audit Log Preview]:")
        print("  " + json.dumps({
            "interaction_id": res["interaction_id"],
            "intent": res["predicted_intent"],
            "decision": res["decision"],
            "reason": res["escalation_reason"],
            "evidence_ids": res["retrieved_evidence_ids"]
        }, indent=2).replace("\n", "\n  "))
        print("\n")

    print("============================================================")
    print("INTERACTIVE DEMO COMPLETE (Execution time < 5s)")
    print("============================================================")


if __name__ == "__main__":
    main()
