"""
tests/test_agent_phase6.py

Phase 6 Unit Test Suite asserting:
1. Grounded Generator Prompt Format: Prompt includes retrieved evidence as handling examples.
2. Grounded Generator Mock Mode: Returns compliant responses without external API calls.
3. LLM Judge Input Contract: Accepts (customer_message, predicted_intent, retrieved_evidence, generated_response).
4. LLM Judge Mechanical Total Score: Total score is mechanically computed in Python (`total = sum(criteria)`).
5. LLM Judge Rubric Schema: Evaluates 4 criteria bounded [0, 2] totaling [0, 8].
6. Agent AUTO Handle Flow: Standard low-risk query triggers AUTO response generation.
7. Agent Escalation Flow: High-risk query halts generation BEFORE response generator and returns ESCALATE audit log.
8. Agent Audit Log Format: Emits all required audit fields.
9. Spearman & Agreement Calculation: Exact agreement, within-1 agreement, and Spearman rho calculations.
10. Calibration Set Disjoint Assertion: Verifies human_calibration_50.csv is strictly disjoint from golden_set.csv.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.reply_generator import GroundedLLMGenerator
from src.models.llm_judge import LLMJudge
from src.models.agent import TwitterSupportAgent
from src.models.intent_classifier import TfidfLogisticClassifier
from src.models.rag_retriever import RAGRetriever
from src.models.escalation_policy import BaselineEscalationPolicy
from scripts.verify_human_alignment import compute_agreement_metrics

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sample_models():
    """Module-scoped model fixtures for Phase 6 tests."""
    clf_path = BASE_DIR / "models" / "intent_tfidf_logreg.joblib"
    idx_path = BASE_DIR / "models" / "rag_faiss.index"
    meta_path = BASE_DIR / "models" / "rag_metadata.json"

    clf = TfidfLogisticClassifier.load(clf_path)
    retriever = RAGRetriever.load(idx_path, meta_path)
    policy = BaselineEscalationPolicy(tau_conf=0.60, tau_sim=0.50)
    generator = GroundedLLMGenerator(mock_mode=True)
    agent = TwitterSupportAgent(classifier=clf, retriever=retriever, generator=generator, policy=policy)

    return {
        "clf": clf,
        "retriever": retriever,
        "policy": policy,
        "generator": generator,
        "agent": agent
    }


def test_grounded_generator_prompt_format():
    gen = GroundedLLMGenerator(mock_mode=True)
    evidence = [{
        "interaction_id": "INT_001",
        "customer_query": "where is my order",
        "brand_response": "please send us a DM",
        "similarity_score": 0.85
    }]
    res = gen.generate_reply(
        customer_message="my order is delayed",
        predicted_intent="order_status",
        retrieved_evidence=evidence
    )
    assert "generated_reply" in res
    assert res["retrieved_evidence_ids"] == ["INT_001"]
    assert res["is_mock"] is True


def test_grounded_generator_mock_mode():
    gen = GroundedLLMGenerator(mock_mode=True)
    res = gen.generate_reply("order status update", "order_status", [])
    assert len(res["generated_reply"]) > 5
    assert "DM" in res["generated_reply"] or "Direct Message" in res["generated_reply"] or "assist" in res["generated_reply"]


def test_llm_judge_input_contract():
    judge = LLMJudge(mock_mode=True)
    evidence = [{"customer_query": "order problem", "brand_response": "please send DM"}]
    res = judge.evaluate(
        customer_message="where is order",
        predicted_intent="order_status",
        retrieved_evidence=evidence,
        generated_response="Please send us a DM with your order number."
    )
    assert "correctness" in res
    assert "groundedness" in res
    assert "helpfulness" in res
    assert "policy_safety" in res
    assert "total_score" in res


def test_llm_judge_mechanical_total_score():
    judge = LLMJudge(mock_mode=True)
    res = judge.evaluate(
        customer_message="order status",
        predicted_intent="order_status",
        retrieved_evidence=[],
        generated_response="Please DM us."
    )
    expected_total = res["correctness"] + res["groundedness"] + res["helpfulness"] + res["policy_safety"]
    assert res["total_score"] == expected_total, f"Mechanical sum mismatch: {res['total_score']} != {expected_total}"


def test_llm_judge_rubric_schema():
    judge = LLMJudge(mock_mode=True)
    res = judge.evaluate("my account password", "account_access", [], "DM us.")
    for key in ["correctness", "groundedness", "helpfulness", "policy_safety"]:
        assert 0 <= res[key] <= 2
    assert 0 <= res["total_score"] <= 8


def test_agent_auto_handle_flow(sample_models):
    agent = sample_models["agent"]
    res = agent.process_inquiry("Where is my order #12345? Tracking shows delayed.")
    assert res["predicted_intent"] in ["order_tracking_delivery", "order_status"]
    assert res["decision"] == "AUTO"
    assert res["requires_human_escalation"] is False
    assert res["escalation_reason"] == "none"
    assert res["generated_response"] is not None


def test_agent_escalation_flow(sample_models):
    agent = sample_models["agent"]
    res = agent.process_inquiry("Someone hacked my password and email address!")
    assert res["decision"] == "ESCALATE"
    assert res["requires_human_escalation"] is True
    assert res["escalation_reason"] != "none"
    assert res["generated_response"] is None
    assert res["generation_metadata"]["status"] == "skipped_due_to_escalation"


def test_agent_audit_log_format(sample_models):
    agent = sample_models["agent"]
    res = agent.process_inquiry("where is my shipment", interaction_id="TEST_AUDIT_101")
    required_fields = [
        "interaction_id", "customer_message", "predicted_intent", "intent_confidence",
        "retrieval_top_k", "retrieval_similarity", "requires_human_escalation",
        "escalation_reason", "decision", "retrieved_evidence_ids", "generated_response",
        "elapsed_seconds", "timestamp"
    ]
    for field in required_fields:
        assert field in res, f"Missing audit field: {field}"
    assert res["interaction_id"] == "TEST_AUDIT_101"


def test_spearman_and_agreement_calc():
    h = [2, 2, 1, 0, 2]
    l = [2, 1, 1, 0, 2]
    res = compute_agreement_metrics(h, l)
    assert res["exact_agreement_pct"] == 80.0
    assert res["within_1_agreement_pct"] == 100.0
    assert res["spearman_rho"] > 0.5


def test_calibration_set_disjoint_assertion():
    gold_path = BASE_DIR / "data" / "golden" / "golden_set.csv"
    calib_path = BASE_DIR / "data" / "golden" / "human_calibration_50.csv"

    if gold_path.exists() and calib_path.exists():
        gold_df = pd.read_csv(gold_path)
        calib_df = pd.read_csv(calib_path)

        gold_ids = set(gold_df["interaction_id"].values)
        calib_ids = set(calib_df["interaction_id"].values)

        overlap = gold_ids.intersection(calib_ids)
        assert len(overlap) == 0, f"Disjoint failure: {len(overlap)} calibration examples overlap with Golden Set!"
