"""
tests/test_golden_set.py

Automated test suite asserting data provenance, dual zero leakage, exact partition
reconstruction, historical reply integrity, escalation triage logic, conversation uniqueness,
and schema completeness for Phase 4 Golden Evaluation Set artifacts.
"""

import pytest
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
VAL_PATH = BASE_DIR / "data" / "processed" / "validation.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"
TEST_POOL_PATH = BASE_DIR / "data" / "processed" / "test_pool.csv"

TEMPLATE_PATH = BASE_DIR / "data" / "golden" / "annotation_template.csv"
GOLDEN_CSV_PATH = BASE_DIR / "data" / "golden" / "golden_set.csv"
REPORT_PATH = BASE_DIR / "reports" / "golden_set_profile.md"

VALID_ESCALATION_REASONS = {
    "high_risk",
    "ambiguous_request",
    "insufficient_context",
    "sensitive_account_issue",
    "potential_policy_exception",
    "complex_unresolved_issue",
    "none"
}

VALID_RISK_PRIORS = {"low", "medium", "high"}

def test_files_exist():
    assert TEMPLATE_PATH.exists(), f"Annotation template {TEMPLATE_PATH} missing."
    assert GOLDEN_CSV_PATH.exists(), f"Golden set CSV {GOLDEN_CSV_PATH} missing."
    assert TEST_POOL_PATH.exists(), f"Test pool CSV {TEST_POOL_PATH} missing."
    assert REPORT_PATH.exists(), f"Profile report {REPORT_PATH} missing."

def test_dataset_sizes():
    gold_df = pd.read_csv(GOLDEN_CSV_PATH)
    pool_df = pd.read_csv(TEST_POOL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    assert len(gold_df) == 200, f"Expected 200 golden examples, got {len(gold_df)}"
    assert len(test_df) == 7500, f"Expected 7,500 total test items, got {len(test_df)}"
    assert len(pool_df) <= 7300, f"Expected <= 7,300 test pool items, got {len(pool_df)}"

def test_dual_leakage_and_partition_reconstruction():
    gold_df = pd.read_csv(GOLDEN_CSV_PATH)
    pool_df = pd.read_csv(TEST_POOL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    
    gold_ints = set(gold_df["interaction_id"])
    pool_ints = set(pool_df["interaction_id"])
    test_ints = set(test_df["interaction_id"])
    train_ints = set(train_df["interaction_id"])
    val_ints = set(val_df["interaction_id"])
    
    gold_convs = set(gold_df["conversation_id"])
    pool_convs = set(pool_df["conversation_id"])
    train_convs = set(train_df["conversation_id"])
    val_convs = set(val_df["conversation_id"])
    
    # 1. Zero leakage assertions across splits
    assert gold_ints.isdisjoint(train_ints), "Leakage: Golden interaction_id in train.csv!"
    assert gold_ints.isdisjoint(val_ints), "Leakage: Golden interaction_id in validation.csv!"
    assert gold_convs.isdisjoint(train_convs), "Leakage: Golden conversation_id in train.csv!"
    assert gold_convs.isdisjoint(val_convs), "Leakage: Golden conversation_id in validation.csv!"
    
    # 2. Golden vs Test Pool separation (both interaction_id and conversation_id non-splitting)
    assert gold_ints.isdisjoint(pool_ints), "Leakage: Golden interaction_id in test_pool.csv!"
    assert gold_convs.isdisjoint(pool_convs), "Leakage: Golden conversation_id split into test_pool.csv!"
    
    # 3. Exact partition reconstruction assertion
    all_golden_conv_ints = set(test_df[test_df["conversation_id"].isin(gold_convs)]["interaction_id"])
    reconstructed_ints = all_golden_conv_ints.union(pool_ints)
    assert reconstructed_ints == test_ints, "Partition mismatch: Golden ∪ Test Pool != Original Test!"

def test_conversation_uniqueness():
    gold_df = pd.read_csv(GOLDEN_CSV_PATH)
    assert gold_df["conversation_id"].nunique() == 200, "Golden set must contain 200 unique conversation_ids."
    assert gold_df["example_id"].nunique() == 200, "Golden set example_ids must be unique."

def test_historical_reply_integrity():
    gold_df = pd.read_csv(GOLDEN_CSV_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    test_reply_pairs = set(zip(test_df["interaction_id"], test_df["brand_response"]))
    
    for idx, row in gold_df.iterrows():
        int_id = row["interaction_id"]
        hist_reply = str(row["historical_reference_reply"])
        assert (int_id, hist_reply) in test_reply_pairs, \
            f"Historical reply integrity check failed for {int_id}!"

def test_escalation_enum_and_logic():
    gold_df = pd.read_csv(GOLDEN_CSV_PATH)
    
    for idx, row in gold_df.iterrows():
        esc = bool(row["requires_human_escalation"])
        reason = str(row["escalation_reason"])
        
        assert reason in VALID_ESCALATION_REASONS, f"Invalid escalation_reason '{reason}' at row {idx}"
        
        if not esc:
            assert reason == "none", f"Logic failure: requires_human_escalation=False but escalation_reason='{reason}'"
        else:
            assert reason != "none", f"Logic failure: requires_human_escalation=True but escalation_reason='none'"

def test_schema_completeness_and_versioning():
    gold_df = pd.read_csv(GOLDEN_CSV_PATH)
    
    required_cols = [
        "example_id",
        "conversation_id",
        "interaction_id",
        "customer_message",
        "ground_truth_intent",
        "historical_reference_reply",
        "is_multi_turn",
        "message_position",
        "risk_tier",
        "requires_human_escalation",
        "escalation_reason",
        "annotation_notes",
        "annotation_version"
    ]
    
    for col in required_cols:
        assert col in gold_df.columns, f"Missing column {col} in golden_set.csv"
        assert gold_df[col].isna().sum() == 0, f"Null values found in column {col}"
        
    for idx, row in gold_df.iterrows():
        assert str(row["risk_tier"]) in VALID_RISK_PRIORS, f"Invalid risk_tier in row {idx}"
        assert str(row["annotation_version"]) == "phase4_v1", f"Invalid annotation_version in row {idx}"
        
        # Check is_multi_turn logic
        has_ctx = pd.notna(row["conversation_context"]) and str(row["conversation_context"]).strip() != ""
        assert bool(row["is_multi_turn"]) == has_ctx, f"is_multi_turn mismatch at row {idx}"
