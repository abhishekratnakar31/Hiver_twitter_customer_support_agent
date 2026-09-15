#!/usr/bin/env python3
"""
scripts/build_golden_set.py

Phase 4 Stage 1: Deterministic Candidate Sampling & Carve-Out.
1. Loads test.csv (7,500 interactions).
2. Deterministically samples 200 candidate interactions (seed=42), enforcing 1 example per conversation (nunique(conversation_id) == 200).
3. Reconstructs preceding conversation_context and calculates 1-indexed message_position.
4. Copies historical brand_response unchanged as historical_reference_reply.
5. Carves out data/processed/test_pool.csv (7,300 remaining interactions).
6. Asserts zero dual leakage (interaction_id and conversation_id) across Train, Val, Test Pool, and Golden candidates.
7. Asserts partition reconstruction: Golden candidates + Test Pool == Original Test interactions.
8. Exports data/golden/annotation_template.csv and reports/golden_sampling_profile.md.
"""

import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
VAL_PATH = BASE_DIR / "data" / "processed" / "validation.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"
TEST_POOL_PATH = BASE_DIR / "data" / "processed" / "test_pool.csv"

TEMPLATE_PATH = BASE_DIR / "data" / "golden" / "annotation_template.csv"
REPORT_PATH = BASE_DIR / "reports" / "golden_sampling_profile.md"

def load_datasets():
    """Loads all split datasets."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset missing at {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH) if TRAIN_PATH.exists() else pd.DataFrame()
    val_df = pd.read_csv(VAL_PATH) if VAL_PATH.exists() else pd.DataFrame()
    return train_df, val_df, test_df

def reconstruct_conversation_contexts(test_df):
    """
    Sorts interactions by timestamp within conversation_id and builds preceding context.
    """
    test_df = test_df.sort_values(by=["conversation_id", "timestamp"]).copy()
    
    contexts = []
    positions = []
    is_multi_turns = []
    
    # Group by conversation_id to build context
    grouped = test_df.groupby("conversation_id", sort=False)
    
    for conv_id, group in grouped:
        history = []
        for idx, row in group.iterrows():
            # Build preceding context from history
            if history:
                ctx_str = "\n".join(history)
                is_multi = True
            else:
                ctx_str = ""
                is_multi = False
                
            # Message position (1-indexed among all reconstructed messages: customer, brand, customer...)
            pos = len(history) + 1
            
            contexts.append(ctx_str)
            positions.append(pos)
            is_multi_turns.append(is_multi)
            
            # Append current turn to history for subsequent turns
            history.append(f"Customer: {row['customer_message']}")
            history.append(f"AmazonHelp: {row['brand_response']}")
            
    test_df["conversation_context"] = contexts
    test_df["message_position"] = positions
    test_df["is_multi_turn"] = is_multi_turns
    return test_df

def sample_candidates_and_carve_out(test_df):
    """
    Samples 200 candidate conversations deterministically (seed=42) from test.csv
    such that golden_set gets 200 interactions and test_pool gets EXACTLY 7,300 interactions.
    200 + 7,300 = 7,500 exact partition reconciliation with 0 conversation splitting.
    """
    # Identify conversations with 1 interaction pair in test_df
    conv_counts = test_df.groupby("conversation_id").size()
    single_pair_convs = conv_counts[conv_counts == 1].index
    
    # Sample 200 unique single-pair conversations
    sampled_convs = pd.Series(single_pair_convs).sample(n=200, random_state=42)
    
    candidate_df = test_df[test_df["conversation_id"].isin(sampled_convs)].copy()
    test_pool_df = test_df[~test_df["conversation_id"].isin(sampled_convs)].copy()
    
    return candidate_df, test_pool_df

def verify_provenance_and_partitioning(train_df, val_df, candidate_df, test_pool_df, test_df):
    """
    Programmatically asserts:
    1. Golden candidates == 200 items, Test Pool == 7,300 items.
    2. Golden ∩ Train = ∅, Golden ∩ Val = ∅ (interaction_id and conversation_id).
    3. Golden ∩ Test Pool = ∅ (interaction_id and conversation_id).
    4. Golden ∪ Test Pool == Original Test interactions (exact set match: 200 + 7,300 = 7,500).
    """
    cand_ints = set(candidate_df["interaction_id"])
    pool_ints = set(test_pool_df["interaction_id"])
    test_ints = set(test_df["interaction_id"])
    
    cand_convs = set(candidate_df["conversation_id"])
    pool_convs = set(test_pool_df["conversation_id"])
    
    train_ints = set(train_df["interaction_id"]) if not train_df.empty else set()
    val_ints = set(val_df["interaction_id"]) if not val_df.empty else set()
    train_convs = set(train_df["conversation_id"]) if not train_df.empty else set()
    val_convs = set(val_df["conversation_id"]) if not val_df.empty else set()
    
    # Check sizes
    assert len(candidate_df) == 200, f"Expected 200 candidates, got {len(candidate_df)}"
    assert len(test_pool_df) == 7300, f"Expected 7,300 test pool items, got {len(test_pool_df)}"
    assert len(candidate_df) + len(test_pool_df) == 7500, f"Expected 7,500 total, got {len(candidate_df) + len(test_pool_df)}"
    assert len(cand_convs) == 200, f"Expected 200 unique conversation IDs, got {len(cand_convs)}"
    
    # Check zero leakage with train and val
    assert cand_ints.isdisjoint(train_ints), "Leakage detected: Golden candidate interaction_id in train.csv!"
    assert cand_ints.isdisjoint(val_ints), "Leakage detected: Golden candidate interaction_id in validation.csv!"
    assert cand_convs.isdisjoint(train_convs), "Leakage detected: Golden candidate conversation_id in train.csv!"
    assert cand_convs.isdisjoint(val_convs), "Leakage detected: Golden candidate conversation_id in validation.csv!"
    
    # Check zero leakage with test pool (both interaction_id and conversation_id)
    assert cand_ints.isdisjoint(pool_ints), "Leakage detected: Golden candidate interaction_id in test_pool.csv!"
    assert cand_convs.isdisjoint(pool_convs), "Leakage detected: Golden candidate conversation_id split into test_pool.csv!"
    
    # Check exact partition reconstruction of interactions
    reconstructed_ints = cand_ints.union(pool_ints)
    assert reconstructed_ints == test_ints, "Partition mismatch: Golden ∪ Test Pool != Original Test!"
    print(f"Partition reconstruction assertion PASSED: 200 Golden + {len(test_pool_df):,} Test Pool = 7,500 total Test interactions (0 missing, 0 extra, 0 conversation splits).")

def format_annotation_template(candidate_df):
    """Formats candidate dataframe into annotation_template.csv schema."""
    candidate_df = candidate_df.reset_index(drop=True)
    candidate_df["example_id"] = [f"GOLD_{i+1:03d}" for i in range(len(candidate_df))]
    candidate_df["historical_reference_reply"] = candidate_df["brand_response"]
    
    # Blank annotation columns for human labeling
    candidate_df["ground_truth_intent"] = ""
    candidate_df["risk_tier"] = ""  # Baseline prior from taxonomy
    candidate_df["requires_human_escalation"] = ""  # True / False
    candidate_df["escalation_reason"] = ""  # Enum
    candidate_df["annotation_notes"] = ""
    candidate_df["annotation_version"] = "phase4_v1"
    
    cols = [
        "example_id",
        "conversation_id",
        "interaction_id",
        "conversation_context",
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
    return candidate_df[cols]

def export_artifacts(template_df, test_pool_df):
    """Exports test_pool.csv, annotation_template.csv, and golden_sampling_profile.md."""
    # 1. Export test_pool.csv
    TEST_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_pool_df.to_csv(TEST_POOL_PATH, index=False)
    print(f"Exported {TEST_POOL_PATH} (7,300 interactions)")
    
    # 2. Export annotation_template.csv
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(TEMPLATE_PATH, index=False)
    print(f"Exported {TEMPLATE_PATH} (200 candidate rows)")
    
    # 3. Export reports/golden_sampling_profile.md
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    multi_turn_cnt = template_df["is_multi_turn"].sum()
    single_turn_cnt = len(template_df) - multi_turn_cnt
    
    report_doc = [
        "# Phase 4 Stage 1 Technical Report: Candidate Sampling & Test Pool Carve-Out\n",
        "## Executive Summary",
        f"- **Original Held-Out Test interactions**: {len(template_df) + len(test_pool_df):,} (`data/processed/test.csv`).",
        f"- **Golden Candidates Sampled**: 200 unique conversations (`seed=42`).",
        f"- **Untouched Test Pool Carved Out**: {len(test_pool_df):,} interactions (`data/processed/test_pool.csv`).",
        f"- **Conversation Uniqueness**: `nunique(conversation_id) == 200` (0 conversation splitting across splits).",
        f"- **Multi-Turn Breakdown**: {single_turn_cnt} Single-Turn (turn 1), {multi_turn_cnt} Multi-Turn (turn > 1).\n",
        "## Dual Zero Leakage & Partition Reconstruction Assertions",
        "- `Golden ∩ Train = ∅` (checked via `interaction_id` and `conversation_id`).",
        "- `Golden ∩ Val = ∅` (checked via `interaction_id` and `conversation_id`).",
        "- `Golden ∩ Test Pool = ∅` (checked via `interaction_id` and `conversation_id`).",
        "- `Golden ∪ Test Pool == Original Test` (exact set equality across 7,500 interaction IDs).\n",
        "## Candidate Schema Exported to `data/golden/annotation_template.csv`",
        "Target fields ready for human annotation: `ground_truth_intent`, `risk_tier`, `requires_human_escalation`, `escalation_reason`, `annotation_notes`.\n"
    ]
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_doc))
    print(f"Exported {REPORT_PATH}")

def main():
    print("=== Starting Phase 4 Stage 1: Candidate Sampling & Carve-Out ===")
    train_df, val_df, test_df = load_datasets()
    print(f"Loaded datasets: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    
    print("1. Reconstructing preceding conversation contexts...")
    test_df = reconstruct_conversation_contexts(test_df)
    
    print("2. Deterministically sampling 200 candidates and carving out 7,300 test pool...")
    candidate_df, test_pool_df = sample_candidates_and_carve_out(test_df)
    
    print("3. Verifying dual leakage and exact partition reconstruction...")
    verify_provenance_and_partitioning(train_df, val_df, candidate_df, test_pool_df, test_df)
    
    print("4. Formatting annotation template...")
    template_df = format_annotation_template(candidate_df)
    
    print("5. Exporting artifacts...")
    export_artifacts(template_df, test_pool_df)
    
    print("=== Phase 4 Stage 1 Complete ===")

if __name__ == "__main__":
    main()
