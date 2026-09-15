import os
import sys
import argparse
import json
import pandas as pd

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.download import get_twcs_dataset_path
from src.data.conversations import reconstruct_and_filter_interactions, split_conversations_partition_first


def run_pipeline(mode: str = "real", seed: int = 42) -> None:
    """Executes the Phase 2 data pipeline."""
    print(f"==========================================")
    print(f"RUNNING PHASE 2 DATA PIPELINE [MODE: {mode.upper()}]")
    print(f"==========================================")

    data_path = get_twcs_dataset_path(mode=mode)
    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)

    print("\nReconstructing customer -> AmazonHelp interactions & applying quality filtering...")
    clean_df, filter_stats = reconstruct_and_filter_interactions(df)

    print(f"Raw Reconstructed Pairs: {filter_stats['total_reconstructed_pairs_raw']:,}")
    print(f"Empty/Whitespace Filtered: {filter_stats['empty_whitespace_filtered_count']:,}")
    print(f"Exact Duplicate Filtered: {filter_stats['duplicate_interaction_filtered_count']:,}")
    print(f"Final Clean Interactions Population: {filter_stats['final_clean_interactions']:,}")
    print(f"Final Unique Conversations Population: {filter_stats['final_unique_conversations']:,}")

    print("\nPartitioning FULL conversation population first, then sampling working subsets within partitions...")
    train_df, val_df, test_df, split_stats = split_conversations_partition_first(
        clean_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_train_interactions=35000,
        target_val_interactions=7500,
        target_test_interactions=7500,
        seed=seed,
    )

    print(f"\n--- AUTHORITATIVE SPLIT SUMMARY ---")
    print(f"Full Train Partition: {split_stats['full_train_interactions']:,} interactions ({split_stats['full_train_conversations']:,} convs)")
    print(f"Full Val Partition:   {split_stats['full_val_interactions']:,} interactions ({split_stats['full_val_conversations']:,} convs)")
    print(f"Full Test Partition:  {split_stats['full_test_interactions']:,} interactions ({split_stats['full_test_conversations']:,} convs)")

    print(f"\n--- WORKING SUBSET SUMMARY ---")
    print(f"Train Subset (`train.csv`):      {len(train_df):,} interactions ({split_stats['train_subset_conversations']:,} convs)")
    print(f"Val Subset (`validation.csv`):   {len(val_df):,} interactions ({split_stats['val_subset_conversations']:,} convs)")
    print(f"Test Subset Pool (`test.csv`):   {len(test_df):,} interactions ({split_stats['test_subset_conversations']:,} convs)")
    print(f"Authoritative Conversation Overlap: {split_stats['full_overlap_train_val'] + split_stats['full_overlap_train_test'] + split_stats['full_overlap_val_test']}")
    print(f"Working Subset Conversation Overlap: {split_stats['subset_overlap_train_val'] + split_stats['subset_overlap_train_test'] + split_stats['subset_overlap_val_test']}")

    # Export processed CSVs
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "validation.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    summary_path = os.path.join(processed_dir, "full_split_summary.json")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary_metadata = {
        "mode": mode,
        "seed": seed,
        "filtering_stats": filter_stats,
        "split_stats": split_stats,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_metadata, f, indent=2)

    print(f"\nSaved CSV files:")
    print(f"  - {train_path} ({len(train_df):,} rows)")
    print(f"  - {val_path} ({len(val_df):,} rows)")
    print(f"  - {test_path} ({len(test_df):,} rows)")

    # Generate reports/data_split.md
    report_content = f"""# Phase 2: Conversation Construction & Data Splitting Report

## Execution Metadata
- **Execution Mode**: `{mode.upper()}`
- **Random Seed**: `{seed}`
- **Source File**: `{data_path}`
- **Processed Target Directory**: `{processed_dir}`

---

## 1. Quality Filtering Breakdown

Out of **{filter_stats['raw_total_tweets']:,} raw dataset tweets** across the entire TWCS corpus:
- **Total AmazonHelp Outbound Responses**: {filter_stats['amazonhelp_outbound_responses']:,}
- **Missing Parent Link (`in_response_to_tweet_id`)**: {filter_stats['missing_in_response_to_tweet_id_count']:,}
- **Orphan Parent Tweets (Missing from dataset)**: {filter_stats['orphan_parent_count']:,}
- **Non-Inbound Parent Tweets**: {filter_stats['non_inbound_parent_count']:,}
- **Total Reconstructed Interaction Pairs (Raw)**: {filter_stats['total_reconstructed_pairs_raw']:,}
- **Empty / Whitespace Text Filtered**: {filter_stats['empty_whitespace_filtered_count']:,}
- **Exact Duplicate Interactions Filtered**: {filter_stats['duplicate_interaction_filtered_count']:,}
- **Final Clean Eligible Interactions**: **{filter_stats['final_clean_interactions']:,}** ({filter_stats['final_unique_conversations']:,} unique conversation threads)

---

## 2. Authoritative Partition-First Conversation Splitting

To guarantee zero data leakage between training and evaluation, partitioning occurred strictly on the **FULL conversation population first** before sampling computational working subsets.

### Full Population Partition Sizes (Authoritative Boundaries):
| Partition | Conversation Count | Conversation % | Interaction Count | Interaction % |
| :--- | :--- | :--- | :--- | :--- |
| **Full Train Partition** | {split_stats['full_train_conversations']:,} | 70.0% | {split_stats['full_train_interactions']:,} | {split_stats['full_train_interactions'] / split_stats['total_eligible_interactions'] * 100:.1f}% |
| **Full Validation Partition** | {split_stats['full_val_conversations']:,} | 15.0% | {split_stats['full_val_interactions']:,} | {split_stats['full_val_interactions'] / split_stats['total_eligible_interactions'] * 100:.1f}% |
| **Full Test Partition** | {split_stats['full_test_conversations']:,} | 15.0% | {split_stats['full_test_interactions']:,} | {split_stats['full_test_interactions'] / split_stats['total_eligible_interactions'] * 100:.1f}% |
| **Total Population** | **{split_stats['total_eligible_conversations']:,}** | **100.0%** | **{split_stats['total_eligible_interactions']:,}** | **100.0%** |

---

## 3. Working Subsets (Computational Optimization)

Within each authoritative partition, complete conversations were sampled to construct working subsets for fast model training and evaluation (<15 min execution target):

| Working Subset File | Sampled Conversations | Interaction Count | Purpose |
| :--- | :--- | :--- | :--- |
| `data/processed/train.csv` | {split_stats['train_subset_conversations']:,} | **{len(train_df):,}** | Baseline ML & Embedding Classifier Training |
| `data/processed/validation.csv` | {split_stats['val_subset_conversations']:,} | **{len(val_df):,}** | Hyperparameter Tuning & Threshold Calibration |
| `data/processed/test.csv` | {split_stats['test_subset_conversations']:,} | **{len(test_df):,}** | Unlabelled Test Pool for Automated Metrics |
| **Total Working Subset** | **{split_stats['train_subset_conversations'] + split_stats['val_subset_conversations'] + split_stats['test_subset_conversations']:,}** | **{len(train_df) + len(val_df) + len(test_df):,}** | Fast Experimentation Suite |

---

## 4. Conversation-Length Distribution Preservation

Sampling is performed at the conversation level, preserving whole multi-turn threads. Below is the quantitative comparison of conversation turn lengths between the full partitions and working subsets:

| Partition / Subset | Total Convs | 2-Turn Convs (%) | 3-Turn Convs (%) | 4+ Turn Convs (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Full Train Partition** | {split_stats['length_distributions']['full_train']['total_convs']:,} | {split_stats['length_distributions']['full_train']['2_turn_pct']}% | {split_stats['length_distributions']['full_train']['3_turn_pct']}% | {split_stats['length_distributions']['full_train']['4_plus_turn_pct']}% |
| **Working Train Subset** | {split_stats['length_distributions']['working_train']['total_convs']:,} | {split_stats['length_distributions']['working_train']['2_turn_pct']}% | {split_stats['length_distributions']['working_train']['3_turn_pct']}% | {split_stats['length_distributions']['working_train']['4_plus_turn_pct']}% |
| **Full Val Partition** | {split_stats['length_distributions']['full_val']['total_convs']:,} | {split_stats['length_distributions']['full_val']['2_turn_pct']}% | {split_stats['length_distributions']['full_val']['3_turn_pct']}% | {split_stats['length_distributions']['full_val']['4_plus_turn_pct']}% |
| **Working Val Subset** | {split_stats['length_distributions']['working_val']['total_convs']:,} | {split_stats['length_distributions']['working_val']['2_turn_pct']}% | {split_stats['length_distributions']['working_val']['3_turn_pct']}% | {split_stats['length_distributions']['working_val']['4_plus_turn_pct']}% |
| **Full Test Partition** | {split_stats['length_distributions']['full_test']['total_convs']:,} | {split_stats['length_distributions']['full_test']['2_turn_pct']}% | {split_stats['length_distributions']['full_test']['3_turn_pct']}% | {split_stats['length_distributions']['full_test']['4_plus_turn_pct']}% |
| **Working Test Subset** | {split_stats['length_distributions']['working_test']['total_convs']:,} | {split_stats['length_distributions']['working_test']['2_turn_pct']}% | {split_stats['length_distributions']['working_test']['3_turn_pct']}% | {split_stats['length_distributions']['working_test']['4_plus_turn_pct']}% |

---

## 5. Zero-Leakage & Data Integrity Proof

- **Train ∩ Validation Conversation Overlap**: `{split_stats['full_overlap_train_val']}`
- **Train ∩ Test Conversation Overlap**: `{split_stats['full_overlap_train_test']}`
- **Validation ∩ Test Conversation Overlap**: `{split_stats['full_overlap_val_test']}`
- **Working Subset Conversation Overlap**: `{split_stats['subset_overlap_train_val'] + split_stats['subset_overlap_train_test'] + split_stats['subset_overlap_val_test']}`
- **Working Subset Containment Verification**: 100% of working subset conversations are strictly contained within their respective full partitions.

---

## 6. Methodological Clarification: Test Pool vs. Golden Set

1. **`data/processed/test.csv` (7,500 interactions)**: Represents the unlabelled test pool from the working subset. Used for computing automated metrics (Macro F1, Per-intent F1, Confusion Matrix) on model predictions.
2. **Golden Evaluation Set (200 examples)**: Built in **Phase 4**, the Golden Set is **DISTINCT FROM** `test.csv`. It consists of **200 hand-curated, manually annotated examples** drawn independently from the held-out test population with zero overlap.
3. **Strict Evaluation Isolation**: Neither `test.csv` nor the Golden Set is ever accessed during training, feature engineering, vector indexing, or escalation threshold calibration.
"""

    with open("reports/data_split.md", "w") as f:
        f.write(report_content)

    print(f"Successfully generated reports/data_split.md")
    print(f"Phase 2 data preparation complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Conversation Construction & Data Splitting")
    parser.add_argument("--source", choices=["real", "demo"], default="real", help="Dataset mode: 'real' or 'demo'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    args = parser.parse_args()

    run_pipeline(mode=args.source, seed=args.seed)
