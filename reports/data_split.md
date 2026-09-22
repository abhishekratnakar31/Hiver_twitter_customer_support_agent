# Phase 2: Conversation Construction & Data Splitting Report

## Execution Metadata
- **Execution Mode**: `DEMO`
- **Random Seed**: `42`
- **Source File**: `data/raw/twcs_demo.csv`
- **Processed Target Directory**: `/private/var/folders/43/skj2qk7x5rb85vwhdbt4qwhc0000gn/T/pytest-of-abhishekratnakar/pytest-2/test_run_pipeline_demo_mode0/processed`

---

## 1. Quality Filtering Breakdown

Out of **6 raw dataset tweets** across the entire TWCS corpus:
- **Total AmazonHelp Outbound Responses**: 2
- **Missing Parent Link (`in_response_to_tweet_id`)**: 0
- **Orphan Parent Tweets (Missing from dataset)**: 0
- **Non-Inbound Parent Tweets**: 0
- **Total Reconstructed Interaction Pairs (Raw)**: 2
- **Empty / Whitespace Text Filtered**: 0
- **Exact Duplicate Interactions Filtered**: 0
- **Final Clean Eligible Interactions**: **2** (2 unique conversation threads)

---

## 2. Authoritative Partition-First Conversation Splitting

To guarantee zero data leakage between training and evaluation, partitioning occurred strictly on the **FULL conversation population first** before sampling computational working subsets.

### Full Population Partition Sizes (Authoritative Boundaries):
| Partition | Conversation Count | Conversation % | Interaction Count | Interaction % |
| :--- | :--- | :--- | :--- | :--- |
| **Full Train Partition** | 1 | 70.0% | 1 | 50.0% |
| **Full Validation Partition** | 0 | 15.0% | 0 | 0.0% |
| **Full Test Partition** | 1 | 15.0% | 1 | 50.0% |
| **Total Population** | **2** | **100.0%** | **2** | **100.0%** |

---

## 3. Working Subsets (Computational Optimization)

Within each authoritative partition, complete conversations were sampled to construct working subsets for fast model training and evaluation (<15 min execution target):

| Working Subset File | Sampled Conversations | Interaction Count | Purpose |
| :--- | :--- | :--- | :--- |
| `data/processed/train.csv` | 1 | **1** | Baseline ML & Embedding Classifier Training |
| `data/processed/validation.csv` | 0 | **0** | Hyperparameter Tuning & Threshold Calibration |
| `data/processed/test.csv` | 1 | **1** | Unlabelled Test Pool for Automated Metrics |
| **Total Working Subset** | **2** | **2** | Fast Experimentation Suite |

---

## 4. Conversation-Length Distribution Preservation

Sampling is performed at the conversation level, preserving whole multi-turn threads. Below is the quantitative comparison of conversation turn lengths between the full partitions and working subsets:

| Partition / Subset | Total Convs | 2-Turn Convs (%) | 3-Turn Convs (%) | 4+ Turn Convs (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Full Train Partition** | 1 | 100.0% | 0.0% | 0.0% |
| **Working Train Subset** | 1 | 100.0% | 0.0% | 0.0% |
| **Full Val Partition** | 0 | 0.0% | 0.0% | 0.0% |
| **Working Val Subset** | 0 | 0.0% | 0.0% | 0.0% |
| **Full Test Partition** | 1 | 100.0% | 0.0% | 0.0% |
| **Working Test Subset** | 1 | 100.0% | 0.0% | 0.0% |

---

## 5. Zero-Leakage & Data Integrity Proof

- **Train ∩ Validation Conversation Overlap**: `0`
- **Train ∩ Test Conversation Overlap**: `0`
- **Validation ∩ Test Conversation Overlap**: `0`
- **Working Subset Conversation Overlap**: `0`
- **Working Subset Containment Verification**: 100% of working subset conversations are strictly contained within their respective full partitions.

---

## 6. Methodological Clarification: Test Pool vs. Golden Set

1. **`data/processed/test.csv` (7,500 interactions)**: Represents the unlabelled test pool from the working subset. Used for computing automated metrics (Macro F1, Per-intent F1, Confusion Matrix) on model predictions.
2. **Golden Evaluation Set (200 examples)**: Built in **Phase 4**, the Golden Set is **DISTINCT FROM** `test.csv`. It consists of **200 hand-curated, manually annotated examples** drawn independently from the held-out test population with zero overlap.
3. **Strict Evaluation Isolation**: Neither `test.csv` nor the Golden Set is ever accessed during training, feature engineering, vector indexing, or escalation threshold calibration.
