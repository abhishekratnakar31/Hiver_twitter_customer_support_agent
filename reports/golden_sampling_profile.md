# Phase 4 Stage 1 Technical Report: Candidate Sampling & Test Pool Carve-Out

## Executive Summary
- **Original Held-Out Test interactions**: 7,500 (`data/processed/test.csv`).
- **Golden Candidates Sampled**: 200 unique conversations (`seed=42`).
- **Untouched Test Pool Carved Out**: 7,300 interactions (`data/processed/test_pool.csv`).
- **Conversation Uniqueness**: `nunique(conversation_id) == 200` (0 conversation splitting across splits).
- **Multi-Turn Breakdown**: 200 Single-Turn (turn 1), 0 Multi-Turn (turn > 1).

## Dual Zero Leakage & Partition Reconstruction Assertions
- `Golden ∩ Train = ∅` (checked via `interaction_id` and `conversation_id`).
- `Golden ∩ Val = ∅` (checked via `interaction_id` and `conversation_id`).
- `Golden ∩ Test Pool = ∅` (checked via `interaction_id` and `conversation_id`).
- `Golden ∪ Test Pool == Original Test` (exact set equality across 7,500 interaction IDs).

## Candidate Schema Exported to `data/golden/annotation_template.csv`
Target fields ready for human annotation: `ground_truth_intent`, `risk_tier`, `requires_human_escalation`, `escalation_reason`, `annotation_notes`.
