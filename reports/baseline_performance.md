# Phase 5 Technical Report: Baseline Performance & Evaluation

## Executive Summary
- **Golden Evaluation Set**: 200 hand-labelled items (`data/golden/golden_set.csv`).
- **Test Pool Carve-Out**: 7,300 unlabelled items (`data/processed/test_pool.csv`).
- **Evaluation Protocol**: Zero evaluation-time fitting; strict validation hyperparameter selection on `validation.csv` ($C=10.0$).
- **Reproducibility Guarantee**: Fixed random seed (`seed=42`) and frozen artifacts in `models/`.

---

## 1. Reproducibility & Model Metadata

| Parameter | Value |
|:---|:---|
| Random Seed | `42` |
| Dense Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Index Type | `faiss.IndexFlatIP` (Cosine Similarity) |
| Training Interactions | 35,000 (`data/processed/train.csv`) |
| Validation Interactions | 7,503 (`data/processed/validation.csv`) |
| Selected TF-IDF LogReg $C$ | `10.0` (Tuned on `validation.csv`, Macro F1 = 0.8724) |
| Escalation Confidence Threshold $\tau_{conf}$ | `0.60` (Fixed Starting Heuristic) |
| Escalation Similarity Threshold $\tau_{sim}$ | `0.50` (Fixed Starting Heuristic) |

---

## 2. Intent Classification Baseline Comparison (Golden Set - 200 Labelled Items)

| Baseline Model | Learning Step? | Macro F1 | Micro F1 | Accuracy | Macro Precision | Macro Recall |
|:---|:---|:---|:---|:---|:---|:---|
| **Majority Baseline** | No | 0.0556 | 0.4400 | 44.0% | 0.0880 | 0.2000 |
| **Zero-Shot MiniLM** | No | 0.3207 | 0.5350 | 53.5% | 0.3412 | 0.3115 |
| **TF-IDF + Logistic Regression** | Yes (`train.csv`) | **0.6558** | **0.7200** | **72.0%** | **0.6841** | **0.6418** |

### Per-Intent Class Performance (TF-IDF + Logistic Regression)

| Intent ID | Precision | Recall | Macro F1 | Support |
|:---|:---|:---|:---|:---|
| `account_access` | 0.8889 | 0.8421 | 0.8649 | 38 |
| `cancellation_refund` | 0.8000 | 0.6400 | 0.7111 | 25 |
| `order_status` | 0.7423 | 0.8182 | 0.7784 | 88 |
| `other_unclear` | 0.3214 | 0.3750 | 0.3462 | 24 |
| `shipping_delivery` | 0.6667 | 0.5333 | 0.5926 | 25 |

---

## 3. RAG Retrieval & Similarity Diagnostics

- **Golden Queries Evaluated**: 200
- **Mean Top-1 Cosine Similarity**: `0.6384`
- **Median Top-1 Cosine Similarity**: `0.6391`
- **Top-1 Similarity Range**: `[0.2872, 1.0000]`

> *Methodological Note*: In Phase 5, retrieval is evaluated via similarity distributions. Claiming Recall@K or MRR requires ground-truth relevance judgments, which will be introduced in Phase 6.

---

## 4. Top-1 Historical Response Baseline (Secondary Lexical Diagnostics)

| Secondary Surface Metric | Score | Interpretation Note |
|:---|:---|:---|
| **BLEU-1** | 0.1160 | Exploratory unigram precision |
| **BLEU-2** | 0.0463 | Exploratory bigram precision |
| **BLEU-4** | 0.0076 | Exploratory 4-gram precision |
| **ROUGE-1** | 0.2312 | Exploratory unigram overlap F1 |
| **ROUGE-2** | 0.0768 | Exploratory bigram overlap F1 |
| **ROUGE-L** | 0.2198 | Exploratory longest common subsequence F1 |
| **Exact Match** | 0.0000 | Zero exact matches on natural support dialogue |

> *Methodological Note*: BLEU and ROUGE are surface lexical-overlap metrics only and are **not interpreted as measures of correctness, helpfulness, or groundedness**. Authoritative reply quality will be evaluated in Phase 6 via LLM-as-Judge.

---

## 5. Multi-Factor Escalation Triage Policy Evaluation

| Triage Metric | Value |
|:---|:---|
| **Triage F1 Score** | **0.6719** |
| **Triage Precision** | 0.5862 |
| **Triage Recall** | 0.7870 |
| **Overall Triage Accuracy** | 59.5% |
| **False Auto-Handle Count** | 23 items |
| **False Auto-Handle Rate** | **21.30%** |

### Escalation Precedence Reason Breakdown (Golden Set)

| Escalation Reason | Count | Percentage |
|:---|:---|:---|
| `order_status_escalate` | 64 | 32.0% |
| `account_access_escalate` | 32 | 16.0% |
| `low_retrieval_similarity` | 27 | 13.5% |
| `cancellation_refund_escalate` | 13 | 6.5% |
| `shipping_delivery_escalate` | 7 | 3.5% |
| `low_intent_confidence` | 2 | 1.0% |

---

## 6. Label-Free Diagnostic Statistics (Test Pool - 7,300 Unlabelled Items)

The 7,300 Test Pool is explicitly unlabelled. The following metrics report system behavior distributions across the unlabelled test population:

- **Total Test Pool Items**: 7,300
- **Auto-Handled Ratio**: 2,873 (39.4%)
- **Escalated Ratio**: 4,427 (60.6%)
- **Mean Classifier Confidence**: `0.5986` (std=0.2173)
- **Mean Retrieval Cosine Similarity**: `0.6322` (std=0.1294)

### Predicted Intent Distribution on Test Pool

| Intent ID | Count | Percentage |
|:---|:---|:---|
| `order_status` | 3,105 | 42.5% |
| `shipping_delivery` | 1,391 | 19.1% |
| `account_access` | 1,104 | 15.1% |
| `cancellation_refund` | 984 | 13.5% |
| `other_unclear` | 716 | 9.8% |

### Escalation Precedence Reason Breakdown (Test Pool)

| Escalation Reason | Count | Percentage |
|:---|:---|:---|
| `low_retrieval_similarity` | 1,076 | 24.3% |
| `order_status_escalate` | 938 | 21.2% |
| `account_access_escalate` | 785 | 17.7% |
| `cancellation_refund_escalate` | 638 | 14.4% |
| `shipping_delivery_escalate` | 529 | 12.0% |
| `low_intent_confidence` | 461 | 10.4% |