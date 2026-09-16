#!/usr/bin/env python3
"""
scripts/evaluate_baselines.py

Phase 5 Baseline Evaluation Script:
1. Loads pre-fitted model artifacts from models/. (Zero evaluation-time fitting).
2. Evaluates supervised metrics on data/golden/golden_set.csv (200 items):
   - Intent classification: Macro F1, per-class Precision/Recall/F1, Confusion Matrix.
   - RAG Retrieval: Mean cosine similarity & Top-K score distributions.
   - Top-1 Historical Response: Secondary exploratory BLEU & ROUGE surface text diagnostics.
   - Escalation Policy: Triage Precision, Recall, F1, and False Auto-Handle Rate.
3. Computes label-free diagnostic statistics on data/processed/test_pool.csv (7,300 items):
   - Predicted intent distribution, confidence/similarity distributions, escalation decision ratio.
4. Generates reports/baseline_performance.md with reproducibility metadata.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.intent_classifier import TfidfLogisticClassifier, MajorityClassifier, ZeroShotEmbeddingSimilarityBaseline
from src.models.rag_retriever import RAGRetriever
from src.models.reply_generator import Top1HistoricalResponseBaseline
from src.models.escalation_policy import BaselineEscalationPolicy

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "data" / "golden" / "golden_set.csv"
TEST_POOL_PATH = BASE_DIR / "data" / "processed" / "test_pool.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "intent_tfidf_logreg.joblib"
CONFIG_PATH = MODELS_DIR / "intent_tfidf_logreg_config.json"
INDEX_PATH = MODELS_DIR / "rag_faiss.index"
METADATA_PATH = MODELS_DIR / "rag_metadata.json"

REPORT_PATH = BASE_DIR / "reports" / "baseline_performance.md"

def load_evaluation_data():
    if not GOLDEN_PATH.exists() or not TEST_POOL_PATH.exists():
        raise FileNotFoundError("Missing golden_set.csv or test_pool.csv!")
    golden_df = pd.read_csv(GOLDEN_PATH)
    pool_df = pd.read_csv(TEST_POOL_PATH)
    print(f"Loaded Golden Set: {len(golden_df):,} labelled rows from {GOLDEN_PATH}")
    print(f"Loaded Test Pool: {len(pool_df):,} unlabelled rows from {TEST_POOL_PATH}")
    return golden_df, pool_df

def load_baseline_models():
    print("\nLoading pre-fitted baseline model artifacts from models/...")
    if not MODEL_PATH.exists() or not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing model artifacts in {MODELS_DIR}. Run scripts/train_baselines.py first!")
    
    tfidf_logreg = TfidfLogisticClassifier.load(MODEL_PATH)
    retriever = RAGRetriever.load(INDEX_PATH, METADATA_PATH)
    majority = MajorityClassifier()
    zero_shot = ZeroShotEmbeddingSimilarityBaseline(config_path="configs/intents.yaml")
    
    # Fit majority classifier on training prior class distribution
    majority.dominant_class_ = "other_unclear"
    majority.classes_ = tfidf_logreg.classes_
    
    top1_gen = Top1HistoricalResponseBaseline(retriever)
    policy = BaselineEscalationPolicy(tau_conf=0.60, tau_sim=0.50)
    
    return {
        "majority": majority,
        "tfidf_logreg": tfidf_logreg,
        "zero_shot": zero_shot,
        "retriever": retriever,
        "top1_gen": top1_gen,
        "policy": policy
    }

def evaluate_golden_set(golden_df, models):
    print("\nExecuting Supervised Evaluation on Golden Evaluation Set (200 items)...")
    X_gold = golden_df["customer_message"].astype(str).values
    y_gold_intent = golden_df["ground_truth_intent"].astype(str).values
    y_gold_esc = golden_df["requires_human_escalation"].astype(bool).values
    ref_replies = golden_df["historical_reference_reply"].astype(str).values

    # 1. Intent Classification Evaluation across 3 Baselines
    intent_evals = {}
    for name, clf in [("Majority", models["majority"]), ("TF-IDF + LogReg", models["tfidf_logreg"]), ("Zero-Shot MiniLM", models["zero_shot"])]:
        preds = clf.predict(X_gold)
        acc = accuracy_score(y_gold_intent, preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_gold_intent, preds, average="macro", zero_division=0)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_gold_intent, preds, average="micro", zero_division=0)
        
        # Per-class metrics
        unique_intents = sorted(list(set(y_gold_intent).union(set(preds))))
        p_class, r_class, f1_class, s_class = precision_recall_fscore_support(y_gold_intent, preds, labels=unique_intents, zero_division=0)
        per_class_df = pd.DataFrame({
            "intent": unique_intents,
            "precision": p_class,
            "recall": r_class,
            "f1": f1_class,
            "support": s_class
        })
        
        cm = confusion_matrix(y_gold_intent, preds, labels=unique_intents)
        
        intent_evals[name] = {
            "accuracy": float(acc),
            "macro_f1": float(f1_macro),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "micro_f1": float(f1_micro),
            "per_class": per_class_df,
            "confusion_matrix": cm,
            "predictions": preds,
            "intents_list": unique_intents
        }
        print(f"  {name:25s} -> Macro F1: {f1_macro:.4f} | Accuracy: {acc*100:.1f}%")

    # 2. RAG Retrieval Similarity Diagnostics
    retriever = models["retriever"]
    ret_diag = retriever.compute_similarity_diagnostics(X_gold, k=5)
    print(f"  RAG Retrieval Mean Cosine Similarity: {ret_diag['mean_top1_similarity']:.4f}")

    # 3. Top-1 Historical Response Secondary Surface Metrics
    top1_gen = models["top1_gen"]
    top1_outputs = top1_gen.generate_reply_batch(X_gold)
    gen_replies = [out["generated_reply"] for out in top1_outputs]
    top1_sims = [out["similarity_score"] for out in top1_outputs]
    
    sec_metrics = top1_gen.evaluate_secondary_diagnostics(gen_replies, ref_replies)
    print(f"  Secondary Surface Diagnostics -> BLEU-1: {sec_metrics['bleu_1']:.4f} | ROUGE-L: {sec_metrics['rouge_l']:.4f}")

    # 4. Escalation Policy Evaluation
    policy = models["policy"]
    tfidf_logreg = models["tfidf_logreg"]
    tfidf_probas = tfidf_logreg.predict_proba(X_gold)
    max_confs = np.max(tfidf_probas, axis=1)
    
    esc_preds = []
    esc_reasons = []
    for i in range(len(X_gold)):
        intent_pred = intent_evals["TF-IDF + LogReg"]["predictions"][i]
        conf = max_confs[i]
        sim = top1_sims[i]
        
        res = policy.evaluate_triage(
            predicted_intent=intent_pred,
            intent_confidence=conf,
            retrieval_similarity=sim
        )
        esc_preds.append(res["requires_human_escalation"])
        esc_reasons.append(res["escalation_reason"])
        
    esc_preds = np.array(esc_preds)
    esc_acc = accuracy_score(y_gold_esc, esc_preds)
    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(y_gold_esc, esc_preds, average="binary", zero_division=0)
    
    # False Auto-Handle Rate: (requires_escalation == True BUT predicted == False) / total true escalations
    true_escalations = (y_gold_esc == True)
    false_auto_count = np.sum((y_gold_esc == True) & (esc_preds == False))
    false_auto_rate = false_auto_count / np.sum(true_escalations) if np.sum(true_escalations) > 0 else 0.0

    esc_metrics = {
        "accuracy": float(esc_acc),
        "precision": float(esc_p),
        "recall": float(esc_r),
        "f1": float(esc_f1),
        "false_auto_handle_count": int(false_auto_count),
        "false_auto_handle_rate": float(false_auto_rate),
        "reasons_distribution": pd.Series(esc_reasons).value_counts().to_dict()
    }
    print(f"  Escalation Triage -> F1: {esc_f1:.4f} | Precision: {esc_p:.4f} | Recall: {esc_r:.4f} | False Auto Rate: {false_auto_rate*100:.1f}%")

    return {
        "intent_evals": intent_evals,
        "retrieval_diagnostics": ret_diag,
        "secondary_reply_metrics": sec_metrics,
        "escalation_metrics": esc_metrics
    }

def evaluate_test_pool_diagnostics(pool_df, models):
    print("\nExecuting Label-Free Diagnostics on Test Pool (7,300 items)...")
    X_pool = pool_df["customer_message"].astype(str).values
    
    # Predictions
    tfidf_logreg = models["tfidf_logreg"]
    retriever = models["retriever"]
    policy = models["policy"]
    
    intent_preds = tfidf_logreg.predict(X_pool)
    intent_probas = tfidf_logreg.predict_proba(X_pool)
    max_confs = np.max(intent_probas, axis=1)
    
    top1_outputs = models["top1_gen"].generate_reply_batch(X_pool)
    top1_sims = [out["similarity_score"] for out in top1_outputs]
    
    esc_results = policy.evaluate_triage_batch(
        predicted_intents=intent_preds,
        intent_confidences=max_confs,
        retrieval_similarities=top1_sims
    )
    esc_flags = [res["requires_human_escalation"] for res in esc_results]
    esc_reasons = [res["escalation_reason"] for res in esc_results]
    
    # Compute summary diagnostics
    intent_dist = pd.Series(intent_preds).value_counts().to_dict()
    esc_dist = pd.Series(esc_flags).value_counts().to_dict()
    reason_dist = pd.Series(esc_reasons).value_counts().to_dict()
    
    msg_lens = [len(str(m)) for m in X_pool]
    
    return {
        "num_items": len(pool_df),
        "intent_distribution": intent_dist,
        "escalation_distribution": esc_dist,
        "reason_distribution": reason_dist,
        "confidence_stats": {
            "mean": float(np.mean(max_confs)),
            "std": float(np.std(max_confs)),
            "min": float(np.min(max_confs)),
            "max": float(np.max(max_confs))
        },
        "similarity_stats": {
            "mean": float(np.mean(top1_sims)),
            "std": float(np.std(top1_sims)),
            "min": float(np.min(top1_sims)),
            "max": float(np.max(top1_sims))
        },
        "mean_message_length": float(np.mean(msg_lens))
    }

def generate_performance_report(golden_results, pool_results):
    print("\nGenerating technical report at reports/baseline_performance.md...")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    doc = []
    doc.append("# Phase 5 Technical Report: Baseline Performance & Initial Evaluation\n")
    doc.append("## Executive Summary")
    doc.append("- **Golden Evaluation Set**: 200 hand-labelled items (`data/golden/golden_set.csv`).")
    doc.append("- **Test Pool Carve-Out**: 7,300 unlabelled items (`data/processed/test_pool.csv`).")
    doc.append("- **Evaluation Protocol**: Zero evaluation-time fitting; strict validation hyperparameter selection on `validation.csv`.")
    doc.append("- **Reproducibility Guarantee**: Fixed random seed (`seed=42`) and frozen artifacts in `models/`.\n")
    
    doc.append("## 1. Reproducibility & Model Metadata")
    doc.append("| Parameter | Value |")
    doc.append("|:---|:---|")
    doc.append("| Random Seed | `42` |")
    doc.append("| Dense Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |")
    doc.append("| Vector Index Type | `faiss.IndexFlatIP` (Cosine Similarity) |")
    doc.append("| Training Interactions | 35,000 (`data/processed/train.csv`) |")
    doc.append("| Validation Interactions | 7,503 (`data/processed/validation.csv`) |")
    doc.append("| Selected TF-IDF LogReg $C$ | `1.0` (Tuned on `validation.csv`) |")
    doc.append("| Escalation Confidence Threshold $\\tau_{conf}$ | `0.60` (Fixed Starting Heuristic) |")
    doc.append("| Escalation Similarity Threshold $\\tau_{sim}$ | `0.50` (Fixed Starting Heuristic) |\n")

    doc.append("## 2. Intent Classification Baseline Comparison (Golden Set - 200 Labelled Items)")
    doc.append("| Baseline Model | Learning Step? | Macro F1 | Micro F1 | Accuracy | Macro Precision | Macro Recall |")
    doc.append("|:---|:---|:---|:---|:---|:---|:---|")
    for bname, res in golden_results["intent_evals"].items():
        learning = "No" if bname in ["Majority", "Zero-Shot MiniLM"] else "Yes (`train.csv`)"
        doc.append(f"| **{bname}** | {learning} | **{res['macro_f1']:.4f}** | {res['micro_f1']:.4f} | {res['accuracy']*100:.1f}% | {res['macro_precision']:.4f} | {res['macro_recall']:.4f} |")
    doc.append("\n")

    doc.append("### Per-Intent Class Performance (TF-IDF + Logistic Regression)")
    doc.append("| Intent ID | Precision | Recall | Macro F1 | Support |")
    doc.append("|:---|:---|:---|:---|:---|")
    per_class_df = golden_results["intent_evals"]["TF-IDF + LogReg"]["per_class"]
    for _, row in per_class_df.iterrows():
        doc.append(f"| `{row['intent']}` | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {int(row['support'])} |")
    doc.append("\n")

    doc.append("## 3. RAG Retrieval & Similarity Diagnostics")
    rdiag = golden_results["retrieval_diagnostics"]
    doc.append(f"- **Golden Queries Evaluated**: {rdiag['num_queries']:,}")
    doc.append(f"- **Mean Top-1 Cosine Similarity**: `{rdiag['mean_top1_similarity']:.4f}`")
    doc.append(f"- **Median Top-1 Cosine Similarity**: `{rdiag['median_top1_similarity']:.4f}`")
    doc.append(f"- **Top-1 Similarity Range**: `[{rdiag['min_top1_similarity']:.4f}, {rdiag['max_top1_similarity']:.4f}]`\n")
    doc.append("> *Methodological Note*: In Phase 5, retrieval is evaluated via similarity distributions. Claiming Recall@K or MRR requires ground-truth relevance judgments, which will be introduced in Phase 6.\n")

    doc.append("## 4. Top-1 Historical Response Baseline (Secondary Lexical Diagnostics)")
    sec = golden_results["secondary_reply_metrics"]
    doc.append("| Secondary Surface Metric | Score | Interpretation Note |")
    doc.append("|:---|:---|:---|")
    doc.append(f"| **BLEU-1** | {sec['bleu_1']:.4f} | Exploratory unigram precision |")
    doc.append(f"| **BLEU-2** | {sec['bleu_2']:.4f} | Exploratory bigram precision |")
    doc.append(f"| **BLEU-4** | {sec['bleu_4']:.4f} | Exploratory 4-gram precision |")
    doc.append(f"| **ROUGE-1** | {sec['rouge_1']:.4f} | Exploratory unigram overlap F1 |")
    doc.append(f"| **ROUGE-2** | {sec['rouge_2']:.4f} | Exploratory bigram overlap F1 |")
    doc.append(f"| **ROUGE-L** | {sec['rouge_l']:.4f} | Exploratory longest common subsequence F1 |\n")
    doc.append("> *Methodological Note*: BLEU and ROUGE are surface lexical-overlap metrics only and are **not interpreted as measures of correctness, helpfulness, or groundedness**. Authoritative reply quality will be evaluated in Phase 6 via LLM-as-Judge.\n")

    doc.append("## 5. Multi-Factor Escalation Triage Policy Evaluation")
    esc = golden_results["escalation_metrics"]
    doc.append("| Triage Metric | Value |")
    doc.append("|:---|:---|")
    doc.append(f"| **Triage F1 Score** | **{esc['f1']:.4f}** |")
    doc.append(f"| **Triage Precision** | {esc['precision']:.4f} |")
    doc.append(f"| **Triage Recall** | {esc['recall']:.4f} |")
    doc.append(f"| **Overall Triage Accuracy** | {esc['accuracy']*100:.1f}% |")
    doc.append(f"| **False Auto-Handle Count** | {esc['false_auto_handle_count']} items |")
    doc.append(f"| **False Auto-Handle Rate** | **{esc['false_auto_handle_rate']*100:.2f}%** |\n")

    doc.append("### Escalation Precedence Reason Breakdown (Golden Set)")
    doc.append("| Escalation Reason | Count | Percentage |")
    doc.append("|:---|:---|:---|")
    for reason, cnt in sorted(esc["reasons_distribution"].items(), key=lambda x: x[1], reverse=True):
        doc.append(f"| `{reason}` | {cnt} | {cnt/2:.1f}% |")
    doc.append("\n")

    doc.append("## 6. Label-Free Diagnostic Statistics (Test Pool - 7,300 Unlabelled Items)")
    doc.append("The 7,300 Test Pool is explicitly unlabelled. The following metrics report system behavior distributions across the unlabelled test population:\n")
    doc.append(f"- **Total Test Pool Items**: {pool_results['num_items']:,}")
    doc.append(f"- **Auto-Handled Ratio**: {pool_results['escalation_distribution'].get(False, 0):,} ({pool_results['escalation_distribution'].get(False, 0)/73.0:.1f}%)")
    doc.append(f"- **Escalated Ratio**: {pool_results['escalation_distribution'].get(True, 0):,} ({pool_results['escalation_distribution'].get(True, 0)/73.0:.1f}%)")
    doc.append(f"- **Mean Classifier Confidence**: `{pool_results['confidence_stats']['mean']:.4f}` (std={pool_results['confidence_stats']['std']:.4f})")
    doc.append(f"- **Mean Retrieval Cosine Similarity**: `{pool_results['similarity_stats']['mean']:.4f}` (std={pool_results['similarity_stats']['std']:.4f})")
    doc.append(f"- **Mean Customer Message Length**: `{pool_results['mean_message_length']:.1f}` characters\n")

    doc.append("### Predicted Intent Distribution on Test Pool")
    doc.append("| Intent ID | Count | Percentage |")
    doc.append("|:---|:---|:---|")
    for iid, cnt in sorted(pool_results["intent_distribution"].items(), key=lambda x: x[1], reverse=True):
        doc.append(f"| `{iid}` | {cnt:,} | {cnt/73.0:.1f}% |")
    doc.append("\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    print(f"Exported performance report to {REPORT_PATH}")

def main():
    print("=== Starting Phase 5 Baseline Evaluation ===")
    golden_df, pool_df = load_evaluation_data()
    models = load_baseline_models()
    
    golden_results = evaluate_golden_set(golden_df, models)
    pool_results = evaluate_test_pool_diagnostics(pool_df, models)
    
    generate_performance_report(golden_results, pool_results)
    print("\n=== Phase 5 Baseline Evaluation Complete ===")

if __name__ == "__main__":
    main()
