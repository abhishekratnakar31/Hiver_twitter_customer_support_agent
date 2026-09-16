#!/usr/bin/env python3
"""
scripts/tune_escalation.py

Sub-Phase 8B: Escalation Threshold Optimization.
Performs grid search over tau_conf and tau_sim strictly on data/processed/escalation_calibration_300.csv
(human-labelled 300-example calibration set disjoint from Golden 200).

Optimizes Triage F1 subject to False Auto-Handle Rate <= 15.0%.
Exports optimal configuration to configs/escalation_config.json.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.intent_classifier import TfidfLogisticClassifier
from src.models.rag_retriever import RAGRetriever
from src.models.escalation_policy import BaselineEscalationPolicy
from scripts.build_escalation_calibration import build_escalation_calibration_set

CALIB_PATH = BASE_DIR / "data" / "processed" / "escalation_calibration_300.csv"
CONFIG_PATH = BASE_DIR / "configs" / "escalation_config.json"
MODELS_DIR = BASE_DIR / "models"


def run_threshold_grid_search(calib_path=CALIB_PATH, config_path=CONFIG_PATH):
    """
    Performs grid search over tau_conf in [0.40, 0.85] and tau_sim in [0.40, 0.85].
    Evaluates on 300-example human-labelled calibration set.
    Selects optimal thresholds enforcing False Auto-Handle Rate <= 15.0%.
    """
    if not Path(calib_path).exists():
        print(f"[Tune Escalation] Calibration set missing at {calib_path}. Building now...")
        build_escalation_calibration_set(output_path=calib_path)

    calib_df = pd.read_csv(calib_path)
    
    # Load model artifacts
    clf_path = MODELS_DIR / "intent_tfidf_logreg.joblib"
    idx_path = MODELS_DIR / "rag_faiss.index"
    meta_path = MODELS_DIR / "rag_metadata.json"

    if not clf_path.exists() or not idx_path.exists():
        raise FileNotFoundError("Missing model artifacts. Run scripts/train_baselines.py first!")

    clf = TfidfLogisticClassifier.load(clf_path)
    retriever = RAGRetriever.load(idx_path, meta_path)

    # Pre-compute classifier confidences and retrieval similarities
    messages = calib_df["customer_message"].tolist()
    gt_escalations = calib_df["requires_human_escalation"].values

    predicted_intents = clf.predict(messages)
    probas = clf.predict_proba(messages)
    confidences = np.max(probas, axis=1)

    similarities = []
    for msg in messages:
        docs = retriever.retrieve(msg, k=1)
        sim = docs[0]["similarity_score"] if docs else 0.0
        similarities.append(sim)
    similarities = np.array(similarities)

    # Grid search grid
    tau_conf_grid = np.round(np.arange(0.40, 0.86, 0.05), 2)
    tau_sim_grid = np.round(np.arange(0.40, 0.86, 0.05), 2)

    best_f1 = -1.0
    best_params = {"tau_conf": 0.60, "tau_sim": 0.50}  # Baseline default fallback
    best_metrics = {}

    grid_results = []

    for t_conf in tau_conf_grid:
        for t_sim in tau_sim_grid:
            policy = BaselineEscalationPolicy(tau_conf=t_conf, tau_sim=t_sim)
            
            triage_results = policy.evaluate_triage_batch(
                predicted_intents=predicted_intents,
                intent_confidences=confidences,
                retrieval_similarities=similarities
            )
            
            pred_escalations = np.array([r["requires_human_escalation"] for r in triage_results])

            tp = np.sum((gt_escalations == True) & (pred_escalations == True))
            fp = np.sum((gt_escalations == False) & (pred_escalations == True))
            fp_auto = np.sum((gt_escalations == True) & (pred_escalations == False))
            tn = np.sum((gt_escalations == False) & (pred_escalations == False))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fp_auto) if (tp + fp_auto) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            total_true_escalate = tp + fp_auto
            false_auto_rate = fp_auto / total_true_escalate if total_true_escalate > 0 else 0.0

            res_entry = {
                "tau_conf": float(t_conf),
                "tau_sim": float(t_sim),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "false_auto_rate": round(float(false_auto_rate), 4),
                "tp": int(tp), "fp": int(fp), "fp_auto": int(fp_auto), "tn": int(tn)
            }
            grid_results.append(res_entry)

            # Selection criterion: Maximize F1 subject to false_auto_rate <= 0.15
            if false_auto_rate <= 0.1501 and f1 > best_f1:
                best_f1 = f1
                best_params = {"tau_conf": float(t_conf), "tau_sim": float(t_sim)}
                best_metrics = res_entry

    print("=== Escalation Threshold Grid Search Complete ===")
    print(f"Optimal Thresholds Selected: tau_conf={best_params['tau_conf']}, tau_sim={best_params['tau_sim']}")
    print(f"Metrics on Calibration Set: F1={best_metrics.get('f1', 0.0)}, Precision={best_metrics.get('precision', 0.0)}, Recall={best_metrics.get('recall', 0.0)}, False Auto Rate={best_metrics.get('false_auto_rate', 0.0):.1%}")

    # Export to config json
    export_payload = {
        "tau_conf": best_params["tau_conf"],
        "tau_sim": best_params["tau_sim"],
        "baseline_tau_conf": 0.60,
        "baseline_tau_sim": 0.50,
        "calibration_metrics": best_metrics,
        "calibration_sample_size": len(calib_df),
        "constraint": "false_auto_rate <= 15.0%"
    }

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"Saved optimal escalation thresholds to {config_path}")

    return export_payload


if __name__ == "__main__":
    run_threshold_grid_search()
