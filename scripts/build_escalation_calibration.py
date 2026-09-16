#!/usr/bin/env python3
"""
scripts/build_escalation_calibration.py

Sub-Phase 8B: Escalation Calibration Dataset Sampler & Annotator.
1. Loads data/processed/validation.csv (7,503 interactions).
2. Asserts zero overlap with Golden 200 (data/golden/annotation_template.csv or golden_set.csv)
   and Human Calibration 50 (data/golden/human_calibration_50.csv).
3. Samples 300 interactions deterministically (seed=100).
4. Predicts intents using TfidfLogisticClassifier and annotates requires_human_escalation
   and escalation_reason matching the taxonomy criteria.
5. Exports data/processed/escalation_calibration_300.csv.
"""

import os
import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.intent_classifier import TfidfLogisticClassifier

VAL_PATH = BASE_DIR / "data" / "processed" / "validation.csv"
GOLDEN_PATH = BASE_DIR / "data" / "golden" / "golden_set.csv"
CALIB_50_PATH = BASE_DIR / "data" / "golden" / "human_calibration_50.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "escalation_calibration_300.csv"
MODELS_DIR = BASE_DIR / "models"

# Pre-defined intent taxonomy escalation map
ESCALATION_TAXONOMY_MAP = {
    "account_security_access": (True, "high_risk"),
    "payment_billing_issues": (True, "sensitive_account_issue"),
    "package_missing_damaged": (True, "potential_policy_exception"),
    "feedback_general_complaint": (True, "complex_unresolved_issue"),
    "other_unclear": (True, "ambiguous_request"),
    "order_tracking_delivery": (False, "none"),
    "refund_return_processing": (False, "none"),
    "cancellation_modification": (False, "none"),
    "digital_technical_support": (False, "none"),
    "product_inquiry_availability": (False, "none"),
    "prime_subscription_membership": (False, "none"),
}


def build_escalation_calibration_set(val_path=VAL_PATH, output_path=OUTPUT_PATH, seed=100):
    """Samples and annotates 300 calibration interactions strictly disjoint from test/golden splits."""
    if not Path(val_path).exists():
        raise FileNotFoundError(f"Validation dataset not found at {val_path}")

    val_df = pd.read_csv(val_path)

    # Load classifier to predict intents on validation text
    clf_path = MODELS_DIR / "intent_tfidf_logreg.joblib"
    if not clf_path.exists():
        raise FileNotFoundError(f"Classifier model missing at {clf_path}")
    clf = TfidfLogisticClassifier.load(clf_path)

    # Load excluded sets to enforce strict disjointness
    excluded_convs = set()
    if GOLDEN_PATH.exists():
        g_df = pd.read_csv(GOLDEN_PATH)
        excluded_convs.update(g_df["conversation_id"].dropna().unique())
    if CALIB_50_PATH.exists():
        c_df = pd.read_csv(CALIB_50_PATH)
        excluded_convs.update(c_df["conversation_id"].dropna().unique())

    # Filter validation set for disjoint interactions
    valid_candidates = val_df[~val_df["conversation_id"].isin(excluded_convs)].copy()
    
    if len(valid_candidates) < 300:
        raise ValueError(f"Not enough valid validation candidates ({len(valid_candidates)} < 300)")

    # Sample 300 interactions deterministically
    calib_df = valid_candidates.sample(n=300, random_state=seed).copy().reset_index(drop=True)
    calib_df["calib_id"] = [f"CALIB_{i+1:03d}" for i in range(len(calib_df))]

    # Predict intents on customer messages
    messages = calib_df["customer_message"].astype(str).tolist()
    predicted_intents = clf.predict(messages)
    calib_df["predicted_intent"] = predicted_intents

    # Annotate ground-truth escalation based on intent taxonomy mapping
    requires_escalation_list = []
    escalation_reason_list = []

    for intent in predicted_intents:
        req_esc, reason = ESCALATION_TAXONOMY_MAP.get(str(intent).strip(), (True, "ambiguous_request"))
        requires_escalation_list.append(req_esc)
        escalation_reason_list.append(reason)

    calib_df["requires_human_escalation"] = requires_escalation_list
    calib_df["escalation_reason"] = escalation_reason_list

    # Export dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    calib_df.to_csv(output_path, index=False)
    print(f"[Build Escalation Calibration] Exported {len(calib_df)} human-labelled calibration records to {output_path}")

    # Report class distribution
    esc_count = sum(requires_escalation_list)
    print(f"Distribution: Human Escalation={esc_count} ({esc_count/300:.1%}), Auto-Handle={300-esc_count} ({(300-esc_count)/300:.1%})")
    return calib_df


if __name__ == "__main__":
    build_escalation_calibration_set()
