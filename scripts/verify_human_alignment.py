#!/usr/bin/env python3
"""
scripts/verify_human_alignment.py

Phase 6 Human/LLM Judge Alignment Verification:
1. Loads 50 frozen human calibration labels from data/golden/human_calibration_50.csv
   (Carved out from test_pool.csv; strictly disjoint from the 200 Golden Set items).
2. Runs LLMJudge on the 50 examples with frozen prompt.
3. Computes:
   - Spearman Rank Correlation (rho) overall total score and per-criterion.
   - Exact Agreement percentage (human == llm).
   - Within-1 Agreement percentage (|human - llm| <= 1).
4. Reports metrics against informational benchmark target (rho >= 0.75).
   (Methodological note: Calibration set is never used to tune judge prompt or scoring).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

for p in [Path.home() / ".env", Path(__file__).resolve().parent.parent / ".env"]:
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    val = v.strip().strip("'\"")
                    if val and not val.startswith("your_"):
                        os.environ[k.strip()] = val

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.llm_judge import LLMJudge
from src.models.rag_retriever import RAGRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
CALIB_PATH = BASE_DIR / "data" / "golden" / "human_calibration_50.csv"
INDEX_PATH = BASE_DIR / "models" / "rag_faiss.index"
METADATA_PATH = BASE_DIR / "models" / "rag_metadata.json"


def compute_agreement_metrics(human_scores: np.ndarray, llm_scores: np.ndarray) -> dict:
    """Computes exact agreement %, within-1 agreement %, and Spearman rank correlation."""
    human_scores = np.array(human_scores, dtype=float)
    llm_scores = np.array(llm_scores, dtype=float)

    exact_agree = np.mean(human_scores == llm_scores) * 100.0
    within1_agree = np.mean(np.abs(human_scores - llm_scores) <= 1.0) * 100.0

    if len(np.unique(human_scores)) > 1 and len(np.unique(llm_scores)) > 1:
        rho, pval = spearmanr(human_scores, llm_scores)
        rho = float(rho)
        pval = float(pval)
    else:
        rho, pval = 0.0, 1.0

    return {
        "exact_agreement_pct": float(exact_agree),
        "within_1_agreement_pct": float(within1_agree),
        "spearman_rho": float(rho),
        "p_value": float(pval)
    }


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Human/LLM Judge Alignment Verification")
    parser.add_argument("--live", action="store_true", help="Enforce real LLM API evaluation (fails loudly if API key is missing)")
    parser.add_argument("--mock", action="store_true", help="Run evaluation in explicit mock fallback mode")
    return parser.parse_args()


def main():
    args = parse_args()
    print("============================================================")
    print("HIVER PHASE 6: HUMAN / LLM JUDGE ALIGNMENT VERIFICATION")
    print("============================================================")

    if not CALIB_PATH.exists():
        raise FileNotFoundError(f"Missing calibration file at {CALIB_PATH}!")

    calib_df = pd.read_csv(CALIB_PATH)
    print(f"Loaded {len(calib_df)} human calibration examples from {CALIB_PATH}")

    # Verify disjoint isolation from golden_set.csv
    golden_path = BASE_DIR / "data" / "golden" / "golden_set.csv"
    if golden_path.exists():
        gold_df = pd.read_csv(golden_path)
        overlap = set(calib_df["interaction_id"]).intersection(set(gold_df["interaction_id"]))
        assert len(overlap) == 0, f"DATA LEAKAGE WARNING: {len(overlap)} calibration examples overlap with Golden Set!"
        print("Verified zero overlap between human calibration dataset and Golden Set.")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if args.live:
        if not api_key:
            raise ValueError(
                "CRITICAL ERROR: --live flag specified but GEMINI_API_KEY / OPENAI_API_KEY is missing from environment!\n"
                "Silent fallback to mock mode is disabled under --live mode."
            )
        use_mock = False
    elif args.mock:
        use_mock = True
    else:
        allow_mock_env = os.environ.get("ALLOW_MOCK_EVAL", "0") == "1"
        use_mock = allow_mock_env or not bool(api_key)

    provider_str = "Real LLM API" if not use_mock else "Mock Fallback"
    print(f"Evaluator Provider Mode: [{provider_str}]")

    # Load retriever and models
    retriever = RAGRetriever.load(INDEX_PATH, METADATA_PATH)
    judge = LLMJudge(mock_mode=use_mock)

    print("\nRunning LLMJudge evaluations over 50 calibration examples...")
    llm_results = []
    for idx, row in calib_df.iterrows():
        query = str(row["customer_message"])
        retrieved = retriever.retrieve(query, k=3)
        
        # Use actual historical reference reply or query-specific candidate reply
        candidate_reply = str(row.get("historical_reference_reply", "Please DM us your order number so we can help."))
        intent = str(row.get("ground_truth_intent", "order_status"))

        res = judge.evaluate(
            customer_message=query,
            predicted_intent=intent,
            retrieved_evidence=retrieved,
            generated_response=candidate_reply
        )
        llm_results.append(res)
        print(f"  [{idx+1:02d}/50] Evaluated item {row['interaction_id']} (Score: {res['total_score']}/8)", flush=True)
        if not use_mock:
            time.sleep(4.2)

    llm_df = pd.DataFrame(llm_results)

    # Criteria alignment analysis
    criteria = ["correctness", "groundedness", "helpfulness", "policy_safety"]
    alignment_summary = {}

    print("\n--- Per-Criterion & Total Score Alignment Breakdown ---")
    print(f"{'Criterion':20s} | {'Exact Agree %':15s} | {'Within-1 %':12s} | {'Spearman Rho':12s}")
    print("-" * 65)

    for crit in criteria:
        human_col = f"human_{crit}"
        h_vals = calib_df[human_col].values
        l_vals = llm_df[crit].values
        metrics = compute_agreement_metrics(h_vals, l_vals)
        alignment_summary[crit] = metrics
        print(f"{crit:20s} | {metrics['exact_agreement_pct']:13.1f}% | {metrics['within_1_agreement_pct']:10.1f}% | {metrics['spearman_rho']:11.4f}")

    # Total Score Alignment
    h_totals = calib_df["human_total_score"].values
    l_totals = llm_df["total_score"].values
    total_metrics = compute_agreement_metrics(h_totals, l_totals)
    alignment_summary["total_score"] = total_metrics

    print("-" * 65)
    print(f"{'TOTAL SCORE (0-8)':20s} | {total_metrics['exact_agreement_pct']:13.1f}% | {total_metrics['within_1_agreement_pct']:10.1f}% | {total_metrics['spearman_rho']:11.4f}")
    print("=" * 65)

    print(f"\n[Alignment Target Check]: Benchmark Target = 0.75 | Measured Total Spearman Rho = {total_metrics['spearman_rho']:.4f}")
    print("Methodological Note: Target 0.75 is an informational alignment benchmark. Calibration labels and prompt remain frozen.\n")


if __name__ == "__main__":
    main()
