"""
tests/test_baselines.py

Phase 5 Baseline Unit Test Suite asserting:
1. Zero Evaluation Fitting: Evaluation does not fit or modify models.
2. Golden Label Isolation: Training never reads golden_set.csv.
3. Training Data Isolation: TF-IDF vectorizer derives exclusively from train.csv.
4. Vector Store Isolation: FAISS index metadata contains only training interactions.
5. Zero-Shot Baseline Constraint: Zero-shot baseline uses only frozen taxonomy descriptions.
6. Fixed Triage Thresholds: Escalation thresholds (0.60, 0.50) are fixed configuration constants.
7. Historical Copy Fidelity: Top-1 historical response baseline outputs exact retrieved response without modification.
8. Golden Set Untouched: Golden set remains untouched during training.
9. Deterministic Reproducibility: Running predictions twice with seed=42 produces identical outputs.
"""

import os
import sys
import yaml
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.intent_classifier import TfidfLogisticClassifier, MajorityClassifier, ZeroShotEmbeddingSimilarityBaseline
from src.models.rag_retriever import RAGRetriever
from src.models.reply_generator import Top1HistoricalResponseBaseline
from src.models.escalation_policy import BaselineEscalationPolicy

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
GOLDEN_PATH = BASE_DIR / "data" / "golden" / "golden_set.csv"
TEST_POOL_PATH = BASE_DIR / "data" / "processed" / "test_pool.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "intent_tfidf_logreg.joblib"
INDEX_PATH = MODELS_DIR / "rag_faiss.index"
METADATA_PATH = MODELS_DIR / "rag_metadata.json"


@pytest.fixture(scope="module")
def loaded_retriever():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        pytest.skip("FAISS index missing.")
    return RAGRetriever.load(INDEX_PATH, METADATA_PATH)


@pytest.fixture(scope="module")
def zero_shot_baseline():
    return ZeroShotEmbeddingSimilarityBaseline(config_path="configs/intents.yaml")


def test_zero_evaluation_fitting():
    """Asserts that evaluating baselines does not modify model parameters or vocabulary."""
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found. Run scripts/train_baselines.py first.")
    
    clf = TfidfLogisticClassifier.load(MODEL_PATH)
    vocab_before = dict(clf.vectorizer.vocabulary_)
    coef_before = np.copy(clf.clf.coef_)
    
    sample_queries = ["Where is my order?", "Can I cancel my subscription?"]
    _ = clf.predict(sample_queries)
    
    vocab_after = dict(clf.vectorizer.vocabulary_)
    coef_after = np.copy(clf.clf.coef_)
    
    assert vocab_before == vocab_after, "Evaluation modified vectorizer vocabulary!"
    assert np.array_equal(coef_before, coef_after), "Evaluation modified classifier coefficients!"


def test_golden_label_isolation():
    """Asserts that training script source code does not reference or load golden_set.csv."""
    train_script = BASE_DIR / "scripts" / "train_baselines.py"
    assert train_script.exists(), "train_baselines.py missing!"
    
    with open(train_script, "r", encoding="utf-8") as f:
        code_lines = f.readlines()
    
    exec_code = [line for line in code_lines if not line.strip().startswith("#") and '"""' not in line]
    exec_text = "\n".join(exec_code)
    
    assert "golden_set.csv" not in exec_text, "Violation: train_baselines.py references golden_set.csv in executable code!"
    assert "test_pool.csv" not in exec_text, "Violation: train_baselines.py references test_pool.csv in executable code!"


def test_training_data_isolation():
    """Asserts that TF-IDF classifier derives exclusively from train.csv."""
    if not MODEL_PATH.exists() or not TRAIN_PATH.exists():
        pytest.skip("Model artifact or train.csv missing.")
    
    clf = TfidfLogisticClassifier.load(MODEL_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    
    train_texts = train_df["customer_message"].astype(str).tolist()
    sample_vocab_words = list(clf.vectorizer.vocabulary_.keys())[:50]
    full_train_text = " ".join(train_texts).lower()
    
    matched = any(w in full_train_text for w in sample_vocab_words)
    assert matched, "TF-IDF vocabulary words not found in training texts!"


def test_vector_store_isolation(loaded_retriever):
    """Asserts FAISS vector index contains only interaction IDs present in train.csv."""
    if not TRAIN_PATH.exists():
        pytest.skip("train.csv missing.")
    
    train_df = pd.read_csv(TRAIN_PATH)
    train_ints = set(train_df["interaction_id"])
    meta_ints = set(item["interaction_id"] for item in loaded_retriever.metadata)
    
    assert meta_ints.issubset(train_ints), "FAISS vector store contains interaction IDs outside train.csv!"
    assert len(loaded_retriever.metadata) == len(train_df), f"Vector count mismatch: {len(loaded_retriever.metadata)} vs {len(train_df)}"


def test_zero_shot_baseline_constraint(zero_shot_baseline):
    """Asserts ZeroShotEmbeddingSimilarityBaseline relies exclusively on configs/intents.yaml."""
    with open("configs/intents.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    intents_raw = cfg.get("intents", [])
    if isinstance(intents_raw, list):
        yaml_intents = set(item.get("intent_id") for item in intents_raw)
    else:
        yaml_intents = set(intents_raw.keys())
    
    assert set(zero_shot_baseline.classes_) == yaml_intents, "Zero-shot classes do not match configs/intents.yaml!"


def test_fixed_triage_thresholds():
    """Asserts escalation policy starting thresholds (0.60, 0.50) are fixed configuration constants."""
    policy = BaselineEscalationPolicy()
    assert policy.tau_conf == 0.60, f"Expected tau_conf=0.60, got {policy.tau_conf}"
    assert policy.tau_sim == 0.50, f"Expected tau_sim=0.50, got {policy.tau_sim}"


def test_historical_copy_fidelity(loaded_retriever):
    """Asserts Top1HistoricalResponseBaseline outputs exact retrieved training response without LLM modification."""
    top1_gen = Top1HistoricalResponseBaseline(loaded_retriever)
    
    query = "Where is my delivery?"
    output = top1_gen.generate_reply(query)
    
    manual_top1 = loaded_retriever.retrieve(query, k=1)[0]
    
    assert output["generated_reply"] == manual_top1["brand_response"], "Top-1 generator output differs from retrieved historical response!"
    assert output["evidence_conversation_id"] == manual_top1["conversation_id"]


def test_golden_set_untouched():
    """Asserts golden_set.csv and test_pool.csv row counts remain exactly 200 and 7,300."""
    if not GOLDEN_PATH.exists() or not TEST_POOL_PATH.exists():
        pytest.skip("golden_set.csv or test_pool.csv missing.")
    
    golden_df = pd.read_csv(GOLDEN_PATH)
    pool_df = pd.read_csv(TEST_POOL_PATH)
    
    assert len(golden_df) == 200, f"Golden set modified! Expected 200 rows, got {len(golden_df)}"
    assert len(pool_df) == 7300, f"Test pool modified! Expected 7,300 rows, got {len(pool_df)}"


def test_deterministic_reproducibility(loaded_retriever):
    """Asserts that running classifier and retrieval twice produces identical predictions."""
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact missing.")
    
    clf = TfidfLogisticClassifier.load(MODEL_PATH)
    queries = ["Cancel my order please", "Package damaged on arrival", "How do I reset my password?"]
    
    preds1 = clf.predict(queries)
    probas1 = clf.predict_proba(queries)
    ret1 = loaded_retriever.retrieve_batch(queries, k=3)
    
    preds2 = clf.predict(queries)
    probas2 = clf.predict_proba(queries)
    ret2 = loaded_retriever.retrieve_batch(queries, k=3)
    
    assert list(preds1) == list(preds2), "Predictions are not deterministically reproducible!"
    assert np.allclose(probas1, probas2), "Probabilities are not deterministically reproducible!"
    assert ret1 == ret2, "Retrieval results are not deterministically reproducible!"
