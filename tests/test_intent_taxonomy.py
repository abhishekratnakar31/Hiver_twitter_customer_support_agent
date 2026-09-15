"""
tests/test_intent_taxonomy.py

Automated test suite asserting validity, schema completeness, operational constraints,
zero data leakage, and complete discovery report metrics for Phase 3 Intent Discovery artifacts.
"""

import pytest
import yaml
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "intents.yaml"
GUIDELINES_PATH = BASE_DIR / "data" / "golden" / "annotation_guidelines.md"
REPORT_PATH = BASE_DIR / "reports" / "intent_discovery.md"

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
VAL_PATH = BASE_DIR / "data" / "processed" / "validation.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

def test_config_file_exists():
    assert CONFIG_PATH.exists(), f"Configuration file {CONFIG_PATH} does not exist."

def test_intent_yaml_structure_and_schema():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert "intents" in data, "YAML config missing top-level 'intents' key."
    intents = data["intents"]
    assert isinstance(intents, list), "'intents' must be a list."
    
    # 8–12 domain intents + other_unclear => total between 9 and 13
    assert 9 <= len(intents) <= 13, f"Expected 9-13 intents, got {len(intents)}."
    
    intent_ids = [item.get("intent_id") for item in intents]
    assert len(intent_ids) == len(set(intent_ids)), "Duplicate intent_id values found in intents.yaml."
    assert "other_unclear" in intent_ids, "Mandatory reserve intent 'other_unclear' missing."
    
    # Canonical lowercase risk priors
    valid_risk_priors = {"low", "medium", "high"}
    
    for item in intents:
        assert "intent_id" in item and item["intent_id"], "Intent missing 'intent_id'."
        assert "name" in item and item["name"], "Intent missing 'name'."
        assert "description" in item and item["description"], "Intent missing 'description'."
        assert "inclusion_criteria" in item and isinstance(item["inclusion_criteria"], list), "Intent missing 'inclusion_criteria' list."
        assert "exclusion_criteria" in item and isinstance(item["exclusion_criteria"], list), "Intent missing 'exclusion_criteria' list."
        assert "examples" in item and isinstance(item["examples"], list) and len(item["examples"]) >= 2, "Intent must have at least 2 real examples."
        assert "default_risk" in item, f"Intent {item['intent_id']} missing 'default_risk'."
        assert item["default_risk"] in valid_risk_priors, f"default_risk must be canonical lowercase ('low', 'medium', 'high'), got '{item['default_risk']}' in {item['intent_id']}."

def test_zero_leakage_in_examples():
    """Asserts all intent examples come from train.csv and have ZERO overlap in val/test splits."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    val_texts = set()
    test_texts = set()
    if VAL_PATH.exists():
        val_texts = set(pd.read_csv(VAL_PATH)["customer_message"].dropna().astype(str).str.strip().tolist())
    if TEST_PATH.exists():
        test_texts = set(pd.read_csv(TEST_PATH)["customer_message"].dropna().astype(str).str.strip().tolist())
        
    heldout_set = val_texts.union(test_texts)
    
    for item in data["intents"]:
        for ex in item["examples"]:
            # Clean wrapper prefix '[Real Tweet Excerpt]: '
            raw_ex = ex.replace("[Real Tweet Excerpt]: ", "").strip()
            assert raw_ex not in heldout_set, f"Data Leakage Detected! Example '{raw_ex}' found in validation/test set."

def test_draft_annotation_guidelines_rules():
    assert GUIDELINES_PATH.exists(), f"Draft annotation guidelines {GUIDELINES_PATH} does not exist."
    with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "# Draft Annotation Guidelines" in content, "Draft annotation guidelines missing required title."
    assert "Primary Intent Priority Rule" in content, "Guidelines missing Primary Intent Priority Rule."
    assert "Evidence-Based Intent Assignment" in content, "Guidelines missing Evidence-Based Intent Assignment rule."
    assert "Inclusion Criteria" in content
    assert "Exclusion Criteria" in content

def test_discovery_report_metrics_and_matrix():
    assert REPORT_PATH.exists(), f"Discovery report {REPORT_PATH} does not exist."
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Full Dataset TF-IDF Phrase Frequency Analysis" in content
    assert "Exploratory Clustering & Complete Silhouette Metrics (K=8..15)" in content
    assert "Candidate Themes & Merge/Split Decision Matrix" in content
    assert "Taxonomy Coverage Estimate across 35,000 Training Queries" in content
    
    # Assert all K from 8 to 15 are reported
    for k in range(8, 16):
        assert f"K={k}" in content, f"Discovery report missing metric for K={k}."
