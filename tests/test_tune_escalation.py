"""
tests/test_tune_escalation.py

Unit tests for threshold optimization script in scripts/tune_escalation.py.
Asserts calibration set generation, grid search execution, constraint enforcement, and config export.
"""

import json
import pytest
from pathlib import Path
from scripts.build_escalation_calibration import build_escalation_calibration_set
from scripts.tune_escalation import run_threshold_grid_search

BASE_DIR = Path(__file__).resolve().parent.parent


def test_build_escalation_calibration_set(tmp_path):
    """Asserts calibration set generation generates 300 examples with required schema."""
    val_path = BASE_DIR / "data" / "processed" / "validation.csv"
    if not val_path.exists():
        pytest.skip("validation.csv not available")

    out_csv = tmp_path / "test_calib_300.csv"
    df = build_escalation_calibration_set(val_path=val_path, output_path=out_csv, seed=100)

    assert len(df) == 300
    assert "requires_human_escalation" in df.columns
    assert "escalation_reason" in df.columns
    assert df["requires_human_escalation"].dtype == bool


def test_run_threshold_grid_search(tmp_path):
    """Asserts grid search execution exports escalation_config.json matching schema."""
    calib_csv = BASE_DIR / "data" / "processed" / "escalation_calibration_300.csv"
    if not calib_csv.exists():
        build_escalation_calibration_set(output_path=calib_csv)

    out_config = tmp_path / "escalation_config.json"
    res = run_threshold_grid_search(calib_path=calib_csv, config_path=out_config)

    assert out_config.exists()
    assert "tau_conf" in res
    assert "tau_sim" in res
    assert 0.40 <= res["tau_conf"] <= 0.85
    assert 0.40 <= res["tau_sim"] <= 0.85
    assert "calibration_metrics" in res
    assert res["calibration_metrics"]["false_auto_rate"] <= 0.1501
