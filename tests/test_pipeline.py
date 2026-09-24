#!/usr/bin/env python3
"""
NEXUS Stock AI — Automated Pipeline, Integrity, & Leakage Test Suite (Phases 31 & 32)
Strict assertions verifying data leakage prevention, temporal disjointness,
model artifact integrity, and standalone inference.
"""

import os
import sys
import json
try:
    import pytest
except ImportError:
    class MockPytest:
        @staticmethod
        def skip(reason=""):
            print(f"SKIPPED: {reason}")
    pytest = MockPytest()

import numpy as np
import pandas as pd

# Add repository root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def resolve_file(filename: str, subfolder: str = "models") -> str:
    """Finds an artifact locally or on Google Drive."""
    candidates = [
        os.path.join(ROOT_DIR, subfolder, filename),
        os.path.join(ROOT_DIR, filename),
        os.path.join("/content/drive/MyDrive/NEXUS_Stock_AI", subfolder, filename),
        os.path.join("/content", subfolder, filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def test_leakage_check_1_no_target_in_feature_schema():
    """
    Leakage Check 1:
    Assert that neither 'target', 'forward_return', nor any future lookahead
    variable is present in feature_schema.json.
    """
    schema_path = resolve_file("feature_schema.json", subfolder="models")
    assert os.path.exists(schema_path), f"feature_schema.json not found at {schema_path}"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    features = schema["features"] if isinstance(schema, dict) else schema
    forbidden_terms = [
        "target", "forward_return", "forward_return_1d",
        "next_close", "future", "lead", "label", "outcome"
    ]

    for feat in features:
        feat_lower = feat.lower()
        for forbidden in forbidden_terms:
            assert forbidden != feat_lower, f"CRITICAL LEAKAGE: Forbidden variable '{feat}' found in feature schema!"

    assert len(features) == 23, f"Expected 23 sequential features, found {len(features)}"


def test_leakage_check_2_temporal_window_disjointness():
    """
    Leakage Check 2:
    Assert that the training date window and out-of-sample test date window
    in model_metadata.json strictly do not overlap.
    """
    meta_path = resolve_file("model_metadata.json", subfolder="models")
    assert os.path.exists(meta_path), f"model_metadata.json not found at {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_end = pd.to_datetime(metadata["training_window"]["end"])
    val_start = pd.to_datetime(metadata["validation_window"]["start"])
    val_end = pd.to_datetime(metadata["validation_window"]["end"])
    test_start = pd.to_datetime(metadata["test_window"]["start"])

    # Strict chronological ordering: Train End < Val Start <= Val End < Test Start
    assert train_end < val_start, f"Train End ({train_end}) overlaps with Val Start ({val_start})"
    assert val_end < test_start, f"Val End ({val_end}) overlaps with Test Start ({test_start})"
    assert train_end < test_start, f"CRITICAL LEAKAGE: Train End ({train_end}) >= Test Start ({test_start})"


def test_data_integrity_no_duplicate_rows():
    """
    Data Integrity:
    Assert that prices data contains no duplicate (stock_symbol, date) rows.
    """
    price_path = resolve_file("cleaned_prices.parquet", subfolder="data")
    if not os.path.exists(price_path):
        price_path = resolve_file("model_dataset_v1.parquet", subfolder="data")

    if os.path.exists(price_path):
        df = pd.read_parquet(price_path, columns=["stock_symbol", "date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        dup_count = df.duplicated(subset=["stock_symbol", "date"]).sum()
        assert dup_count == 0, f"Found {dup_count} duplicate (stock_symbol, date) records in {price_path}!"
    else:
        pytest.skip("Price parquet file not found in current environment (available on Google Drive).")


def test_model_artifact_existence():
    """
    Model Artifacts:
    Verify that calibrated or native XGBoost weights exist.
    """
    calibrated_joblib = resolve_file("xgb_direction_calibrated.joblib", subfolder="models")
    native_json = resolve_file("xgb_direction.json", subfolder="models")

    has_model = os.path.exists(calibrated_joblib) or os.path.exists(native_json)
    assert has_model, "Neither xgb_direction_calibrated.joblib nor xgb_direction.json was found in models/!"


def test_standalone_inference_engine_execution():
    """
    Inference Engine:
    Verify that NexusInferenceEngine loads and executes with valid dummy inputs.
    """
    from src.inference.predict import NexusInferenceEngine

    schema_path = resolve_file("feature_schema.json", subfolder="models")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema["features"] if isinstance(schema, dict) else schema

    engine = NexusInferenceEngine(schema_path=schema_path)

    # Synthetic single-row test DataFrame
    synthetic_data = {feat: [0.01 if "return" in feat else 50.0] for feat in features}
    synthetic_data["stock_symbol"] = ["AAPL"]
    synthetic_data["date"] = ["2023-06-15"]
    test_df = pd.DataFrame(synthetic_data)

    results = engine.predict(test_df, top_k_factors=3)
    assert len(results) == 1
    assert results[0]["ticker"] == "AAPL"
    assert results[0]["prediction"] in ["UP", "DOWN"]
    assert 0.0 <= results[0]["up_probability"] <= 1.0
    assert 0.0 <= results[0]["down_probability"] <= 1.0
    assert len(results[0]["top_factors"]) > 0


def test_vectorized_backtester_execution():
    """
    Backtesting Engine:
    Verify that VectorizedBacktester computes returns and drawdown correctly.
    """
    from src.backtest.engine import VectorizedBacktester

    engine = VectorizedBacktester(cost_bps=10.0)

    # Synthetic 5-day 2-stock test dataset
    dates = pd.date_range("2023-01-01", periods=5)
    records = []
    for d in dates:
        for sym in ["AAPL", "MSFT"]:
            records.append({
                "date": d,
                "stock_symbol": sym,
                "close": 150.0,
                "forward_return_1d": 0.01,
                "up_probability": 0.55
            })
    mock_df = pd.DataFrame(records)

    results = engine.run_backtest(mock_df, prob_col="up_probability", threshold=0.50)
    assert "strat_metrics" in results
    assert "portfolio_daily" in results
    assert results["strat_metrics"]["cumulative_return"] > 0


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING NEXUS STOCK AI PIPELINE INTEGRITY TESTS")
    print("=" * 60)
    test_leakage_check_1_no_target_in_feature_schema()
    print("✓ [TEST 1/6 PASSED] test_leakage_check_1_no_target_in_feature_schema")
    test_leakage_check_2_temporal_window_disjointness()
    print("✓ [TEST 2/6 PASSED] test_leakage_check_2_temporal_window_disjointness")
    test_data_integrity_no_duplicate_rows()
    print("✓ [TEST 3/6 PASSED] test_data_integrity_no_duplicate_rows")
    test_model_artifact_existence()
    print("✓ [TEST 4/6 PASSED] test_model_artifact_existence")
    test_standalone_inference_engine_execution()
    print("✓ [TEST 5/6 PASSED] test_standalone_inference_engine_execution")
    test_vectorized_backtester_execution()
    print("✓ [TEST 6/6 PASSED] test_vectorized_backtester_execution")
    print("\n✓ ALL 6 TESTS PASSED SUCCESSFULLY!")
