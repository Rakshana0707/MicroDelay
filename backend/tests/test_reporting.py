"""Automated tests for evaluation metrics, residual diagnostics, and reporting in MicroDelay."""

import os
import json
import pytest
import numpy as np
import pandas as pd

from ml.reporting import (
    calculate_standardized_metrics,
    perform_residual_analysis,
    segment_performance,
    determine_model_status,
    ModelEvaluationStatus,
    EvaluationReporter,
    REAL_REPORTS_DIR,
    SYNTHETIC_REPORTS_DIR,
)


def test_calculate_standardized_metrics():
    """Verify calculation of MAE, RMSE, MedAE, tolerance windows, and baseline improvement."""
    y_true = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    y_pred = np.array([1.0, 6.0, 9.0, 14.0, 26.0, 31.0])
    # Absolute errors: [1, 1, 1, 1, 6, 6]
    # Within 5m: 4/6 = 66.67%
    # Within 10m: 6/6 = 100.0%
    # MAE = (1+1+1+1+6+6)/6 = 16/6 = 2.667

    metrics = calculate_standardized_metrics(y_true, y_pred, baseline_mae=5.0)

    assert metrics["sample_count"] == 6
    assert np.isclose(metrics["mae"], 2.667, atol=1e-3)
    assert np.isclose(metrics["medae"], 1.0, atol=1e-3)
    assert np.isclose(metrics["tolerance_5m_pct"], 66.67, atol=0.1)
    assert np.isclose(metrics["tolerance_10m_pct"], 100.0)

    # Baseline improvement: ((5.0 - 2.667) / 5.0) * 100 = 46.66%
    assert metrics["baseline_improvement_pct"] > 45.0

    # Zero baseline MAE handling: should return 0.0 without division by zero
    zero_base_metrics = calculate_standardized_metrics(y_true, y_pred, baseline_mae=0.0)
    assert zero_base_metrics["baseline_improvement_pct"] == 0.0


def test_r2_guardrails_small_sample_and_zero_variance():
    """Verify that R² is flagged as not meaningful on small sample sizes (<5) or zero variance."""
    # 1. Sample count < 5
    small_true = np.array([2.0, 4.0, 6.0])
    small_pred = np.array([2.1, 3.9, 6.2])
    m_small = calculate_standardized_metrics(small_true, small_pred)
    assert m_small["r2_meaningful"] is False
    assert m_small["r2"] is None
    assert "too small" in m_small["r2_reason"]

    # 2. Zero target variance (all targets = 5.0m delay)
    zero_var_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    zero_var_pred = np.array([4.8, 5.1, 4.9, 5.2, 5.0, 4.9])
    m_zero_var = calculate_standardized_metrics(zero_var_true, zero_var_pred)
    assert m_zero_var["r2_meaningful"] is False
    assert m_zero_var["r2"] is None
    assert "zero or near-zero variance" in m_zero_var["r2_reason"]

    # 3. Valid sample count (>= 5) and valid variance
    valid_true = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    valid_pred = np.array([1.0, 4.0, 11.0, 14.0, 21.0, 24.0])
    m_valid = calculate_standardized_metrics(valid_true, valid_pred)
    assert m_valid["r2_meaningful"] is True
    assert m_valid["r2"] is not None
    assert m_valid["r2"] > 0.85


def test_perform_residual_analysis():
    """Test residual diagnostics: bias, over/underprediction, large errors, and condition flags."""
    y_true = np.array([5.0, 10.0, 2.0, 8.0, 4.0, 15.0])
    y_pred = np.array([8.0, 7.0, 2.2, 9.5, 4.1, 30.0])
    # Residuals (pred - true): [+3.0, -3.0, +0.2, +1.5, +0.1, +15.0]

    df = pd.DataFrame({
        "event_name": ["ML Lab", "ML Lab", "Seminar", "Seminar", "Library Study", "Library Study"],
        "day_of_week": ["Monday", "Monday", "Tuesday", "Tuesday", "Friday", "Friday"],
        "importance": ["high", "high", "medium", "medium", "low", "high"],
        "bag_prepared": [0, 0, 1, 1, 1, 0]
    })

    res = perform_residual_analysis(y_true, y_pred, df=df)

    assert "mean_signed_bias" in res
    assert res["overpredictions"]["count"] == 3  # +3.0, +1.5, +15.0
    assert res["underpredictions"]["count"] == 1  # -3.0
    assert res["exact_predictions"]["count"] == 2  # +0.2, +0.1 (within +/- 0.5)

    # Large error (>10m): index 5 (+15.0)
    assert res["large_errors"]["count"] == 1
    assert len(res["large_errors"]["sample_cases"]) == 1
    assert res["large_errors"]["sample_cases"][0]["actual_delay"] == 15.0
    assert res["large_errors"]["sample_cases"][0]["predicted_delay"] == 30.0

    # Error distribution
    assert res["error_distribution"]["max"] == 15.0
    assert res["error_distribution"]["min"] == -3.0


def test_segment_performance():
    """Test multi-dimensional error segmentation across days, events, importance, and prep duration."""
    df = pd.DataFrame({
        "day_of_week": ["Monday", "Monday", "Tuesday", "Wednesday"],
        "event_name": ["ML Lab", "ML Lab", "Seminar", "Math Lecture"],
        "importance": ["high", "high", "medium", "low"],
        "bag_prepared": [0, 0, 1, 1],
        "estimated_prep_duration_min": [10.0, 20.0, 35.0, 50.0]
    })
    y_true = np.array([10.0, 5.0, 2.0, 4.0])
    y_pred = np.array([8.0, 6.0, 2.0, 3.0])

    seg = segment_performance(df, y_true, y_pred)

    assert "day_of_week" in seg
    assert "event_name" in seg
    assert "importance" in seg
    assert "bag_prepared" in seg
    assert "prep_duration_bucket" in seg

    # Check that buckets are present
    prep_buckets = [b["segment"] for b in seg["prep_duration_bucket"]]
    assert "< 15m" in prep_buckets or "15 - 30m" in prep_buckets


def test_determine_model_status_state_machine():
    """Test transitions of the model evaluation status state machine."""
    # 1. Insufficient data
    status, reason = determine_model_status(
        train_count=5, is_validated=False, baseline_mae=5.0, model_mae=4.0, min_samples=8
    )
    assert status == ModelEvaluationStatus.INSUFFICIENT_DATA
    assert "Insufficient observations" in reason

    # 2. Training completed but not validated
    status, reason = determine_model_status(
        train_count=20, is_validated=False, baseline_mae=5.0, model_mae=None
    )
    assert status == ModelEvaluationStatus.TRAINING_COMPLETED

    # 3. Model underperforms baseline
    status, reason = determine_model_status(
        train_count=20, is_validated=True, baseline_mae=4.0, model_mae=6.5
    )
    assert status == ModelEvaluationStatus.MODEL_UNDERPERFORMS_BASELINE
    assert "underperformed" in reason

    # 4. Large error rate high (>40%)
    status, reason = determine_model_status(
        train_count=20, is_validated=True, baseline_mae=5.0, model_mae=3.0,
        large_error_pct=45.0
    )
    assert status == ModelEvaluationStatus.EVALUATION_COMPLETED

    # 5. Model approved
    status, reason = determine_model_status(
        train_count=20, is_validated=True, baseline_mae=5.0, model_mae=3.0,
        tolerance_5m_pct=85.0, large_error_pct=5.0
    )
    assert status == ModelEvaluationStatus.MODEL_APPROVED
    assert "approved" in reason


def test_evaluation_reporter_artifacts_generation(tmp_path):
    """Test JSON and Markdown report artifact generation with directory separation."""
    reporter = EvaluationReporter(user_id=777, model_version="2.0.0", provenance="synthetic")

    metrics = {
        "sample_count": 10,
        "mae": 3.12,
        "rmse": 4.05,
        "r2": 0.62,
        "r2_meaningful": True,
        "r2_reason": "Valid",
        "medae": 2.5,
        "tolerance_5m_pct": 80.0,
        "tolerance_10m_pct": 100.0,
        "baseline_improvement_pct": 25.5
    }

    residual_analysis = {
        "mean_signed_bias": 0.45,
        "bias_characterization": "Unbiased (near zero)",
        "overpredictions": {"count": 5, "percentage": 50.0, "mean_overprediction_min": 2.1},
        "underpredictions": {"count": 3, "percentage": 30.0, "mean_underprediction_min": 1.8},
        "exact_predictions": {"count": 2, "percentage": 20.0},
        "large_errors": {"count": 0, "percentage": 0.0, "sample_cases": []},
        "error_distribution": {"min": -2.0, "q25": -0.5, "median": 0.2, "q75": 1.5, "max": 3.0, "std": 1.4, "skewness": 0.1},
        "poor_performing_conditions": []
    }

    segmentation = {
        "day_of_week": [{"segment": "Monday", "count": 5, "mae": 3.0, "rmse": 3.8, "tolerance_5m_pct": 80.0}]
    }

    json_path, md_path = reporter.generate_report(
        metrics=metrics,
        residual_analysis=residual_analysis,
        segmentation=segmentation,
        model_name="RandomForestRegressor",
        hyperparameters={"n_estimators": 50, "max_depth": 5},
        training_metadata={"train_count": 10},
        evaluation_status=ModelEvaluationStatus.MODEL_APPROVED,
        status_reason="Outperformed baseline."
    )

    # 1. Verify JSON file
    assert os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["user_id"] == 777
    assert data["metadata"]["is_demonstration_only"] is True
    assert data["metrics"]["mae"] == 3.12

    # 2. Verify Markdown file
    assert os.path.exists(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "# MicroDelay ML Model Evaluation Report" in md_text
    assert "SYNTHETIC DEMONSTRATION DATA" in md_text
    assert "3.12 min" in md_text
    assert "MODEL_APPROVED" in md_text


def test_api_latest_report_endpoint(client, auth_headers, test_user, db_session):
    """Test GET /api/reports/latest returns the latest evaluation report for the user."""
    from database import DepartureLog, DataProvenance, DataSource
    import datetime

    # Seed 15 demo departure logs so model can be trained and evaluated
    base_date = datetime.date(2026, 8, 1)
    for i in range(15):
        curr_date = base_date + datetime.timedelta(days=i)
        log = DepartureLog(
            user_id=test_user.id,
            event_name="ML Lab" if i % 2 == 0 else "Seminar",
            date=curr_date.isoformat(),
            day_of_week=curr_date.strftime("%A"),
            planned_prep_start="08:00:00",
            planned_departure="08:20:00",
            estimated_prep_duration_min=20.0,
            estimated_travel_duration_min=20.0,
            actual_prep_start="08:00:00",
            actual_departure="08:25:00",
            actual_arrival="08:45:00",
            actual_prep_duration_min=25.0,
            actual_travel_duration_min=20.0,
            total_delay_min=5.0,
            bag_prepared=True,
            provenance=DataProvenance.SYNTHETIC.value,
            data_source=DataSource.GENERATOR.value,
            is_synthetic=True
        )
        db_session.add(log)
    db_session.commit()

    import models
    res = models.train_and_evaluate(db_session, test_user.id, training_mode="synthetic_only")
    assert res["success"] is True

    # Query latest report
    resp = client.get("/api/reports/latest?provenance=synthetic", headers=auth_headers)
    assert resp.status_code == 200
    report_json = resp.json()

    assert report_json["metadata"]["user_id"] == test_user.id
    assert "metrics" in report_json
    assert "residual_analysis" in report_json
    assert "segmentation" in report_json
    assert report_json["metadata"]["evaluation_status"] in [
        ModelEvaluationStatus.MODEL_APPROVED,
        ModelEvaluationStatus.EVALUATION_COMPLETED,
        ModelEvaluationStatus.MODEL_UNDERPERFORMS_BASELINE
    ]
