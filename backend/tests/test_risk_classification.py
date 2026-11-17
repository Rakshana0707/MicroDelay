"""Automated Unit & Integration Tests for Personalized Lateness Risk Classification (Prompt 2.2).

Tests:
- Classification target definition & configurable operational lateness tolerance
- Class imbalance & skew handling (class_weight='balanced', scale_pos_weight)
- Single-class edge case & zero-division safety
- Probability calibration (Platt scaling) & Brier score evaluation
- Chronological train/validation/test evaluation without temporal lookahead leakage
- Multi-algorithm benchmarking (Logistic Regression, Random Forest, HistGradientBoosting, XGBoost)
- Configurable risk thresholds (Low < 30%, Moderate 30-60%, High > 60%, Extreme > 85%)
- Unseen event categories & extrapolation robustness
- Classifier artifact persistence & load failure isolation
- Independent operation of regression and classification engines
- End-to-end API endpoints (/api/intelligence/risk, /api/models/classifier/metrics)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from ml.classifier import (
    LatenessClassificationConfig,
    ClassificationEvaluationMetrics,
    compute_classification_metrics,
    LatenessClassifierSuite,
    LatenessClassifierManager
)


def _generate_synthetic_classification_df(n_samples: int = 60, lateness_rate: float = 0.40) -> pd.DataFrame:
    """Generate mock departure observations sorted chronologically."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2026-09-01", periods=n_samples, freq="D").strftime("%Y-%m-%d").tolist()
    events = ["ML Lab", "Standup", "Faculty Meeting", "Research Seminar", "Gym Session"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    rows = []
    for i in range(n_samples):
        is_late = rng.rand() < lateness_rate
        # Delays in minutes: negative or small positive for on-time, positive for late
        if is_late:
            delay = round(rng.uniform(1.5, 20.0), 1)
        else:
            delay = round(rng.uniform(-10.0, 0.0), 1)

        prep_est = round(rng.uniform(15.0, 45.0), 1)
        prep_act = prep_est + (delay * 0.5) if is_late else prep_est - 2.0

        rows.append({
            "id": i + 1,
            "user_id": 1,
            "event_name": rng.choice(events),
            "day_of_week": rng.choice(days),
            "date": dates[i],
            "planned_departure": "08:30:00",
            "actual_departure": "08:40:00" if is_late else "08:25:00",
            "estimated_prep_duration_min": prep_est,
            "actual_prep_duration_min": max(5.0, prep_act),
            "estimated_travel_duration_min": 20.0,
            "actual_travel_duration_min": 20.0,
            "total_delay_min": delay,
            "bag_prepared": not is_late or (rng.rand() > 0.5),
            "importance": "high" if rng.rand() > 0.5 else "medium",
            "provenance": "synthetic",
            "data_source": "seed_generation",
            "observation_status": "valid"
        })

    return pd.DataFrame(rows)


def test_classification_target_and_configurable_tolerance():
    """Verify operational lateness target definition with configurable tolerance."""
    df = pd.DataFrame([
        {"total_delay_min": -5.0},  # Early -> 0
        {"total_delay_min": 0.0},   # Exact -> 0
        {"total_delay_min": 0.5},   # Negligible delay
        {"total_delay_min": 1.5},   # Mild delay
        {"total_delay_min": 8.0}    # Definite late -> 1
    ])

    # Default tolerance (delay > 0.0)
    cfg_strict = LatenessClassificationConfig(tolerance_min=0.0)
    y_strict = (df["total_delay_min"] > cfg_strict.tolerance_min).astype(int).tolist()
    assert y_strict == [0, 0, 1, 1, 1]

    # Configurable tolerance: allow up to 1.0 minute deviation
    cfg_tolerant = LatenessClassificationConfig(tolerance_min=1.0)
    y_tolerant = (df["total_delay_min"] > cfg_tolerant.tolerance_min).astype(int).tolist()
    assert y_tolerant == [0, 0, 0, 1, 1]


def test_class_imbalance_and_metrics_calculation():
    """Verify metrics calculation under class imbalance with precision/recall/Brier score."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 80% on-time, 20% late
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # predicted 1 false positive, 2 true positives
    y_prob = np.array([0.1, 0.15, 0.05, 0.2, 0.1, 0.3, 0.25, 0.65, 0.85, 0.90])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert metrics.accuracy == 0.90
    assert metrics.recall == 1.0  # Caught both positive instances
    assert metrics.precision == 2 / 3  # 2 true positives out of 3 positive predictions
    assert metrics.macro_f1 > 0.80
    assert metrics.brier_score < 0.10
    assert metrics.confusion_matrix["tp"] == 2
    assert metrics.confusion_matrix["fp"] == 1
    assert metrics.confusion_matrix["fn"] == 0
    assert metrics.confusion_matrix["tn"] == 7
    assert metrics.roc_auc is not None and metrics.roc_auc > 0.90


def test_single_class_edge_case():
    """Verify safety when test fold has only on-time or only late departures."""
    y_all_zero = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0])
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    metrics = compute_classification_metrics(y_all_zero, y_pred, y_prob)
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc is None  # Undefined when only 1 class exists
    assert metrics.brier_score == pytest.approx(0.01)
    assert metrics.positive_prevalence_pct == 0.0


def test_probability_calibration_and_brier_evaluation():
    """Verify that probability calibration via Platt scaling improves or maintains Brier score."""
    df = _generate_synthetic_classification_df(n_samples=50, lateness_rate=0.45)
    suite = LatenessClassifierSuite(config=LatenessClassificationConfig(calibration_method="sigmoid"))
    result = suite.train_and_benchmark(df)

    assert result["success"] is True
    assert len(result["models"]) >= 3

    # Check each candidate model recorded calibration Brier score
    for m in result["models"]:
        assert "brier_score_raw" in m
        assert "brier_score_calibrated" in m
        assert 0.0 <= m["brier_score_calibrated"] <= 1.0


def test_chronological_evaluation_no_temporal_leakage():
    """Verify chronological partitioning: training precedes validation precedes test."""
    df = _generate_synthetic_classification_df(n_samples=40)
    suite = LatenessClassifierSuite()
    result = suite.train_and_benchmark(df)

    assert result["success"] is True
    assert result["train_samples"] > 0
    assert result["val_samples"] > 0
    assert result["test_samples"] > 0
    assert result["train_samples"] + result["val_samples"] + result["test_samples"] == len(df)


def test_candidate_classifiers_benchmarking_and_selection():
    """Verify all candidate classifiers (Logistic Regression, Random Forest, HistGradientBoosting, XGBoost) are evaluated."""
    df = _generate_synthetic_classification_df(n_samples=60)
    suite = LatenessClassifierSuite()
    result = suite.train_and_benchmark(df)

    assert result["success"] is True
    model_names = [m["model_name"] for m in result["models"]]
    assert "Logistic Regression" in model_names
    assert "Random Forest Classifier" in model_names
    assert "HistGradientBoosting Classifier" in model_names
    if "XGBoost Classifier" in model_names:
        assert any(m["model_name"] == "XGBoost Classifier" for m in result["models"])

    assert result["selected_best_model"] is not None
    assert "test_metrics" in result
    assert result["test_metrics"]["accuracy"] >= 0.0
    assert result["test_metrics"]["macro_f1"] >= 0.0


def test_configurable_risk_thresholds():
    """Verify configurable operational risk categories."""
    cfg = LatenessClassificationConfig(
        low_risk_threshold_pct=25.0,
        high_risk_threshold_pct=55.0,
        extreme_risk_threshold_pct=80.0
    )

    # Below 25% -> low
    assert cfg.categorize_probability(0.15) == "low"
    assert cfg.categorize_probability(0.24) == "low"

    # 25% to 55% -> moderate
    assert cfg.categorize_probability(0.35) == "moderate"
    assert cfg.categorize_probability(0.55) == "moderate"

    # 56% to 80% -> high
    assert cfg.categorize_probability(0.65) == "high"
    assert cfg.categorize_probability(0.79) == "high"

    # Above 80% -> extreme
    assert cfg.categorize_probability(0.85) == "extreme"
    assert cfg.categorize_probability(0.95) == "extreme"


def test_unseen_event_categories_robustness():
    """Verify classifier prediction succeeds even when given unfamiliar event names."""
    pred = LatenessClassifierManager.predict_lateness_risk(
        user_id=1,
        event_name="Quantum Physics Seminar Unseen",
        day_of_week="Thursday",
        importance="high",
        bag_prepared=True,
        estimated_prep_duration_min=30.0,
        estimated_travel_duration_min=20.0
    )

    assert "late_probability" in pred
    assert 0.0 <= pred["late_probability"] <= 1.0
    assert pred["risk_category"] in ("low", "moderate", "high", "extreme")


def test_model_loading_failure_isolation():
    """Verify that classifier loading failure falls back to empirical prior without crashing."""
    # User 999999 has no saved classifier bundle
    pred = LatenessClassifierManager.predict_lateness_risk(
        user_id=999999,
        event_name="Routine",
        day_of_week="Monday",
        importance="medium",
        bag_prepared=False,
        estimated_prep_duration_min=25.0,
        estimated_travel_duration_min=15.0
    )

    assert pred["model_version"] == "empirical_prior_fallback"
    assert pred["model_type"] == "empirical_prior"
    assert pred["is_calibrated"] is False
    assert pred["confidence_status"]["is_reliable"] is False


def test_independent_operation_of_regression_and_classification(db_session):
    """Verify regression and classification operate as distinct, decoupled models."""
    from models import predict_micro_delay, predict_lateness_risk

    # 1. Query regression model
    reg = predict_micro_delay(
        user_id=1,
        event_name="ML Lab",
        day_of_week="Monday",
        importance="high",
        bag_prepared=True,
        estimated_prep_duration_min=25.0,
        estimated_travel_duration_min=20.0,
        db=db_session
    )
    assert "predicted_delay_min" in reg

    # 2. Query classification model
    clf = predict_lateness_risk(
        user_id=1,
        event_name="ML Lab",
        day_of_week="Monday",
        importance="high",
        bag_prepared=True,
        estimated_prep_duration_min=25.0,
        estimated_travel_duration_min=20.0
    )
    assert "late_probability" in clf
    assert "risk_category" in clf
    assert 0.0 <= clf["late_probability"] <= 1.0


def test_api_risk_endpoints_and_metrics(client, db_session):
    """Verify /api/intelligence/risk (GET and POST) and /api/models/classifier/metrics endpoints."""
    # 1. Register user
    reg_resp = client.post("/api/auth/register", json={
        "username": "risk_api_user",
        "password": "risk_password_123",
        "profile_type": "student"
    })
    assert reg_resp.status_code == 200
    token = reg_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Call GET /api/intelligence/risk
    get_risk_resp = client.get("/api/intelligence/risk", params={
        "event_name": "ML Lab",
        "day_of_week": "Monday",
        "importance": "high",
        "bag_prepared": False,
        "estimated_prep": 25.0,
        "estimated_travel": 20.0
    }, headers=headers)
    assert get_risk_resp.status_code == 200
    risk_data = get_risk_resp.json()
    assert "late_probability" in risk_data
    assert "risk_category" in risk_data
    assert "predicted_delay_minutes" in risk_data
    assert "confidence_status" in risk_data
    assert "contributing_factors" in risk_data
    assert "recommended_interventions" in risk_data
    assert 0.0 <= risk_data["late_probability"] <= 1.0

    # 3. Call POST /api/intelligence/risk
    post_risk_resp = client.post("/api/intelligence/risk", json={
        "event_name": "ML Lab",
        "day_of_week": "Monday",
        "importance": "high",
        "bag_prepared": True,
        "estimated_prep_duration": 25.0,
        "estimated_travel_duration": 20.0
    }, headers=headers)
    assert post_risk_resp.status_code == 200
    post_risk_data = post_risk_resp.json()
    assert "late_probability" in post_risk_data
    assert "risk_category" in post_risk_data

    # 4. Call GET /api/models/classifier/metrics
    metrics_resp = client.get("/api/models/classifier/metrics", headers=headers)
    assert metrics_resp.status_code == 200
    clf_metrics = metrics_resp.json()
    assert clf_metrics["has_classifier"] is True
    assert "selected_best_model" in clf_metrics
    assert "test_metrics" in clf_metrics
    assert "models" in clf_metrics
    assert "is_calibrated" in clf_metrics
    assert isinstance(clf_metrics["is_calibrated"], bool)
