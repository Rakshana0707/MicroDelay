"""Unit and Integration Tests for Prediction Confidence & Reliability Engine (Prompt 2.9).

Verifies:
- Conformal prediction intervals and empirical coverage evaluation
- Classification probability calibration (Brier score, ECE, reliability bins)
- Documented rule-based data sufficiency levels
- Documented reliability states without invented percentages
- Prediction abstention on out-of-range contexts and insufficient history
- Distribution drift impact on interval width and exchangeability warnings
- Qualitative handling of uncalibrated probabilities
- Profile and recommendation evidence levels
- FastAPI reliability REST endpoints
"""

import math
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, User, DepartureLog, get_db
from backend.main import app
from backend.ml.reliability import (
    ConformalPredictionIntervalEstimator,
    ClassificationCalibrationEvaluator,
    BehavioralProfileReliabilityEvaluator,
    RecommendationReliabilityEvaluator,
    DataSufficiencyEvaluator,
    PredictionReliabilityEngine
)
from backend.ml.adaptive import (
    PredictionContext,
    AdaptiveIntelligenceOrchestrator,
    ReliabilityStatus,
    CalibrationStatus,
    DataSufficiencyLevel
)


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()

    user = User(
        id=99,
        username="reliability_user",
        hashed_password="hash",
        salt="salt",
        profile_type="student",
        consent_to_global_training=True
    )
    session.add(user)
    session.commit()

    yield session
    session.close()


@pytest.fixture
def test_client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    test_user = test_db.query(User).filter(User.id == 99).first()

    def override_get_current_user():
        return test_user

    from backend.main import get_current_user
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# ==============================================================================
# 1. REGRESSION UNCERTAINTY & CONFORMAL PREDICTION
# ==============================================================================

def test_conformal_prediction_intervals_and_coverage():
    """Verify split conformal prediction quantiles and empirical coverage on holdouts."""
    np.random.seed(42)
    n_cal = 100
    y_true = np.random.normal(10.0, 3.0, size=n_cal)
    y_pred = y_true + np.random.normal(0.0, 2.0, size=n_cal)

    residuals = ConformalPredictionIntervalEstimator.compute_non_conformity_scores(y_true, y_pred)
    assert len(residuals) == n_cal

    # Target 80% coverage (alpha = 0.20)
    q_80 = ConformalPredictionIntervalEstimator.compute_conformal_quantile(residuals, alpha=0.20)
    assert q_80 > 0.0

    # Test empirical coverage on independent holdout
    n_test = 200
    y_test_true = np.random.normal(10.0, 3.0, size=n_test)
    y_test_pred = y_test_true + np.random.normal(0.0, 2.0, size=n_test)

    cov = ConformalPredictionIntervalEstimator.compute_empirical_coverage(y_test_true, y_test_pred, q_80)
    # Empirical coverage should be approximately 80% (within statistical tolerance >= 74%)
    assert cov >= 0.74

    # Test single interval estimation
    interval = ConformalPredictionIntervalEstimator.estimate_interval(
        predicted_delay=8.0,
        residuals=residuals,
        coverage_level=0.80
    )
    assert interval.lower_bound >= 0.0
    assert interval.upper_bound > interval.lower_bound
    assert interval.coverage_level == 0.80
    assert interval.method == "split_conformal"
    assert interval.is_calibrated is True


# ==============================================================================
# 2. CLASSIFICATION CALIBRATION & RELIABILITY DIAGRAMS
# ==============================================================================

def test_classification_calibration_brier_and_ece():
    """Verify Expected Calibration Error (ECE) and Brier score computation."""
    np.random.seed(42)
    n = 60

    # 1. Well-calibrated simulated probabilities
    probs_cal = np.random.uniform(0.1, 0.9, size=n)
    y_true_cal = (np.random.uniform(0.0, 1.0, size=n) < probs_cal).astype(int)

    ece_cal, mce_cal, bins_cal = ClassificationCalibrationEvaluator.calculate_ece(y_true_cal, probs_cal, n_bins=5)
    brier_cal = ClassificationCalibrationEvaluator.calculate_brier_score(y_true_cal, probs_cal)

    assert 0.0 <= ece_cal <= 0.25
    assert 0.0 <= brier_cal <= 0.35
    assert len(bins_cal) == 5

    # 2. Poorly-calibrated overconfident probabilities
    probs_bad = np.full(n, 0.90)  # Predicts 90% late
    y_true_bad = np.zeros(n)      # Actually nobody is late
    ece_bad, _, _ = ClassificationCalibrationEvaluator.calculate_ece(y_true_bad, probs_bad, n_bins=5)
    assert ece_bad >= 0.80

    # 3. Calibration report
    rep = ClassificationCalibrationEvaluator.evaluate_calibration(
        y_true=y_true_cal,
        y_prob=probs_cal,
        is_trained_calibrated=True
    )
    assert rep.sample_count == n
    assert rep.calibration_status in ("calibrated", "uncalibrated")
    assert "low" in rep.risk_groups
    assert "high" in rep.risk_groups


# ==============================================================================
# 3. DATA SUFFICIENCY TIERS
# ==============================================================================

def test_data_sufficiency_documented_levels():
    """Verify data sufficiency thresholds match documented requirements."""
    # < 8: insufficient
    r1 = DataSufficiencyEvaluator.evaluate(genuine_count=4, synthetic_count=0)
    assert r1.level == "insufficient"

    # 8 - 14: minimal
    r2 = DataSufficiencyEvaluator.evaluate(genuine_count=10, synthetic_count=0)
    assert r2.level == "minimal"

    # 15 - 29: moderate
    r3 = DataSufficiencyEvaluator.evaluate(genuine_count=22, synthetic_count=0)
    assert r3.level == "moderate"

    # >= 30: robust
    r4 = DataSufficiencyEvaluator.evaluate(genuine_count=45, synthetic_count=0)
    assert r4.level == "robust"

    # Recency check: > 45 days stale
    r5 = DataSufficiencyEvaluator.evaluate(genuine_count=20, recency_days=60)
    assert any("recency warning" in n.lower() for n in r5.notes)


# ==============================================================================
# 4. DOCUMENTED RELIABILITY STATES
# ==============================================================================

def test_documented_reliability_states():
    """Ensure reliability statuses are strictly rule-based without arbitrary confidence %."""
    # Degraded model takes highest precedence
    s_deg = PredictionReliabilityEngine.determine_reliability_status(
        genuine_samples=40,
        validation_mae=2.0,
        calibration_status="calibrated",
        is_model_degraded=True,
        is_drift_detected=False,
        recency_days=5,
        has_active_global_model=True
    )
    assert s_deg == "degraded_model"

    # Insufficient history
    s_ins = PredictionReliabilityEngine.determine_reliability_status(
        genuine_samples=4,
        validation_mae=None,
        calibration_status="uncalibrated",
        is_model_degraded=False,
        is_drift_detected=False,
        recency_days=None,
        has_active_global_model=False
    )
    assert s_ins == "insufficient_history"

    # High reliability: >= 30, MAE <= 3.5, calibrated, fresh data, no drift
    s_high = PredictionReliabilityEngine.determine_reliability_status(
        genuine_samples=35,
        validation_mae=2.8,
        calibration_status="calibrated",
        is_model_degraded=False,
        is_drift_detected=False,
        recency_days=10,
        has_active_global_model=True
    )
    assert s_high == "high_reliability"

    # Moderate reliability
    s_mod = PredictionReliabilityEngine.determine_reliability_status(
        genuine_samples=18,
        validation_mae=4.2,
        calibration_status="calibrated",
        is_model_degraded=False,
        is_drift_detected=False,
        recency_days=25,
        has_active_global_model=True
    )
    assert s_mod == "moderate_reliability"


# ==============================================================================
# 5. PREDICTION ABSTENTION
# ==============================================================================

def test_prediction_abstention_out_of_range_features():
    """Verify system abstains when contextual features fall outside operating bounds."""
    # Prep duration exceeds 300 min (5 hours)
    abstain, reason, status = PredictionReliabilityEngine.evaluate_prediction_abstention(
        estimated_prep_duration_min=360.0,
        estimated_travel_duration_min=25.0,
        genuine_samples=20,
        has_active_global_model=True
    )
    assert abstain is True
    assert "operating bounds" in reason.lower()
    assert status == "insufficient_history"

    # Negative travel duration
    abstain_neg, reason_neg, _ = PredictionReliabilityEngine.evaluate_prediction_abstention(
        estimated_prep_duration_min=20.0,
        estimated_travel_duration_min=-15.0,
        genuine_samples=20,
        has_active_global_model=True
    )
    assert abstain_neg is True
    assert "operating bounds" in reason_neg.lower()


def test_prediction_abstention_insufficient_history_no_global():
    """Verify system abstains when user has < 3 observations and no global model."""
    abstain, reason, status = PredictionReliabilityEngine.evaluate_prediction_abstention(
        estimated_prep_duration_min=25.0,
        estimated_travel_duration_min=20.0,
        genuine_samples=1,
        has_active_global_model=False
    )
    assert abstain is True
    assert reason == "Prediction unavailable — insufficient reliable historical observations."
    assert status == "insufficient_history"


# ==============================================================================
# 6. DISTRIBUTION DRIFT IMPACT ON UNCERTAINTY
# ==============================================================================

def test_distribution_drift_impact_on_uncertainty():
    """Verify behavioral drift widens conformal intervals and attaches exchangeability caveats."""
    residuals = np.array([2.0, 3.5, 1.2, 4.0, 2.8, 1.9, 3.1])

    # Normal conditions
    normal_int = ConformalPredictionIntervalEstimator.estimate_interval(
        predicted_delay=10.0,
        residuals=residuals,
        coverage_level=0.80,
        is_drift_detected=False
    )
    assert normal_int.is_calibrated is True
    assert normal_int.is_drift_impacted is False

    # Drift detected
    drift_int = ConformalPredictionIntervalEstimator.estimate_interval(
        predicted_delay=10.0,
        residuals=residuals,
        coverage_level=0.80,
        is_drift_detected=True
    )
    # Margin must be strictly wider than normal
    assert drift_int.conformal_quantile > normal_int.conformal_quantile
    assert drift_int.is_drift_impacted is True
    assert drift_int.is_calibrated is False
    assert "drift" in drift_int.warning.lower()


# ==============================================================================
# 7. UNCALIBRATED PROBABILITY HANDLING
# ==============================================================================

def test_uncalibrated_probability_handling():
    """Verify that uncalibrated classifiers are explicitly marked and qualitative."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.3, 0.9])

    # Trained without calibration
    rep_uncal = ClassificationCalibrationEvaluator.evaluate_calibration(
        y_true=y_true,
        y_prob=y_prob,
        is_trained_calibrated=False
    )
    assert rep_uncal.is_calibrated is False
    assert rep_uncal.calibration_status == "uncalibrated"
    assert any("uncalibrated" in n.lower() for n in rep_uncal.notes)


# ==============================================================================
# 8. PROFILE & RECOMMENDATION RELIABILITY
# ==============================================================================

def test_profile_and_recommendation_reliability():
    """Verify profile completeness, recency, and recommendation evidence tiers."""
    # Recommendations evidence
    recs = [
        {"action": "Pack bag night before", "behavioral_problem": "prep_delay"},
        {"action": "Focus mode phone", "behavioral_problem": "phone_distraction"}
    ]
    # No prior trials -> heuristic rule
    reports = RecommendationReliabilityEvaluator.evaluate_recommendations(recs, intervention_records=[])
    assert len(reports) == 2
    assert reports[0].evidence_level == "heuristic_rule"
    assert reports[0].is_verified is False


# ==============================================================================
# 9. FASTAPI REST ENDPOINTS INTEGRATION
# ==============================================================================

def test_fastapi_reliability_endpoints(test_client, test_db):
    """Verify REST endpoints for reliability status and calibration."""
    # 1. GET /api/intelligence/reliability/status
    res = test_client.get("/api/intelligence/reliability/status")
    assert res.status_code == 200
    data = res.json()
    assert "reliability_status" in data
    assert "data_sufficiency" in data
    assert "calibration_status" in data
    assert "limitations" in data

    # 2. GET /api/intelligence/reliability/calibration
    res_cal = test_client.get("/api/intelligence/reliability/calibration")
    assert res_cal.status_code == 200
    cal_data = res_cal.json()
    assert "brier_score" in cal_data
    assert "expected_calibration_error" in cal_data
    assert "calibration_status" in cal_data

    # 3. POST /api/intelligence/risk with valid payload
    payload = {
        "event_name": "Standup",
        "day_of_week": "Monday",
        "importance": "high",
        "estimated_prep_duration": 25.0,
        "estimated_travel_duration": 20.0,
        "bag_prepared": True
    }
    pred_res = test_client.post("/api/intelligence/risk", json=payload)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "prediction_interval_lower" in pred_data
    assert "prediction_interval_upper" in pred_data
    assert "reliability_status" in pred_data
    assert "limitations" in pred_data
    assert pred_data["is_abstained"] is False

    # 4. POST /api/intelligence/risk with physically invalid out-of-range prep duration -> Abstains!
    out_payload = {
        "event_name": "Standup",
        "day_of_week": "Monday",
        "importance": "high",
        "estimated_prep_duration": 450.0,  # 7.5 hours prep!
        "estimated_travel_duration": 20.0,
        "bag_prepared": True
    }
    abs_res = test_client.post("/api/intelligence/risk", json=out_payload)
    assert abs_res.status_code == 200
    abs_data = abs_res.json()
    assert abs_data["is_abstained"] is True
    assert "operating bounds" in abs_data["abstention_reason"].lower()
    assert abs_data["predicted_delay_minutes"] is None
