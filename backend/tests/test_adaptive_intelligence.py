"""Automated Tests for MicroDelay Adaptive Intelligence Architecture & Foundation.

Tests:
- PredictionContext validation & serialization
- Standardized output contract & backward compatibility adapter
- Lateness risk classifier
- Personalization engine & data maturity tiers
- Behavioral clustering & profile engines
- Feature explainability & adaptive recommendations
- Behavioral drift detection
- Prediction confidence estimation
- Central intelligence orchestration
- Missing model & cold-start user behavior
- Component failure isolation
- Privacy & security (no sensitive data leaks)
- API endpoint backward compatibility
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from ml.adaptive.schemas import (
    ModelAvailability,
    RiskCategory,
    ConfidenceLevel,
    AdaptiveIntelligenceOutput,
    ConfidenceStatus,
    DataQualityStatus
)
from ml.adaptive.context import PredictionContext
from ml.adaptive.risk_classifier import LatenessRiskClassifier
from ml.adaptive.personalization import PersonalizationEngine, PersonalizationTier
from ml.adaptive.clustering import BehavioralClusteringEngine
from ml.adaptive.behavioral_profile import BehavioralProfileEngine
from ml.adaptive.explainability import PredictionExplainer
from ml.adaptive.adaptive_recommendations import AdaptiveRecommendationEngine
from ml.adaptive.drift_detection import BehavioralDriftDetector
from ml.adaptive.confidence import ConfidenceEstimator
from ml.adaptive.orchestrator import AdaptiveIntelligenceOrchestrator


def test_prediction_context_validation():
    """Verify PredictionContext sanitization and validation bounds."""
    # Valid context
    ctx = PredictionContext(
        user_id=1,
        event_name="ML Lecture",
        day_of_week="Monday",
        event_importance="high",
        estimated_prep_duration=25.0,
        estimated_travel_duration=20.0,
        bag_prepared=True
    )
    val = ctx.validate()
    assert val.is_valid is True
    assert len(val.errors) == 0
    assert val.sanitized_values["day_of_week"] == "Monday"
    assert val.sanitized_values["event_importance"] == "high"

    # Context with invalid parameters
    invalid_ctx = PredictionContext(
        user_id=-5,
        event_name="",
        day_of_week="Funday",
        event_importance="unknown_tier",
        estimated_prep_duration=-10.0,
        estimated_travel_duration=500.0,
        bag_prepared=False
    )
    invalid_val = invalid_ctx.validate()
    assert invalid_val.is_valid is False
    assert len(invalid_val.errors) >= 4

    # Serialization and deserialization
    ctx_dict = ctx.to_dict()
    assert ctx_dict["user_id"] == 1
    assert ctx_dict["event_name"] == "ML Lecture"

    from_dict_ctx = PredictionContext.from_dict(ctx_dict)
    assert from_dict_ctx.user_id == 1
    assert from_dict_ctx.event_name == "ML Lecture"
    assert from_dict_ctx.bag_prepared is True

    # Inference dict conversion
    inf_dict = ctx.to_inference_dict()
    assert inf_dict["bag_prepared"] == 1
    assert inf_dict["estimated_prep_duration_min"] == 25.0


def test_adaptive_intelligence_output_contract_and_backward_compatibility():
    """Verify that AdaptiveIntelligenceOutput produces standardized schemas and legacy dicts."""
    output = AdaptiveIntelligenceOutput(
        predicted_delay_minutes=5.2,
        baseline_predicted_delay_minutes=4.5,
        late_probability=0.72,
        risk_category=RiskCategory.HIGH.value,
        contributing_factors=[{
            "factor": "bag_unprepared",
            "impact_minutes": 3.8,
            "direction": "increases_delay",
            "description": "Bag was unprepared"
        }],
        recommended_interventions=[{
            "action": "Pack bag night before",
            "description": "Prepare items early",
            "expected_delay_reduction_min": 3.8,
            "category": "preparation",
            "priority": "high"
        }],
        confidence_status={
            "score": 0.85,
            "level": "high",
            "is_reliable": True,
            "reasons": ["Ample data"]
        },
        model_version="rf_user_1_v2",
        model_type="random_forest",
        model_availability=ModelAvailability.AVAILABLE.value,
        data_quality_status={
            "provenance": "real",
            "sample_count": 25,
            "is_sufficient": True
        }
    )

    full_dict = output.to_dict()
    assert full_dict["predicted_delay_minutes"] == 5.2
    assert full_dict["baseline_predicted_delay_minutes"] == 4.5
    assert full_dict["late_probability"] == 0.72
    assert full_dict["risk_category"] == "high"
    assert full_dict["model_availability"] == "available"
    assert len(full_dict["contributing_factors"]) == 1
    assert len(full_dict["recommended_interventions"]) == 1

    # Verify legacy backward compatibility mapping
    legacy_dict = output.to_legacy_dict()
    assert legacy_dict["predicted_delay_min"] == 5.2
    assert legacy_dict["baseline_predicted_delay_min"] == 4.5
    assert legacy_dict["model_used"] == "random_forest"
    assert legacy_dict["model_version"] == "rf_user_1_v2"
    assert legacy_dict["data_provenance"] == "real"
    assert legacy_dict["is_personalized_real"] is True
    assert legacy_dict["late_probability"] == 0.72
    assert legacy_dict["risk_category"] == "high"


def test_lateness_risk_classifier():
    """Verify calibrated probability calculations and discrete risk categories."""
    classifier = LatenessRiskClassifier()

    # Negative / minimal delay -> Low risk
    p_low, cat_low = classifier.classify(
        context=PredictionContext(1, "Event", "Monday", "medium", 20.0, 15.0, True),
        predicted_delay_min=0.5
    )
    assert p_low is not None
    assert p_low < 0.45
    assert cat_low == RiskCategory.LOW.value

    # Moderate delay (3-5 min) -> Moderate risk
    p_mod, cat_mod = classifier.classify(
        context=PredictionContext(1, "Event", "Monday", "medium", 20.0, 15.0, True),
        predicted_delay_min=4.0
    )
    assert 0.30 <= p_mod <= 0.75
    assert cat_mod in (RiskCategory.MODERATE.value, RiskCategory.HIGH.value)

    # Heavy delay (> 12 min) -> Extreme risk
    p_ext, cat_ext = classifier.classify(
        context=PredictionContext(1, "Event", "Monday", "medium", 20.0, 15.0, False),
        predicted_delay_min=14.0
    )
    assert p_ext > 0.85
    assert cat_ext == RiskCategory.EXTREME.value


def test_personalization_tiers_and_data_maturity():
    """Verify the four tiers of the global-to-personalized learning ladder."""
    engine = PersonalizationEngine(full_personalization_threshold=8)

    # 1. Cold start: 0 real, 0 synthetic
    m_cold = engine.evaluate_maturity(real_samples=0, synthetic_samples=0)
    assert m_cold["tier"] == PersonalizationTier.GLOBAL_COLD_START
    assert m_cold["weight_real"] == 0.0
    assert m_cold["weight_prior"] == 1.0
    assert m_cold["is_personalized"] is False

    # 2. Persona prior: 0 real, 60 synthetic
    m_synth = engine.evaluate_maturity(real_samples=0, synthetic_samples=60)
    assert m_synth["tier"] == PersonalizationTier.PERSONA_PRIOR
    assert m_synth["weight_real"] == 0.0
    assert m_synth["weight_prior"] == 1.0

    # 3. Bayesian blended: 4 real, 60 synthetic
    m_blend = engine.evaluate_maturity(real_samples=4, synthetic_samples=60)
    assert m_blend["tier"] == PersonalizationTier.BAYESIAN_BLENDED
    assert 0.4 <= m_blend["weight_real"] <= 0.6
    assert m_blend["is_calibrated"] is True
    assert m_blend["is_personalized"] is False

    # Blending calculation
    blended = engine.blend_predictions(personalized_pred=8.0, prior_pred=4.0, maturity_info=m_blend)
    assert 4.0 < blended < 8.0

    # 4. Fully personalized: 12 real
    m_full = engine.evaluate_maturity(real_samples=12, synthetic_samples=60)
    assert m_full["tier"] == PersonalizationTier.PERSONALIZED_AUTONOMOUS
    assert m_full["weight_real"] == 1.0
    assert m_full["weight_prior"] == 0.0
    assert m_full["is_personalized"] is True


def test_behavioral_clustering_and_archetypes():
    """Verify context mapping into routine archetypes."""
    clustering = BehavioralClusteringEngine()

    # Unprepared commute
    c_unprep = clustering.assign_archetype(PredictionContext(
        1, "Work", "Monday", "medium", 25.0, 20.0, bag_prepared=False
    ))
    assert c_unprep["archetype_id"] == "unprepared_commute"
    assert c_unprep["risk_multiplier"] > 1.0

    # High stakes departure
    c_high = clustering.assign_archetype(PredictionContext(
        1, "Final Defense", "Tuesday", "critical", 30.0, 20.0, bag_prepared=True
    ))
    assert c_high["archetype_id"] == "high_stakes_departure"

    # Deliberate preparation
    c_delib = clustering.assign_archetype(PredictionContext(
        1, "Weekly Lab", "Wednesday", "medium", 45.0, 15.0, bag_prepared=True
    ))
    assert c_delib["archetype_id"] == "deliberate_preparation"
    assert c_delib["risk_multiplier"] < 1.0


def test_behavioral_profile_building():
    """Verify extraction of persistent behavioral patterns from logs."""
    profile_engine = BehavioralProfileEngine()

    # Empty logs
    p_empty = profile_engine.build_profile([])
    assert p_empty["has_profile"] is False
    assert p_empty["total_records"] == 0

    # Mock historical logs
    mock_logs = [
        {
            "actual_prep_duration_min": 35.0,
            "estimated_prep_duration_min": 25.0,
            "total_delay_min": 10.0,
            "tag_finding_items": True,
            "tag_phone_usage": True,
            "tag_one_last_task": False,
            "delay_finding_items_min": 4.0,
            "delay_phone_usage_min": 3.0,
            "delay_prep_bias_min": 3.0
        },
        {
            "actual_prep_duration_min": 30.0,
            "estimated_prep_duration_min": 25.0,
            "total_delay_min": 5.0,
            "tag_finding_items": True,
            "tag_phone_usage": False,
            "tag_one_last_task": False,
            "delay_finding_items_min": 3.0,
            "delay_prep_bias_min": 2.0
        },
        {
            "actual_prep_duration_min": 25.0,
            "estimated_prep_duration_min": 25.0,
            "total_delay_min": 0.0,
            "tag_finding_items": False,
            "tag_phone_usage": False,
            "tag_one_last_task": False
        }
    ]

    p = profile_engine.build_profile(mock_logs)
    assert p["has_profile"] is True
    assert p["total_records"] == 3
    assert p["prep_bias_min"] > 0
    assert p["item_search_frequency"] == round(2 / 3, 2)
    assert p["punctuality_rate"] == round(1 / 3, 2)
    assert p["dominant_bottleneck"] in ("item_searching", "preparation_underestimation")


def test_explainability_and_adaptive_recommendations():
    """Verify factor attribution and intervention synthesis."""
    explainer = PredictionExplainer()
    recommender = AdaptiveRecommendationEngine()

    ctx = PredictionContext(
        user_id=1,
        event_name="Early Standup",
        day_of_week="Monday",
        event_importance="high",
        estimated_prep_duration=20.0,
        estimated_travel_duration=20.0,
        bag_prepared=False
    )
    profile = {"prep_bias_min": 4.2, "dominant_bottleneck": "phone_distraction"}

    factors = explainer.explain(
        context=ctx,
        predicted_delay_min=7.5,
        baseline_delay_min=4.5,
        profile_data=profile
    )
    assert len(factors) >= 2
    factor_names = [f["factor"] for f in factors]
    assert "bag_unprepared" in factor_names
    assert "monday_start_inertia" in factor_names

    recs = recommender.generate_recommendations(
        context=ctx,
        predicted_delay_min=7.5,
        contributing_factors=factors,
        profile_data=profile
    )
    assert len(recs) >= 2
    actions = [r["action"] for r in recs]
    assert any("Bag Packing" in a for a in actions)
    assert any("Buffer" in a for a in actions)


def test_behavioral_drift_detection():
    """Verify drift detection between baseline and recent operational logs."""
    detector = BehavioralDriftDetector(recent_window_size=3, drift_threshold_min=3.0)

    # Insufficient logs
    res_small = detector.detect_drift([{"total_delay_min": 5.0}] * 4)
    assert res_small["drift_detected"] is False
    assert res_small["status"] == "insufficient_data"

    # Stable behavior (similar delays throughout)
    stable_logs = [{"total_delay_min": 4.0}] * 10
    res_stable = detector.detect_drift(stable_logs)
    assert res_stable["drift_detected"] is False
    assert res_stable["status"] == "stable"
    assert res_stable["mean_shift_min"] == 0.0

    # Drifting behavior: recent 3 delays are 12 min vs older 7 delays at 3 min
    drifting_logs = (
        [{"total_delay_min": 12.0}] * 3 +
        [{"total_delay_min": 3.0}] * 7
    )
    res_drift = detector.detect_drift(drifting_logs)
    assert res_drift["drift_detected"] is True
    assert res_drift["status"] == "significant_drift"
    assert res_drift["direction"] == "degrading"
    assert res_drift["mean_shift_min"] > 5.0


def test_confidence_estimator():
    """Verify confidence score calculations and diagnostic reasons."""
    estimator = ConfidenceEstimator()
    ctx = PredictionContext(1, "Event", "Monday", "medium", 25.0, 20.0, True)

    # High confidence: abundant genuine real data
    conf_high = estimator.estimate_confidence(
        context=ctx,
        sample_counts={"real_samples": 25, "synthetic_samples": 0, "total_samples": 25},
        validation_mae=2.5,
        is_known_event=True
    )
    assert conf_high.score >= 0.70
    assert conf_high.level == ConfidenceLevel.HIGH.value
    assert conf_high.is_reliable is True

    # Low confidence: 0 samples (cold start)
    conf_low = estimator.estimate_confidence(
        context=ctx,
        sample_counts={"real_samples": 0, "synthetic_samples": 0, "total_samples": 0},
        validation_mae=None,
        is_known_event=False
    )
    assert conf_low.score < 0.50
    assert conf_low.level == ConfidenceLevel.LOW.value
    assert conf_low.is_reliable is False


def test_component_failure_isolation():
    """Verify that a failure in one subcomponent never crashes the orchestrator prediction."""
    # Create an orchestrator with mock failing components
    failing_explainer = MagicMock()
    failing_explainer.explain.side_effect = RuntimeError("Explainer crashed unexpectedly")

    failing_risk = MagicMock()
    failing_risk.classify.side_effect = ValueError("Risk classifier math error")

    orchestrator = AdaptiveIntelligenceOrchestrator(
        explainer=failing_explainer,
        risk_classifier=failing_risk
    )

    ctx = PredictionContext(
        user_id=9999,
        event_name="Lab",
        day_of_week="Monday",
        event_importance="high",
        estimated_prep_duration=25.0,
        estimated_travel_duration=20.0,
        bag_prepared=True
    )

    # Must execute safely without throwing exceptions!
    output = orchestrator.predict(ctx, db=None)

    assert isinstance(output, AdaptiveIntelligenceOutput)
    assert output.predicted_delay_minutes > 0
    assert output.component_status["explainability"].startswith("failed:")
    assert output.component_status["risk_classification"].startswith("failed:")
    assert output.risk_category == RiskCategory.MODERATE.value
    # Other components succeeded
    assert output.component_status["recommendations"] == "success"
    assert output.component_status["confidence"] == "success"


def test_missing_user_and_cold_start_orchestration():
    """Verify orchestrator behavior when querying a non-existent or fresh user without history."""
    orchestrator = AdaptiveIntelligenceOrchestrator()
    ctx = PredictionContext(
        user_id=88888,  # non-existent user
        event_name="Fresh Event",
        day_of_week="Tuesday",
        event_importance="medium",
        estimated_prep_duration=30.0,
        estimated_travel_duration=15.0,
        bag_prepared=False
    )

    output = orchestrator.predict(ctx, db=None)
    assert output.predicted_delay_minutes == 4.5
    assert output.model_availability in (ModelAvailability.INSUFFICIENT_DATA.value, ModelAvailability.AVAILABLE.value)
    assert output.fallback_applied is True
    assert output.data_quality_status["provenance"] == "none"
    assert output.confidence_status["is_reliable"] is False


def test_privacy_and_security_scrubbing():
    """Verify that sensitive user data is not included in output or logs."""
    ctx = PredictionContext(
        user_id=101,
        event_name="Private Meeting with Doctor",
        day_of_week="Wednesday",
        event_importance="high",
        estimated_prep_duration=25.0,
        estimated_travel_duration=20.0,
        bag_prepared=True,
        metadata={"private_notes": "Prescription details should remain local"}
    )
    orchestrator = AdaptiveIntelligenceOrchestrator()
    output = orchestrator.predict(ctx, db=None)

    output_dict = output.to_dict()
    # Ensure raw metadata is not leaked into output
    assert "private_notes" not in output_dict
    assert "Prescription" not in str(output_dict["data_quality_status"])


def test_api_adaptive_prediction_endpoints(client, db_session):
    """Verify the /api/predict backward compatibility and /api/intelligence/predict endpoints."""
    # 1. Register test user
    reg_resp = client.post("/api/auth/register", json={
        "username": "adaptive_api_user",
        "password": "adaptive_password_123",
        "profile_type": "student"
    })
    assert reg_resp.status_code == 200
    token = reg_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Test legacy /api/predict maintains backwards compatibility
    legacy_resp = client.get("/api/predict", params={
        "event_name": "ML Lab",
        "day_of_week": "Monday",
        "importance": "high",
        "bag_prepared": True,
        "estimated_prep": 25.0,
        "estimated_travel": 20.0
    }, headers=headers)
    assert legacy_resp.status_code == 200
    legacy_data = legacy_resp.json()
    assert "predicted_delay_min" in legacy_data
    assert "baseline_predicted_delay_min" in legacy_data
    assert "model_used" in legacy_data
    assert "model_version" in legacy_data
    assert "late_probability" in legacy_data
    assert "risk_category" in legacy_data
    assert "model_availability" in legacy_data

    # 3. Test new GET /api/intelligence/predict returns complete AdaptiveIntelligenceOutput
    intel_resp = client.get("/api/intelligence/predict", params={
        "event_name": "ML Lab",
        "day_of_week": "Monday",
        "importance": "high",
        "bag_prepared": False,
        "estimated_prep": 25.0,
        "estimated_travel": 20.0
    }, headers=headers)
    assert intel_resp.status_code == 200
    intel_data = intel_resp.json()
    assert "predicted_delay_minutes" in intel_data
    assert "late_probability" in intel_data
    assert "risk_category" in intel_data
    assert "contributing_factors" in intel_data
    assert "recommended_interventions" in intel_data
    assert "confidence_status" in intel_data
    assert "data_quality_status" in intel_data
    assert "model_availability" in intel_data
    assert intel_data["model_availability"] == "available"

    # 4. Test new POST /api/intelligence/predict returns complete AdaptiveIntelligenceOutput
    intel_post_resp = client.post("/api/intelligence/predict", json={
        "event_name": "ML Lab",
        "day_of_week": "Monday",
        "importance": "high",
        "bag_prepared": True,
        "estimated_prep_duration": 25.0,
        "estimated_travel_duration": 20.0
    }, headers=headers)
    assert intel_post_resp.status_code == 200
    intel_post_data = intel_post_resp.json()
    assert intel_post_data["predicted_delay_minutes"] > 0
    assert len(intel_post_data["contributing_factors"]) > 0
    assert len(intel_post_data["recommended_interventions"]) > 0
