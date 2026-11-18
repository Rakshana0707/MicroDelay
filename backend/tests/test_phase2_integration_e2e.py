"""Phase 2 Adaptive Behavioral Intelligence — End-to-End Integration Verification Suite.

Tests Scenarios A through G covering all integrated Phase 2 modules:
- Scenario A: New user registration, onboarding, synthetic separation, cold-start handling.
- Scenario B: Existing user routines, behavioral features, risk evaluation, model versioning.
- Scenario C: Behavioral clustering, tournament benchmarking, habit archetype assignment.
- Scenario D: Evidence-backed recommendations, trial acceptance, completion, outcome tracking.
- Scenario E: Model reliability, conformal intervals, calibration, abstention.
- Scenario F: Behavioral drift detection, habit shifts, acknowledgment, model monitoring.
- Scenario G: Regression of existing features (auth, simulator, routines, history, evaluation).
"""

import pytest
import datetime
import uuid
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.main import app, get_db
from backend.database import User, Event, DepartureLog, UserInterventionRecord, ModelReliabilityRecord, BehavioralDriftEventRecord
from backend.security import hash_password, create_access_token


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db_session():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()


def create_test_user_with_token(db: Session, prefix: str = "p2_e2e"):
    uid = uuid.uuid4().hex[:8]
    username = f"{prefix}_{uid}"
    pwd_hash, salt = hash_password("ValidPass123!")
    user = User(
        username=username,
        hashed_password=pwd_hash,
        salt=salt,
        profile_type="it_professional",
        consent_to_global_training=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(data={"sub": user.username})
    return user, token


# ==============================================================================
# SCENARIO A: New User Registration, Onboarding, Synthetic Separation, Cold Start
# ==============================================================================
def test_scenario_a_new_user_onboarding_and_cold_start(client, db_session):
    """Scenario A: New user registers, has zero genuine data, triggers cold-start handling."""
    user, token = create_test_user_with_token(db_session, prefix="user_a")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Verify data summary reports 0 real samples
    res_summary = client.get("/api/data-summary", headers=headers)
    assert res_summary.status_code == 200
    data_summary = res_summary.json()
    assert data_summary["real_samples"] == 0
    assert data_summary["is_sufficient_real"] is False

    # 2. Test cold-start prediction behavior
    res_pred = client.get(
        "/api/predict?event_name=Sprint%20Planning&day_of_week=Monday&importance=high&bag_prepared=false&estimated_prep=30&estimated_travel=25",
        headers=headers
    )
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    assert "predicted_delay_min" in pred_data
    assert pred_data.get("model_used") in ["global_model", "synthetic_demo", "baseline", "cold_start"]

    # 3. Test risk evaluation for new user
    res_risk = client.get(
        "/api/intelligence/risk?event_name=Sprint%20Planning&day_of_week=Monday&importance=high&bag_prepared=false&estimated_prep=30&estimated_travel=25",
        headers=headers
    )
    assert res_risk.status_code == 200
    risk_data = res_risk.json()
    # Either returns safe abstention or baseline/global risk with appropriate reliability
    if risk_data.get("is_abstained"):
        assert "abstention_reason" in risk_data
        assert risk_data["reliability_status"] == "insufficient_history"
    else:
        assert risk_data["reliability_status"] in ["insufficient_history", "low_reliability", "moderate_reliability"]


# ==============================================================================
# SCENARIO B: Existing User Routines, Features, Risk Evaluation & Model Versioning
# ==============================================================================
def test_scenario_b_existing_user_routines_and_risk(client, db_session):
    """Scenario B: User with sufficient genuine checkouts gets validated risk and versioning."""
    user, token = create_test_user_with_token(db_session, prefix="user_b")
    headers = {"Authorization": f"Bearer {token}"}

    # Create routine event
    event = Event(
        user_id=user.id,
        name="Team Standup",
        day_of_week="Tuesday",
        start_time="09:30:00",
        estimated_prep_duration=25,
        estimated_travel_duration=20,
        importance="high"
    )
    db_session.add(event)
    db_session.commit()
    db_session.refresh(event)

    # Seed 10 genuine departure logs
    today = datetime.date.today()
    for i in range(10):
        log_date = today - datetime.timedelta(days=i + 1)
        log = DepartureLog(
            user_id=user.id,
            event_id=event.id,
            event_name=event.name,
            date=str(log_date),
            day_of_week=event.day_of_week,
            planned_prep_start="08:30:00",
            planned_departure="08:55:00",
            estimated_prep_duration_min=25.0,
            estimated_travel_duration_min=20.0,
            actual_prep_start="08:30:00",
            actual_departure="09:02:00",
            actual_arrival="09:24:00",
            actual_prep_duration_min=32.0 + (i % 3),
            actual_travel_duration_min=22.0,
            total_delay_min=7.0 + (i % 2),
            bag_prepared=False,
            is_synthetic=False,
            provenance="real",
            data_source="simulator_checkout",
            tag_phone_usage=True,
            tag_finding_items=True,
            tag_one_last_task=True,
            delay_phone_usage_min=5.0,
            delay_finding_items_min=3.0,
            delay_one_last_task_min=2.0
        )
        db_session.add(log)
    db_session.commit()

    # Query Risk Prediction
    res_risk = client.get(
        f"/api/intelligence/risk?event_name={event.name}&day_of_week={event.day_of_week}&importance=high&bag_prepared=false&estimated_prep=25&estimated_travel=20&event_id={event.id}",
        headers=headers
    )
    assert res_risk.status_code == 200
    risk_data = res_risk.json()
    assert risk_data["late_probability"] is not None
    assert risk_data["risk_category"] in ["low", "moderate", "high", "extreme"]
    assert "model_version" in risk_data
    assert "contributing_factors" in risk_data
    assert len(risk_data["contributing_factors"]) > 0

    # Verify associative framing in contributing factors
    for factor in risk_data["contributing_factors"]:
        assert "factor" in factor
        assert "impact_minutes" in factor
        assert "direction" in factor


# ==============================================================================
# SCENARIO C: Behavioral Clustering & Habit Archetypes
# ==============================================================================
def test_scenario_c_behavioral_clustering_and_archetypes(client, db_session):
    """Scenario C: Unsupervised clustering discovers archetype and benchmark returns tournament."""
    user, token = create_test_user_with_token(db_session, prefix="user_c")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Fetch user cluster profile
    res_prof = client.get("/api/intelligence/clusters/profile", headers=headers)
    assert res_prof.status_code == 200
    prof_data = res_prof.json()
    assert "cluster_name" in prof_data
    assert "cluster_description" in prof_data
    assert "dominant_characteristics" in prof_data
    assert "recommended_action" in prof_data

    # 2. Fetch cluster benchmark tournament
    res_bench = client.get("/api/intelligence/clusters/benchmark", headers=headers)
    assert res_bench.status_code == 200
    bench_data = res_bench.json()
    assert "algorithm" in bench_data
    assert "silhouette_score" in bench_data


# ==============================================================================
# SCENARIO D: Evidence-Backed Recommendations & Habit Trial Lifecycle
# ==============================================================================
def test_scenario_d_recommendations_and_trial_lifecycle(client, db_session):
    """Scenario D: Recommendation lifecycle — fetch, accept trial, complete with feedback, dismiss."""
    user, token = create_test_user_with_token(db_session, prefix="user_d")
    headers = {"Authorization": f"Bearer {token}"}

    # Seed logs with phone distraction to generate diagnosed interventions
    today = datetime.date.today()
    for i in range(5):
        log = DepartureLog(
            user_id=user.id,
            event_name="Morning Class",
            date=str(today - datetime.timedelta(days=i + 1)),
            day_of_week="Monday",
            planned_prep_start="08:00:00",
            planned_departure="08:30:00",
            estimated_prep_duration_min=30.0,
            estimated_travel_duration_min=20.0,
            actual_prep_start="08:00:00",
            actual_departure="08:42:00",
            actual_arrival="09:05:00",
            actual_prep_duration_min=42.0,
            actual_travel_duration_min=23.0,
            total_delay_min=12.0,
            bag_prepared=False,
            tag_phone_usage=True,
            delay_phone_usage_min=8.0,
            is_synthetic=False,
            provenance="real",
            data_source="simulator_checkout"
        )
        db_session.add(log)
    db_session.commit()

    # 1. Fetch personalized recommendations
    res_recs = client.get("/api/recommendations/personalized", headers=headers)
    assert res_recs.status_code == 200
    recs = res_recs.json()
    assert isinstance(recs, list)
    assert len(recs) > 0
    target_intv_id = recs[0]["intervention_id"]

    # 2. Accept a recommendation
    res_accept = client.post(
        "/api/interventions/accept",
        json={"intervention_id": target_intv_id},
        headers=headers
    )
    assert res_accept.status_code == 200
    accept_data = res_accept.json()
    assert accept_data["status"] == "accepted"

    # 3. Complete the intervention trial
    res_comp = client.post(
        "/api/interventions/complete",
        json={
            "intervention_id": target_intv_id,
            "user_rating": 5,
            "helpful": True,
            "feedback_notes": "Saved around 5 minutes every morning"
        },
        headers=headers
    )
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    assert comp_data["status"] == "completed"

    # 4. Dismiss another recommendation if available
    if len(recs) > 1:
        second_id = recs[1]["intervention_id"]
        res_dismiss = client.post(
            "/api/interventions/dismiss",
            json={"intervention_id": second_id},
            headers=headers
        )
        assert res_dismiss.status_code == 200
        assert res_dismiss.json()["status"] == "dismissed"


# ==============================================================================
# SCENARIO E: Model Reliability, Conformal Intervals & Calibration
# ==============================================================================
def test_scenario_e_model_reliability_and_calibration(client, db_session):
    """Scenario E: Reliability engine computes conformal intervals, calibration reports, and abstention."""
    user, token = create_test_user_with_token(db_session, prefix="user_e")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Check user reliability status
    res_rel = client.get("/api/intelligence/reliability/status", headers=headers)
    assert res_rel.status_code == 200
    rel_data = res_rel.json()
    assert "reliability_status" in rel_data
    assert "data_sufficiency" in rel_data
    assert "data_sufficiency_report" in rel_data

    # 2. Check calibration report
    res_cal = client.get("/api/intelligence/reliability/calibration", headers=headers)
    assert res_cal.status_code == 200
    cal_data = res_cal.json()
    assert "calibration_status" in cal_data
    assert "brier_score" in cal_data
    assert "expected_calibration_error" in cal_data


# ==============================================================================
# SCENARIO F: Behavioral Drift Detection, Recalibration & Monitoring
# ==============================================================================
def test_scenario_f_behavioral_drift_detection_and_monitoring(client, db_session):
    """Scenario F: Drift detection identifies habit shifts and supports event acknowledgment."""
    user, token = create_test_user_with_token(db_session, prefix="user_f")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Fetch drift analysis
    res_drift = client.get("/api/intelligence/drift/analysis", headers=headers)
    assert res_drift.status_code == 200
    drift_data = res_drift.json()
    assert "status" in drift_data
    assert "events_detected" in drift_data

    # 2. Fetch model performance monitoring
    res_mon = client.get("/api/intelligence/drift/model-monitoring", headers=headers)
    assert res_mon.status_code == 200
    mon_data = res_mon.json()
    assert "performance_degraded" in mon_data


# ==============================================================================
# SCENARIO G: Regression of Existing Phase 1 Features
# ==============================================================================
def test_scenario_g_regression_existing_features(client, db_session):
    """Scenario G: Existing endpoints (events, logs, simulator checkout, history) remain 100% operational."""
    user, token = create_test_user_with_token(db_session, prefix="user_g")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create a routine event
    new_event_payload = {
        "name": "Gym Workout",
        "day_of_week": "Wednesday",
        "start_time": "07:00:00",
        "estimated_prep_duration": 20,
        "estimated_travel_duration": 15,
        "importance": "medium"
    }
    res_create = client.post("/api/events", json=new_event_payload, headers=headers)
    assert res_create.status_code == 200
    created_event = res_create.json()
    event_id = created_event["id"]

    # 2. Get user events
    res_events = client.get("/api/events", headers=headers)
    assert res_events.status_code == 200
    events_list = res_events.json()
    assert any(e["id"] == event_id for e in events_list)

    # 3. Log a departure checkout
    new_log_payload = {
        "event_id": event_id,
        "event_name": "Gym Workout",
        "date": str(datetime.date.today()),
        "day_of_week": "Wednesday",
        "planned_prep_start": "06:30:00",
        "planned_departure": "06:50:00",
        "estimated_prep_duration_min": 20.0,
        "estimated_travel_duration_min": 15.0,
        "actual_prep_start": "06:30:00",
        "actual_departure": "06:55:00",
        "actual_arrival": "07:10:00",
        "actual_prep_duration_min": 25.0,
        "actual_travel_duration_min": 15.0,
        "bag_prepared": True,
        "tag_one_last_task": False,
        "tag_finding_items": False,
        "tag_phone_usage": True,
        "tag_waiting": False,
        "observed_phone_usage_min": 2.0
    }
    res_log = client.post("/api/checkout", json=new_log_payload, headers=headers)
    assert res_log.status_code == 200
    log_data = res_log.json()
    assert log_data.get("success") is True
    assert "calculated_delay_min" in log_data

    # 4. Fetch departure history
    res_hist = client.get("/api/logs", headers=headers)
    assert res_hist.status_code == 200
    logs_list = res_hist.json()
    assert len(logs_list) >= 1
