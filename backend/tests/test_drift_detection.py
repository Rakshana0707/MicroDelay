"""Comprehensive Automated Test Suite for Behavioral Drift Detection & Habit Change Monitoring (Prompt 2.8).

Verifies:
1. Statistical primitives: PSI, KS two-sample test, Welch t-test.
2. Page-Hinkley sequential change detection test.
3. Robustness against isolated outlier checkouts (no false-positive alarm).
4. Stable behavior baseline validation (zero false positives).
5. Abrupt behavioral shift detection (sudden mean increase).
6. Positive behavioral improvement classification (never flagged as negative).
7. Minimum sample guards under cold-start conditions.
8. Separation of input data drift from model predictive degradation (rolling MAE & bias).
9. Full FastAPI REST API drift endpoints integration and user isolation.
"""

import datetime
from typing import Optional
import pytest
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import (
    Base,
    User,
    DepartureLog,
    BehavioralDriftEventRecord,
    UserBehavioralBaselineRecord,
    ModelMonitoringLogRecord,
    migrate_db
)
from main import app, get_db
from security import create_access_token
from ml.drift import (
    calculate_psi,
    calculate_ks_test,
    calculate_welch_ttest,
    PageHinkleyDetector,
    BehavioralBaselineService,
    TemporalDriftAnalyzer,
    ModelPerformanceMonitor,
    AdaptiveDriftCoordinator,
    classify_drift_type_and_valence
)
from ml.adaptive.drift_detection import BehavioralDriftDetector


@pytest.fixture
def test_db():
    """In-memory SQLite database isolated per test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    migrate_db(engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def create_test_user(db, username: str = "drift_user") -> User:
    user = User(
        username=username,
        hashed_password="hashed_pw_test",
        salt="test_salt",
        profile_type="student",
        consent_to_global_training=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_log_entry(
    db,
    user_id: int,
    date_str: str,
    total_delay_min: float,
    actual_prep: float = 30.0,
    estimated_prep: float = 25.0,
    phone_used: bool = False,
    items_searched: bool = False
) -> DepartureLog:
    log = DepartureLog(
        user_id=user_id,
        event_name="Class Routine",
        date=date_str,
        day_of_week="Monday",
        planned_prep_start="08:00:00",
        planned_departure="08:30:00",
        estimated_prep_duration_min=estimated_prep,
        estimated_travel_duration_min=20.0,
        actual_prep_start="08:00:00",
        actual_departure="08:35:00",
        actual_arrival="08:55:00",
        actual_prep_duration_min=actual_prep,
        actual_travel_duration_min=20.0,
        total_delay_min=total_delay_min,
        bag_prepared=True,
        tag_one_last_task=False,
        tag_finding_items=items_searched,
        tag_phone_usage=phone_used,
        tag_waiting=False,
        provenance="real",
        is_synthetic=False,
        recorded_at=f"{date_str}T08:35:00Z",
        observation_status="valid"
    )
    db.add(log)
    db.commit()
    return log


# =====================================================================
# 1. STATISTICAL TEST PRIMITIVES
# =====================================================================

def test_statistical_primitives_psi_and_ks():
    """Verify calculation of Population Stability Index (PSI) and Kolmogorov-Smirnov test."""
    np.random.seed(42)

    # Identical distributions with sufficient sample size -> PSI should be very small (< 0.15)
    baseline_samples = list(np.random.normal(loc=10.0, scale=2.0, size=150))
    stable_samples = list(np.random.normal(loc=10.02, scale=2.0, size=100))
    psi_stable = calculate_psi(baseline_samples, stable_samples)
    assert psi_stable < 0.15

    # Substantially shifted distribution -> PSI should be >= 0.25
    drifted_samples = list(np.random.normal(loc=18.0, scale=3.0, size=100))
    psi_drifted = calculate_psi(baseline_samples, drifted_samples)
    assert psi_drifted >= 0.25

    # KS Test on stable vs shifted
    ks_stat_stable, ks_p_stable = calculate_ks_test(baseline_samples, stable_samples)
    assert ks_p_stable > 0.05  # Null hypothesis cannot be rejected

    ks_stat_drift, ks_p_drift = calculate_ks_test(baseline_samples, drifted_samples)
    assert ks_p_drift < 0.01  # Statistically significant divergence
    assert ks_stat_drift > 0.5


def test_welch_ttest_and_valence_classification():
    """Verify Welch's t-test and critical distinction between improvement and deterioration."""
    b_samples = [10.0, 11.0, 9.5, 10.5, 10.2, 9.8, 11.2, 10.1]
    
    # Significant improvement (delay drops to 2m)
    improved_samples = [2.0, 2.5, 1.8, 2.2, 2.1]
    t_stat, p_val, mean_shift = calculate_welch_ttest(b_samples, improved_samples)
    assert p_val < 0.01
    assert mean_shift < -7.0

    drift_type, valence, severity = classify_drift_type_and_valence(
        "departure_delay_min", 10.2, 2.1, mean_shift
    )
    # MUST BE POSITIVE IMPROVEMENT
    assert drift_type == "behavioral_improvement"
    assert valence == "positive"
    assert severity == "significant"

    # Significant deterioration (delay increases to 22m)
    worse_samples = [21.0, 22.5, 23.0, 20.5, 22.0]
    t_stat2, p_val2, mean_shift2 = calculate_welch_ttest(b_samples, worse_samples)
    assert p_val2 < 0.01
    assert mean_shift2 > 10.0

    drift_type2, valence2, severity2 = classify_drift_type_and_valence(
        "departure_delay_min", 10.2, 21.8, mean_shift2
    )
    assert drift_type2 == "behavioral_deterioration"
    assert valence2 == "negative"
    assert severity2 == "significant"

    # Neutral feature change (prep duration shifted without lateness value judgment)
    drift_type3, valence3, severity3 = classify_drift_type_and_valence(
        "actual_prep_duration_min", 25.0, 35.0, 10.0
    )
    assert drift_type3 == "data_drift"
    assert valence3 == "neutral"


# =====================================================================
# 2. SEQUENTIAL CHANGE DETECTION (PAGE-HINKLEY)
# =====================================================================

def test_page_hinkley_sequential_change_detection():
    """Verify Page-Hinkley cumulative sum test on streaming observations."""
    np.random.seed(123)
    detector = PageHinkleyDetector(delta=1.0, threshold=12.0)

    # Stream of 25 stable observations around mean 5.0
    stable_stream = np.random.normal(loc=5.0, scale=0.8, size=25)
    for x in stable_stream:
        trig, direction, score = detector.update(float(x))
        assert not trig  # Should NOT trigger on stationary series

    # Sudden step increase to 18.0
    step_stream = np.random.normal(loc=18.0, scale=1.0, size=15)
    triggered = False
    for x in step_stream:
        trig, direction, score = detector.update(float(x))
        if trig:
            triggered = True
            assert direction == "increase"
            assert score > 12.0
            break

    assert triggered, "Page-Hinkley should detect abrupt cumulative shift"


# =====================================================================
# 3. ROBUSTNESS AGAINST ISOLATED OUTLIERS
# =====================================================================

def test_isolated_outlier_robustness(test_db):
    """Verify that a single abnormal day (e.g. 45 min delay) does NOT trigger a false alarm."""
    user = create_test_user(test_db, username="outlier_user")

    # 15 Baseline checkouts with normal delay ~3.0m
    base_date = datetime.date(2026, 10, 1)
    for i in range(15):
        d_str = (base_date + datetime.timedelta(days=i)).isoformat()
        create_log_entry(test_db, user.id, d_str, total_delay_min=3.0 + (i % 2) * 0.5)

    # Recent 7 checkouts: 6 normal days (~3.0m) + 1 single isolated catastrophic day (45.0m)
    recent_date = datetime.date(2026, 10, 16)
    for j in range(6):
        d_str = (recent_date + datetime.timedelta(days=j)).isoformat()
        create_log_entry(test_db, user.id, d_str, total_delay_min=3.0)

    # 7th day: isolated outlier
    create_log_entry(test_db, user.id, "2026-10-23", total_delay_min=45.0)

    analysis = TemporalDriftAnalyzer.analyze_user_drift(test_db, user.id)
    assert analysis["status"] == "success"

    # Outlier filter must prevent false alarm on departure delay
    delay_events = [e for e in analysis["events_detected"] if e["feature_name"] == "departure_delay_min" and e["is_statistically_significant"]]
    assert len(delay_events) == 0, "Single isolated outlier day must not trigger statistically significant drift"


# =====================================================================
# 4. CONTROLLED SCENARIOS: STABLE BEHAVIOR
# =====================================================================

def test_stable_behavior_no_false_positives(test_db):
    """Verify that consistent, stationary routines yield zero significant drift events."""
    user = create_test_user(test_db, username="stable_user")

    start_date = datetime.date(2026, 9, 1)
    for i in range(25):
        d_str = (start_date + datetime.timedelta(days=i)).isoformat()
        create_log_entry(
            test_db, user.id, d_str,
            total_delay_min=3.5 + (0.4 if i % 2 == 0 else -0.4),
            actual_prep=28.0,
            estimated_prep=25.0
        )

    analysis = TemporalDriftAnalyzer.analyze_user_drift(test_db, user.id)
    assert analysis["status"] == "success"
    sig_events = [e for e in analysis["events_detected"] if e.get("is_statistically_significant")]
    assert len(sig_events) == 0
    assert "routines remain stable" in analysis["summary"].lower()


# =====================================================================
# 5. CONTROLLED SCENARIOS: SUDDEN MEAN SHIFT & DETERIORATION
# =====================================================================

def test_sudden_mean_change_detection(test_db):
    """Verify abrupt sustained deterioration (exam period or schedule collapse)."""
    user = create_test_user(test_db, username="deteriorate_user")

    # Baseline 14 days: 2 min delay
    start_date = datetime.date(2026, 9, 1)
    for i in range(14):
        d_str = (start_date + datetime.timedelta(days=i)).isoformat()
        create_log_entry(test_db, user.id, d_str, total_delay_min=2.0)

    # Recent 7 days: sustained 14 min delay
    recent_date = datetime.date(2026, 9, 15)
    for j in range(7):
        d_str = (recent_date + datetime.timedelta(days=j)).isoformat()
        create_log_entry(test_db, user.id, d_str, total_delay_min=14.0)

    analysis = TemporalDriftAnalyzer.analyze_user_drift(test_db, user.id)
    assert analysis["status"] == "success"
    assert analysis["deteriorations_count"] >= 1

    delay_event = next(e for e in analysis["events_detected"] if e["feature_name"] == "departure_delay_min")
    assert delay_event["drift_type"] == "behavioral_deterioration"
    assert delay_event["valence"] == "negative"
    assert delay_event["severity"] == "significant"
    assert delay_event["current_value"] == 14.0
    assert delay_event["baseline_value"] == 2.0


# =====================================================================
# 6. BEHAVIORAL IMPROVEMENT VALIDATION
# =====================================================================

def test_behavioral_improvement_positive_flagging(test_db):
    """Verify that when a user adopts a habit and latency decreases, it is labeled POSITIVE."""
    user = create_test_user(test_db, username="improving_user")

    # Baseline 15 days: chronic 15m delay and phone distractions
    start_date = datetime.date(2026, 9, 1)
    for i in range(15):
        d_str = (start_date + datetime.timedelta(days=i)).isoformat()
        create_log_entry(test_db, user.id, d_str, total_delay_min=15.0, phone_used=True)

    # Recent 7 days: delay drops to 1.5m and zero phone usage
    recent_date = datetime.date(2026, 9, 16)
    for j in range(7):
        d_str = (recent_date + datetime.timedelta(days=j)).isoformat()
        create_log_entry(test_db, user.id, d_str, total_delay_min=1.5, phone_used=False)

    analysis = TemporalDriftAnalyzer.analyze_user_drift(test_db, user.id)
    assert analysis["status"] == "success"
    assert analysis["improvements_count"] >= 1

    delay_ev = next(e for e in analysis["events_detected"] if e["feature_name"] == "departure_delay_min")
    assert delay_ev["drift_type"] == "behavioral_improvement"
    assert delay_ev["valence"] == "positive"
    assert delay_ev["severity"] == "significant"
    assert "improved by" in delay_ev["humanized_summary"].lower()


# =====================================================================
# 7. MINIMUM SAMPLE GUARDS & COLD-START
# =====================================================================

def test_minimum_sample_guards(test_db):
    """Verify that users with few checkouts (< 12) return insufficient_data gracefully."""
    user = create_test_user(test_db, username="cold_user")

    # Only 3 logs recorded
    for i in range(3):
        create_log_entry(test_db, user.id, f"2026-10-0{i+1}", total_delay_min=5.0)

    analysis = TemporalDriftAnalyzer.analyze_user_drift(test_db, user.id)
    assert analysis["status"] == "insufficient_data"
    assert len(analysis["events_detected"]) == 0
    assert "insufficient historical checkouts" in analysis["message"].lower()


# =====================================================================
# 8. PREDICTIVE HEALTH & MODEL MONITORING (CONCEPT DRIFT)
# =====================================================================

def test_model_performance_monitoring_and_concept_drift(test_db):
    """Verify separation of input data drift from model performance degradation."""
    user = create_test_user(test_db, username="monitor_user")

    # Generate 12 logs where actual delay drastically exceeds expected model baseline
    for i in range(12):
        create_log_entry(
            test_db, user.id, f"2026-11-0{i+1}" if i < 9 else f"2026-11-{i+1}",
            total_delay_min=24.0,  # Huge error vs baseline MAE (4.0)
            actual_prep=30.0,
            estimated_prep=25.0
        )

    res = ModelPerformanceMonitor.monitor_user_model_performance(test_db, user.id)
    assert res["status"] == "success"
    assert res["rolling_mae"] is not None
    assert res["performance_degraded"] is True
    assert res["retrain_recommended"] is True

    # Check that concept_drift event was recorded
    concept_events = test_db.query(BehavioralDriftEventRecord).filter(
        BehavioralDriftEventRecord.user_id == user.id,
        BehavioralDriftEventRecord.drift_type == "concept_drift"
    ).all()
    assert len(concept_events) >= 1
    assert concept_events[0].feature_name == "prediction_error_mae"


# =====================================================================
# 9. FASTAPI ENDPOINTS & USER ISOLATION
# =====================================================================

def test_fastapi_drift_endpoints_and_isolation(test_db):
    """Verify all REST API endpoints for drift diagnosis, events, and acknowledgment."""
    user_a = create_test_user(test_db, username="user_drift_a")
    user_b = create_test_user(test_db, username="user_drift_b")

    # Seed User A with sudden drift
    for i in range(14):
        create_log_entry(test_db, user_a.id, f"2026-08-0{i+1}" if i < 9 else f"2026-08-{i+1}", total_delay_min=2.0)
    for j in range(7):
        create_log_entry(test_db, user_a.id, f"2026-08-2{j+1}", total_delay_min=15.0)

    app.dependency_overrides[get_db] = lambda: test_db
    client = TestClient(app)

    try:
        token_a = create_access_token({"sub": user_a.username})
        headers_a = {"Authorization": f"Bearer {token_a}"}

        token_b = create_access_token({"sub": user_b.username})
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # 1. GET /api/intelligence/drift/analysis
        res_analysis = client.get("/api/intelligence/drift/analysis", headers=headers_a)
        assert res_analysis.status_code == 200
        an_data = res_analysis.json()
        assert an_data["status"] == "success"
        assert len(an_data["events_detected"]) >= 1

        # 2. GET /api/intelligence/drift/events
        res_events = client.get("/api/intelligence/drift/events", headers=headers_a)
        assert res_events.status_code == 200
        events_list = res_events.json()
        assert len(events_list) >= 1
        ev_id = events_list[0]["drift_event_id"]

        # 3. User B cannot view User A's drift events
        res_events_b = client.get("/api/intelligence/drift/events", headers=headers_b)
        assert res_events_b.status_code == 200
        assert len(res_events_b.json()) == 0

        # 4. User B cannot acknowledge User A's event
        res_ack_b = client.post(
            f"/api/intelligence/drift/events/{ev_id}/acknowledge",
            json={"status": "acknowledged"},
            headers=headers_b
        )
        assert res_ack_b.status_code == 404

        # 5. User A acknowledges event
        res_ack_a = client.post(
            f"/api/intelligence/drift/events/{ev_id}/acknowledge",
            json={"status": "acknowledged"},
            headers=headers_a
        )
        assert res_ack_a.status_code == 200
        assert res_ack_a.json()["status"] == "acknowledged"

        # 6. GET /api/intelligence/drift/model-monitoring
        res_mon = client.get("/api/intelligence/drift/model-monitoring", headers=headers_a)
        assert res_mon.status_code == 200
        assert "performance_degraded" in res_mon.json()

        # 7. POST /api/intelligence/drift/recalibrate
        res_recal = client.post("/api/intelligence/drift/recalibrate", headers=headers_a)
        assert res_recal.status_code == 200
        assert res_recal.json()["success"] is True
    finally:
        app.dependency_overrides.clear()
