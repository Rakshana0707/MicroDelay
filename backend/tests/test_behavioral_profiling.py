"""Comprehensive Automated Test Suite for Personalized Behavioral Profiling Engine (Prompt 2.5).

Verifies:
1. Behavioral Dimensions (A through F) granular calculation.
2. Multi-window rolling calculation (7-day, 30-day, all-time baseline).
3. Evidence-backed insights generator (strictly factual, no causal leaps).
4. Data sufficiency lifecycle (insufficient_data, gathering_history, sufficient).
5. Snapshot versioning, caching, and auto-update on >= 3 new logs.
6. Profile period comparisons and delta statistics.
7. Synthetic vs genuine data isolation.
8. User access isolation & security (no cross-user profile access).
9. FastAPI REST API endpoints integration.
"""

import pytest
import datetime
import json
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, User, DepartureLog, BehavioralProfileSnapshot
from main import app, get_db
import security
from ml.profiling import (
    BehavioralDimensionCalculator,
    BehavioralInsightsGenerator,
    PersonalizedBehavioralProfiler
)


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
    session = TestingSessionLocal()
    yield session
    session.close()


def create_user_with_departure_logs(
    db,
    username: str,
    n_logs: int,
    is_synthetic: bool = False,
    bag_prepared_pct: float = 0.5,
    phone_pct: float = 0.3,
    item_search_pct: float = 0.2,
    base_delay: float = 4.0,
    prep_bias: float = 5.0,
    start_date: str = "2026-10-01"
):
    """Create a user with synthetic or real departure logs spaced across days."""
    user = User(
        username=username,
        hashed_password="hashed_pw",
        salt="test_salt",
        profile_type="it_professional",
        consent_to_global_training=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    base_dt = datetime.datetime.fromisoformat(start_date)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for i in range(n_logs):
        log_dt = base_dt + datetime.timedelta(days=i)
        day_str = days[log_dt.weekday()]
        has_phone = (i % 10) < int(phone_pct * 10)
        has_item_search = (i % 10) < int(item_search_pct * 10)
        has_bag = (i % 10) < int(bag_prepared_pct * 10)

        est_prep = 25.0
        act_prep = est_prep + prep_bias + (1.0 if i % 2 == 0 else -1.0)
        est_travel = 20.0
        act_travel = 22.0
        delay_val = base_delay + (2.0 if day_str == "Monday" else 0.0)

        log = DepartureLog(
            user_id=user.id,
            event_id=1,
            event_name="Morning Office Commute",
            date=log_dt.strftime("%Y-%m-%d"),
            day_of_week=day_str,
            planned_prep_start=f"{log_dt.strftime('%Y-%m-%d')}T08:00:00",
            planned_departure=f"{log_dt.strftime('%Y-%m-%d')}T08:30:00",
            estimated_prep_duration_min=est_prep,
            estimated_travel_duration_min=est_travel,
            actual_prep_start=f"{log_dt.strftime('%Y-%m-%d')}T08:00:00",
            actual_departure=f"{log_dt.strftime('%Y-%m-%d')}T08:35:00",
            actual_arrival=f"{log_dt.strftime('%Y-%m-%d')}T08:57:00",
            actual_prep_duration_min=act_prep,
            actual_travel_duration_min=act_travel,
            total_delay_min=delay_val,
            bag_prepared=has_bag,
            tag_one_last_task=False,
            tag_finding_items=has_item_search,
            tag_phone_usage=has_phone,
            tag_waiting=False,
            delay_one_last_task_min=0.0,
            delay_finding_items_min=3.0 if has_item_search else 0.0,
            delay_phone_usage_min=4.0 if has_phone else 0.0,
            delay_waiting_min=0.0,
            delay_prep_bias_min=prep_bias,
            attribution_metadata=None,
            provenance="synthetic" if is_synthetic else "real",
            data_source="simulator_checkout",
            is_synthetic=is_synthetic,
            recorded_at=log_dt.isoformat(),
            observation_status="valid"
        )
        db.add(log)

    db.commit()
    return user


def test_empty_logs_dimension_calculator():
    """Verify safe fallback metrics for empty observation logs."""
    dims = BehavioralDimensionCalculator.calculate_dimensions([])
    assert dims["preparation"]["observation_count"] == 0
    assert dims["preparation"]["avg_actual_prep_min"] == 0.0
    assert dims["distraction"]["phone_usage_frequency"] == 0.0
    assert dims["organization"]["bag_prepared_consistency"] == 0.0
    assert dims["departure"]["avg_lateness_min"] == 0.0
    assert dims["routine_consistency"]["schedule_adherence_rate"] == 1.0


def test_granular_dimension_calculations(test_db):
    """Verify all 6 dimensions (A through F) are accurately computed."""
    user = create_user_with_departure_logs(
        test_db,
        username="dim_user",
        n_logs=14,
        is_synthetic=False,
        bag_prepared_pct=0.7,
        phone_pct=0.4,
        item_search_pct=0.3,
        base_delay=5.0,
        prep_bias=4.0
    )

    logs = test_db.query(DepartureLog).filter(DepartureLog.user_id == user.id).all()
    dims = BehavioralDimensionCalculator.calculate_dimensions(logs)

    # Dimension A: Preparation
    prep = dims["preparation"]
    assert prep["observation_count"] == 14
    assert 28.0 <= prep["avg_actual_prep_min"] <= 30.0
    assert 3.5 <= prep["prep_bias_min"] <= 4.5
    assert prep["prep_duration_std"] > 0.0
    assert 0.0 <= prep["prep_consistency_score"] <= 1.0

    # Dimension B: Distraction
    dist = dims["distraction"]
    assert 0.2 <= dist["phone_usage_frequency"] <= 0.7
    assert dist["avg_phone_interruption_min"] > 0.0
    assert "Final 15" in dist["common_interruption_period"] or "Rarely observed" in dist["common_interruption_period"]

    # Dimension C: Organization
    org = dims["organization"]
    assert 0.2 <= org["item_search_frequency"] <= 0.6
    assert org["bag_prepared_consistency"] > 0.5

    # Dimension D: Time Estimation
    time_est = dims["time_estimation"]
    assert time_est["prep_estimation_bias_min"] > 0.0
    assert "Morning Office Commute" in time_est["event_specific_prep_bias"]

    # Dimension E: Departure
    dep = dims["departure"]
    assert dep["avg_lateness_min"] >= 4.0
    assert dep["late_departure_frequency"] > 0.8
    assert "Monday" in dep["day_of_week_patterns"]

    # Dimension F: Routine Consistency
    rout = dims["routine_consistency"]
    assert 0.0 <= rout["schedule_adherence_rate"] <= 1.0
    assert "weekday_delay_std" in rout


def test_multi_window_rolling_filtering(test_db):
    """Verify rolling window filtering (7d, 30d, all_time) relative to latest log."""
    user = create_user_with_departure_logs(
        test_db,
        username="window_user",
        n_logs=40,
        is_synthetic=False,
        start_date="2026-09-01"
    )

    all_logs = test_db.query(DepartureLog).filter(
        DepartureLog.user_id == user.id,
        DepartureLog.observation_status == "valid"
    ).all()

    logs_7d = PersonalizedBehavioralProfiler.filter_logs_by_window(all_logs, days=7)
    logs_30d = PersonalizedBehavioralProfiler.filter_logs_by_window(all_logs, days=30)

    # Latest date is around 2026-10-10 (40 days after Sept 1)
    # 7-day window should contain ~7 or 8 logs
    assert len(logs_7d) in (7, 8)
    # 30-day window should contain ~30 or 31 logs
    assert 30 <= len(logs_30d) <= 31
    assert len(all_logs) == 40


def test_data_sufficiency_lifecycle(test_db):
    """Verify state transitions: insufficient_data (<3), gathering_history (3..7), sufficient (>=8)."""
    # 1. Zero logs
    user = User(
        username="zero_user",
        hashed_password="hashed_pw",
        salt="test_salt",
        profile_type="it_professional"
    )
    test_db.add(user)
    test_db.commit()

    p_zero = PersonalizedBehavioralProfiler.compute_full_profile(user.id, test_db)
    assert p_zero["confidence_or_sufficiency_status"] == "insufficient_data"
    assert p_zero["observation_count"] == 0

    # 2. Insufficient logs (N = 2)
    user2 = create_user_with_departure_logs(test_db, "user2", n_logs=2)
    p_two = PersonalizedBehavioralProfiler.compute_full_profile(user2.id, test_db)
    assert p_two["confidence_or_sufficiency_status"] == "insufficient_data"
    assert p_two["observation_count"] == 2

    # 3. Gathering history (N = 5)
    user5 = create_user_with_departure_logs(test_db, "user5", n_logs=5)
    p_five = PersonalizedBehavioralProfiler.compute_full_profile(user5.id, test_db)
    assert p_five["confidence_or_sufficiency_status"] == "gathering_history"
    assert p_five["observation_count"] == 5

    # 4. Sufficient (N = 10)
    user10 = create_user_with_departure_logs(test_db, "user10", n_logs=10)
    p_ten = PersonalizedBehavioralProfiler.compute_full_profile(user10.id, test_db)
    assert p_ten["confidence_or_sufficiency_status"] == "sufficient"
    assert p_ten["observation_count"] == 10


def test_evidence_backed_insights(test_db):
    """Verify that insights generator outputs factual, non-hallucinatory findings."""
    user = create_user_with_departure_logs(
        test_db,
        username="insight_user",
        n_logs=15,
        prep_bias=7.0,
        phone_pct=0.6,
        base_delay=6.0
    )

    profile = PersonalizedBehavioralProfiler.compute_full_profile(user.id, test_db)
    insights = profile["insights"]
    assert len(insights) > 0

    # Every insight must contain required structural fields
    for ins in insights:
        assert "dimension" in ins
        assert "statement" in ins
        assert "supporting_metric" in ins
        assert "factual_evidence" in ins

    findings_text = " ".join([i["statement"] for i in insights])
    # Should detect preparation underestimation bias
    assert "underestimate" in findings_text.lower() or "prep" in findings_text.lower()


def test_snapshot_caching_and_auto_recalculation(test_db):
    """Verify profile caching and automatic recalculation after >= 3 new checkouts."""
    user = create_user_with_departure_logs(test_db, "cache_user", n_logs=8)

    # Initial profile creation -> persists snapshot
    prof1 = PersonalizedBehavioralProfiler.get_or_create_profile(user.id, test_db, provenance="real")
    assert prof1["cached"] is False
    assert prof1["observation_count"] == 8

    # Second call without changes -> returns cached snapshot
    prof2 = PersonalizedBehavioralProfiler.get_or_create_profile(user.id, test_db, provenance="real")
    assert prof2["cached"] is True
    assert prof2["profile_id"] == prof1["profile_id"]

    # Add only 1 new log -> should still return cached
    create_user_with_departure_logs(test_db, "cache_user_ignored", n_logs=0)
    # Add 1 log directly
    log1 = DepartureLog(
        user_id=user.id,
        event_id=1,
        event_name="Work",
        date="2026-11-01",
        day_of_week="Monday",
        planned_prep_start="2026-11-01T08:00:00",
        planned_departure="2026-11-01T08:30:00",
        actual_prep_start="2026-11-01T08:00:00",
        actual_departure="2026-11-01T08:33:00",
        actual_arrival="2026-11-01T08:50:00",
        estimated_prep_duration_min=20.0,
        estimated_travel_duration_min=15.0,
        actual_prep_duration_min=25.0,
        actual_travel_duration_min=17.0,
        total_delay_min=3.0,
        is_synthetic=False,
        provenance="real",
        observation_status="valid"
    )
    test_db.add(log1)
    test_db.commit()

    prof3 = PersonalizedBehavioralProfiler.get_or_create_profile(user.id, test_db, provenance="real")
    assert prof3["cached"] is True

    # Add 2 more logs (total 3 new logs since snapshot) -> auto-recalculation triggered!
    for j in range(2):
        d_str = f"2026-11-0{j+2}"
        test_db.add(DepartureLog(
            user_id=user.id,
            event_id=1,
            event_name="Work",
            date=d_str,
            day_of_week="Tuesday",
            planned_prep_start=f"{d_str}T08:00:00",
            planned_departure=f"{d_str}T08:30:00",
            actual_prep_start=f"{d_str}T08:00:00",
            actual_departure=f"{d_str}T08:33:00",
            actual_arrival=f"{d_str}T08:50:00",
            estimated_prep_duration_min=20.0,
            estimated_travel_duration_min=15.0,
            actual_prep_duration_min=25.0,
            actual_travel_duration_min=17.0,
            total_delay_min=3.0,
            is_synthetic=False,
            provenance="real",
            observation_status="valid"
        ))
    test_db.commit()

    prof4 = PersonalizedBehavioralProfiler.get_or_create_profile(user.id, test_db, provenance="real")
    assert prof4["cached"] is False
    assert prof4["observation_count"] == 11
    assert prof4["profile_id"] != prof1["profile_id"]


def test_profile_comparison_and_deltas(test_db):
    """Verify comparing two profile periods accurately computes differences and trends."""
    user = create_user_with_departure_logs(
        test_db,
        username="compare_user",
        n_logs=25,
        is_synthetic=False,
        prep_bias=4.0,
        base_delay=5.0
    )

    prof = PersonalizedBehavioralProfiler.get_or_create_profile(user.id, test_db)
    dims_all = prof["dimensions"]["all_time"]
    dims_7d = prof["dimensions"]["window_7d"]["metrics"]

    comparison = PersonalizedBehavioralProfiler.compare_profile_periods(
        {"profile_id": "window_all_time", "dimensions": dims_all},
        {"profile_id": "window_7d", "dimensions": dims_7d}
    )

    assert "deltas" in comparison
    assert "prep_bias_min_delta" in comparison["deltas"]
    assert "avg_lateness_min_delta" in comparison["deltas"]
    assert comparison["overall_trend"] in ("improving", "regressing", "stable")
    assert len(comparison["trend_summary"]) > 0


def test_synthetic_vs_real_profile_isolation(test_db):
    """Verify synthetic logs do not contaminate genuine behavioral profiles."""
    user = create_user_with_departure_logs(
        test_db,
        username="isolation_user",
        n_logs=12,
        is_synthetic=True,
        prep_bias=10.0
    )

    # Real profile query when only synthetic exists
    prof_real = PersonalizedBehavioralProfiler.get_or_create_profile(user.id, test_db, provenance="real")
    assert prof_real["observation_count"] == 0
    assert prof_real["confidence_or_sufficiency_status"] == "insufficient_data"

    # Synthetic profile query
    prof_synth = PersonalizedBehavioralProfiler.get_or_create_profile(user.id, test_db, provenance="synthetic")
    assert prof_synth["observation_count"] == 12
    assert prof_synth["confidence_or_sufficiency_status"] == "sufficient"


def test_fastapi_profile_endpoints_integration(test_db):
    """Verify FastAPI GET /api/intelligence/profile and related endpoints."""
    user = create_user_with_departure_logs(test_db, "api_user", n_logs=10, is_synthetic=False)

    app.dependency_overrides[get_db] = lambda: test_db
    client = TestClient(app)

    token = security.create_access_token(data={"sub": user.username})
    headers = {"Authorization": f"Bearer {token}"}

    # 1. GET /api/intelligence/profile
    res = client.get("/api/intelligence/profile", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == user.id
    assert data["observation_count"] == 10
    assert data["confidence_or_sufficiency_status"] == "sufficient"
    assert "dimensions" in data
    assert "all_time" in data["dimensions"]
    assert "insights" in data

    # 2. GET /api/intelligence/profile/history
    res_hist = client.get("/api/intelligence/profile/history", headers=headers)
    assert res_hist.status_code == 200
    history = res_hist.json()
    assert isinstance(history, list)
    assert len(history) >= 1

    # 3. GET /api/intelligence/profile/compare
    res_comp = client.get("/api/intelligence/profile/compare?window_a=all_time&window_b=7d", headers=headers)
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    assert "deltas" in comp_data
    assert "overall_trend" in comp_data

    # 4. POST /api/intelligence/profile/refresh
    res_ref = client.post("/api/intelligence/profile/refresh", headers=headers)
    assert res_ref.status_code == 200
    ref_data = res_ref.json()
    assert ref_data["cached"] is False

    # 5. Access Isolation: Unauthorized user attempting to access profile comparison
    other_user = create_user_with_departure_logs(test_db, "other_user", n_logs=5, is_synthetic=False)
    prof_other = PersonalizedBehavioralProfiler.get_or_create_profile(other_user.id, test_db)

    # API user tries to compare with other_user's profile_id
    res_cross = client.get(
        f"/api/intelligence/profile/compare?profile_id_a={data['profile_id']}&profile_id_b={prof_other['profile_id']}",
        headers=headers
    )
    # Must fail because api_user does not own other_user's profile
    assert res_cross.status_code == 400

    app.dependency_overrides.clear()
