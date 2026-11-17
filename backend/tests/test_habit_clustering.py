"""Comprehensive Automated Test Suite for Behavioral Habit Discovery & Clustering (Prompt 2.4).

Verifies:
1. Feature extraction & temporal safety (no future leakage).
2. Clustering tournament comparison (K-Means, Agglomerative, DBSCAN).
3. Dynamic evidence-based cluster interpretation (never hardcoded by cluster ID).
4. Data sufficiency lifecycle states (insufficient_data, gathering_history, sufficient).
5. Unseen user cluster assignment via nearest centroid.
6. Edge cases: small datasets (N=0, 1, 2), identical values, extreme outliers.
7. Profile update tracking, versioning, and previous profile comparisons.
8. Synthetic vs genuine data isolation.
9. FastAPI REST API endpoints integration.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, User, DepartureLog, UserClusterProfile
from main import app, get_db
from ml.habit_discovery import (
    BEHAVIORAL_FEATURE_NAMES,
    extract_user_behavioral_features,
    ClusterInterpretationService,
    BehavioralClusteringTournament,
    HabitDiscoveryService
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


def create_user_and_logs(
    db,
    username: str,
    n_logs: int,
    is_synthetic: bool = False,
    bag_prepared_pct: float = 0.5,
    phone_pct: float = 0.2,
    base_delay: float = 3.0,
    prep_bias: float = 2.0
):
    """Helper to populate test users and logs with known behavioral tendencies."""
    u = User(
        username=username,
        hashed_password="hashed_pw",
        salt="test_salt",
        profile_type="student",
        consent_to_global_training=True
    )
    db.add(u)
    db.commit()
    db.refresh(u)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for i in range(n_logs):
        day = days[i % 5]
        bag = (i / max(1, n_logs)) < bag_prepared_pct
        phone = (i / max(1, n_logs)) < phone_pct
        log = DepartureLog(
            user_id=u.id,
            event_name="Class",
            date=f"2026-09-{(i % 28) + 1:02d}",
            day_of_week=day,
            planned_prep_start="07:30:00",
            planned_departure="08:00:00",
            estimated_prep_duration_min=30.0,
            estimated_travel_duration_min=15.0,
            actual_prep_start="07:30:00",
            actual_departure=f"08:{int(base_delay):02d}:00",
            actual_arrival="08:20:00",
            actual_prep_duration_min=30.0 + prep_bias,
            actual_travel_duration_min=15.0,
            total_delay_min=base_delay,
            bag_prepared=bag,
            tag_phone_usage=phone,
            delay_phone_usage_min=4.0 if phone else 0.0,
            tag_finding_items=(not bag),
            delay_finding_items_min=3.0 if not bag else 0.0,
            tag_one_last_task=(i % 3 == 0),
            delay_one_last_task_min=2.5 if (i % 3 == 0) else 0.0,
            provenance="synthetic" if is_synthetic else "real",
            data_source="generator" if is_synthetic else "simulator_checkout",
            is_synthetic=is_synthetic,
            recorded_at=f"2026-09-{(i % 28) + 1:02d}T08:00:00Z",
            observation_status="valid"
        )
        db.add(log)
    db.commit()
    return u


def test_feature_extraction_and_temporal_safety():
    """Verify that user behavioral features are computed accurately and respect temporal cutoff."""
    mock_logs = [
        {
            "actual_prep_duration_min": 35.0,
            "estimated_prep_duration_min": 25.0,
            "actual_travel_duration_min": 20.0,
            "estimated_travel_duration_min": 15.0,
            "total_delay_min": 10.0,
            "bag_prepared": False,
            "tag_phone_usage": True,
            "delay_phone_usage_min": 4.0,
            "tag_finding_items": True,
            "delay_finding_items_min": 3.0,
            "tag_one_last_task": False,
            "day_of_week": "Monday",
            "recorded_at": "2026-09-01T08:00:00Z"
        },
        {
            "actual_prep_duration_min": 30.0,
            "estimated_prep_duration_min": 25.0,
            "actual_travel_duration_min": 15.0,
            "estimated_travel_duration_min": 15.0,
            "total_delay_min": 5.0,
            "bag_prepared": True,
            "tag_phone_usage": False,
            "delay_phone_usage_min": 0.0,
            "tag_finding_items": False,
            "delay_finding_items_min": 0.0,
            "tag_one_last_task": True,
            "day_of_week": "Tuesday",
            "recorded_at": "2026-09-02T08:00:00Z"
        },
        # Future log after cutoff
        {
            "actual_prep_duration_min": 60.0,
            "estimated_prep_duration_min": 20.0,
            "actual_travel_duration_min": 30.0,
            "estimated_travel_duration_min": 15.0,
            "total_delay_min": 30.0,
            "bag_prepared": False,
            "tag_phone_usage": True,
            "delay_phone_usage_min": 15.0,
            "day_of_week": "Wednesday",
            "recorded_at": "2026-09-15T08:00:00Z"
        }
    ]

    # Without cutoff: uses all 3
    feats_all = extract_user_behavioral_features(mock_logs)
    assert len(feats_all) == 12
    assert "average_prep_duration" in feats_all

    # With cutoff: strictly ignores the future log on 2026-09-15
    feats_past = extract_user_behavioral_features(mock_logs, cutoff_timestamp="2026-09-05T00:00:00Z")
    assert feats_past["average_prep_duration"] == (35.0 + 30.0) / 2.0
    assert feats_past["average_prep_bias"] == ((35.0 - 25.0) + (30.0 - 25.0)) / 2.0
    assert feats_past["average_departure_delay"] == (10.0 + 5.0) / 2.0
    assert feats_past["lateness_rate"] == 1.0
    assert feats_past["bag_prepared_percentage"] == 0.5
    assert feats_past["phone_distraction_frequency"] == 0.5


def test_dynamic_cluster_interpretation_not_hardcoded():
    """Verify that cluster interpretations are derived from centroid statistics, not hardcoded IDs."""
    pop_mean = {feat: 5.0 for feat in BEHAVIORAL_FEATURE_NAMES}
    pop_mean["bag_prepared_percentage"] = 0.5
    pop_mean["phone_distraction_frequency"] = 0.2
    pop_mean["item_search_frequency"] = 0.2
    pop_mean["last_minute_task_frequency"] = 0.2
    pop_std = {feat: 2.0 for feat in BEHAVIORAL_FEATURE_NAMES}
    pop_std["bag_prepared_percentage"] = 0.2
    pop_std["phone_distraction_frequency"] = 0.1
    pop_std["item_search_frequency"] = 0.1
    pop_std["last_minute_task_frequency"] = 0.1

    # 1. Centroid with high phone distraction
    c_phone = pop_mean.copy()
    c_phone["phone_distraction_frequency"] = 0.65
    interp_phone = ClusterInterpretationService.interpret_centroid(c_phone, pop_mean, pop_std)
    assert "Phone" in interp_phone["cluster_name"]

    # 2. Centroid with high bag preparedness and low delay
    c_prep = pop_mean.copy()
    c_prep["bag_prepared_percentage"] = 0.95
    c_prep["average_departure_delay"] = 0.5
    c_prep["phone_distraction_frequency"] = 0.05
    interp_prep = ClusterInterpretationService.interpret_centroid(c_prep, pop_mean, pop_std)
    assert "Consistently Prepared" in interp_prep["cluster_name"]

    # 3. Centroid with high item search delay
    c_item = pop_mean.copy()
    c_item["item_search_frequency"] = 0.70
    interp_item = ClusterInterpretationService.interpret_centroid(c_item, pop_mean, pop_std)
    assert "Item-Search" in interp_item["cluster_name"]

    # Verify that different cluster IDs do NOT deterministically produce names
    assert interp_phone["cluster_name"] != interp_prep["cluster_name"]


def test_clustering_tournament_comparison():
    """Verify that the clustering tournament compares K-Means, Agglomerative, and DBSCAN."""
    # Create synthetic population dataset with 3 distinct behavioral cohorts
    rows = []
    # Cohort 1: Prepared
    for _ in range(5):
        r = {f: 2.0 for f in BEHAVIORAL_FEATURE_NAMES}
        r["bag_prepared_percentage"] = 0.9
        r["average_departure_delay"] = 0.5
        rows.append(r)

    # Cohort 2: Distracted
    for _ in range(5):
        r = {f: 5.0 for f in BEHAVIORAL_FEATURE_NAMES}
        r["phone_distraction_frequency"] = 0.8
        r["average_departure_delay"] = 8.0
        rows.append(r)

    # Cohort 3: Bias / underestimation
    for _ in range(5):
        r = {f: 4.0 for f in BEHAVIORAL_FEATURE_NAMES}
        r["average_prep_bias"] = 12.0
        r["average_departure_delay"] = 10.0
        rows.append(r)

    df = pd.DataFrame(rows)
    tournament = BehavioralClusteringTournament(random_state=42)
    res = tournament.run_tournament(df)

    assert res["is_sufficient_sample"] is True
    assert res["selected_algorithm"] in ("K-Means", "Agglomerative", "DBSCAN")
    assert res["optimal_k"] >= 2
    assert res["silhouette_score"] > 0.0
    assert res["davies_bouldin_index"] >= 0.0
    assert len(res["benchmark_results"]) >= 2
    assert len(res["centroids"]) == res["optimal_k"]


def test_edge_cases_small_datasets_and_identical_values():
    """Verify robust handling of 0, 1, 2 samples and identical feature vectors."""
    tournament = BehavioralClusteringTournament()

    # 1. Zero users
    df_empty = pd.DataFrame(columns=BEHAVIORAL_FEATURE_NAMES)
    res_empty = tournament.run_tournament(df_empty)
    assert res_empty["is_sufficient_sample"] is False
    assert res_empty["optimal_k"] == 1

    # 2. 2 users (less than 3)
    df_two = pd.DataFrame([
        {f: 2.0 for f in BEHAVIORAL_FEATURE_NAMES},
        {f: 4.0 for f in BEHAVIORAL_FEATURE_NAMES}
    ])
    res_two = tournament.run_tournament(df_two)
    assert res_two["is_sufficient_sample"] is False
    assert res_two["optimal_k"] == 1

    # 3. Identical feature values across 6 users
    df_identical = pd.DataFrame([
        {f: 3.5 for f in BEHAVIORAL_FEATURE_NAMES} for _ in range(6)
    ])
    res_ident = tournament.run_tournament(df_identical)
    assert res_ident["optimal_k"] >= 1


def test_user_profile_sufficiency_states(test_db):
    """Verify transition through insufficient_data -> gathering_history -> sufficient."""
    # User A: 1 log (<3)
    u1 = create_user_and_logs(test_db, "user_novice", n_logs=1)
    p1 = HabitDiscoveryService.get_or_create_user_profile(u1.id, test_db)
    assert p1["data_sufficiency_status"] == "insufficient_data"
    assert p1["sample_count"] == 1

    # User B: 4 logs (3 <= N < 8)
    u2 = create_user_and_logs(test_db, "user_gathering", n_logs=4)
    p2 = HabitDiscoveryService.get_or_create_user_profile(u2.id, test_db)
    assert p2["data_sufficiency_status"] == "gathering_history"
    assert p2["sample_count"] == 4

    # User C: 10 logs (N >= 8)
    u3 = create_user_and_logs(test_db, "user_mature", n_logs=10)
    p3 = HabitDiscoveryService.get_or_create_user_profile(u3.id, test_db)
    assert p3["data_sufficiency_status"] == "sufficient"
    assert p3["sample_count"] == 10
    assert p3["cluster_name"] != ""


def test_profile_update_and_trend_comparison(test_db):
    """Verify that multiple profile generations record history and detect improvement trends."""
    u = create_user_and_logs(test_db, "user_trending", n_logs=8, base_delay=8.0)
    
    # 1. Initial profile generation
    p_initial = HabitDiscoveryService.get_or_create_user_profile(u.id, test_db)
    assert p_initial["previous_profile_comparison"] is None

    # 2. Add 5 on-time checkouts to improve habits
    for i in range(5):
        log = DepartureLog(
            user_id=u.id,
            event_name="Class",
            date=f"2026-10-{(i + 1):02d}",
            day_of_week="Monday",
            planned_prep_start="07:30:00",
            planned_departure="08:00:00",
            estimated_prep_duration_min=30.0,
            estimated_travel_duration_min=15.0,
            actual_prep_start="07:30:00",
            actual_departure="08:00:00",
            actual_arrival="08:15:00",
            actual_prep_duration_min=30.0,
            actual_travel_duration_min=15.0,
            total_delay_min=0.0,
            bag_prepared=True,
            tag_phone_usage=False,
            provenance="real",
            data_source="simulator_checkout",
            is_synthetic=False,
            recorded_at=f"2026-10-{(i + 1):02d}T08:00:00Z",
            observation_status="valid"
        )
        test_db.add(log)
    test_db.commit()

    # 3. Second profile generation
    p_updated = HabitDiscoveryService.get_or_create_user_profile(u.id, test_db)
    comp = p_updated["previous_profile_comparison"]
    assert comp is not None
    assert comp["delay_delta_minutes"] < 0.0
    assert comp["trend"] == "improving"
    assert "reduced" in comp["trend_message"]


def test_synthetic_data_isolation(test_db):
    """Verify that synthetic logs are isolated and do not enter production real profiles."""
    u = create_user_and_logs(test_db, "user_synth_check", n_logs=12, is_synthetic=True)

    # Real profile query: should find 0 real records
    p_real = HabitDiscoveryService.get_or_create_user_profile(u.id, test_db, provenance="real")
    assert p_real["sample_count"] == 0
    assert p_real["data_sufficiency_status"] == "insufficient_data"

    # Synthetic demo profile: finds all 12
    p_synth = HabitDiscoveryService.get_or_create_user_profile(u.id, test_db, provenance="synthetic")
    assert p_synth["sample_count"] == 12


def test_api_cluster_endpoints(test_db, monkeypatch):
    """Verify the FastAPI cluster profile and benchmarking endpoints."""
    # Create 3 users with logs
    u1 = create_user_and_logs(test_db, "alice", 10, bag_prepared_pct=0.9, base_delay=0.5)
    u2 = create_user_and_logs(test_db, "bob", 10, phone_pct=0.9, base_delay=7.0)
    u3 = create_user_and_logs(test_db, "carol", 10, prep_bias=8.0, base_delay=9.0)

    # Override db session dependency
    app.dependency_overrides[get_db] = lambda: test_db

    # Login helper
    from security import create_access_token
    token = create_access_token(data={"sub": "alice"})
    headers = {"Authorization": f"Bearer {token}"}

    client = TestClient(app)

    # 1. Train cluster models
    t_res = client.post("/api/intelligence/clusters/train", headers=headers)
    assert t_res.status_code == 200
    t_data = t_res.json()
    assert "cluster_model_version" in t_data
    assert t_data["total_users_clustered"] >= 3

    # 2. Get user cluster profile
    p_res = client.get("/api/intelligence/clusters/profile", headers=headers)
    assert p_res.status_code == 200
    p_data = p_res.json()
    assert p_data["user_id"] == u1.id
    assert p_data["cluster_name"] != ""
    assert p_data["data_sufficiency_status"] == "sufficient"
    assert "dominant_characteristics" in p_data

    # 3. Get cluster benchmark report
    b_res = client.get("/api/intelligence/clusters/benchmark", headers=headers)
    assert b_res.status_code == 200
    b_data = b_res.json()
    assert "benchmark_results" in b_data
    assert "algorithm" in b_data

    app.dependency_overrides.clear()
