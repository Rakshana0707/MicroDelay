"""Comprehensive Test Suite for Prompt 2.7: Adaptive Personalized Recommendation Engine.

Covers:
- Recommendation eligibility linked to empirical behavioral evidence.
- Multi-criteria adaptive prioritization and personalized learning.
- Duplicate prevention and fatigue cooling penalties.
- Full intervention lifecycle state machine.
- User feedback and rating impact.
- Empirical outcome measurement and confounding factor detection.
- Handling insufficient baseline and post-intervention logs.
- Strict user data isolation.
- FastAPI REST API endpoints integration.
- Backward compatibility with existing recommendation engines.
"""

import pytest
import datetime
import json
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, User, DepartureLog, UserInterventionRecord, migrate_db
from main import app, get_db
from security import create_access_token
from ml.recommendations import (
    RECOMMENDATION_CATALOG,
    RecommendationEligibilityEvaluator,
    AdaptivePrioritizer,
    EffectivenessMeasurementService,
    InterventionService,
    INTERVENTION_CAUSATION_DISCLAIMER
)
from ml.adaptive.adaptive_recommendations import AdaptiveRecommendationEngine
from ml.adaptive.context import PredictionContext


# Setup isolated in-memory SQLite database per test
@pytest.fixture
def test_db():
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


def create_test_user(db, username="rec_tester", profile_type="student") -> User:
    user = User(
        username=username,
        hashed_password="hashed_pw_test",
        salt="test_salt",
        profile_type=profile_type,
        consent_to_global_training=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_test_log(
    db,
    user_id: int,
    total_delay_min: float = 4.0,
    bag_prepared: bool = True,
    finding_items: bool = False,
    delay_finding_items_min: float = 0.0,
    phone_distraction: bool = False,
    delay_phone_min: float = 0.0,
    one_last_task: bool = False,
    delay_one_last_task_min: float = 0.0,
    actual_prep: float = 35.0,
    estimated_prep: float = 25.0,
    day_of_week: str = "Tuesday",
    recorded_at: Optional[str] = None
) -> DepartureLog:
    rec_time = recorded_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
    log = DepartureLog(
        user_id=user_id,
        event_name="Morning Class",
        date="2026-11-18",
        day_of_week=day_of_week,
        planned_prep_start="08:00:00",
        planned_departure="08:30:00",
        estimated_prep_duration_min=estimated_prep,
        estimated_travel_duration_min=15.0,
        actual_prep_start="08:00:00",
        actual_departure="08:34:00",
        actual_arrival="08:50:00",
        actual_prep_duration_min=actual_prep,
        actual_travel_duration_min=16.0,
        total_delay_min=total_delay_min,
        bag_prepared=bag_prepared,
        tag_finding_items=finding_items,
        delay_finding_items_min=delay_finding_items_min,
        tag_phone_usage=phone_distraction,
        delay_phone_usage_min=delay_phone_min,
        tag_one_last_task=one_last_task,
        delay_one_last_task_min=delay_one_last_task_min,
        tag_waiting=False,
        delay_waiting_min=0.0,
        is_synthetic=False,
        provenance="genuine",
        observation_status="valid",
        recorded_at=rec_time
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


# =====================================================================
# TEST CASES
# =====================================================================

def test_recommendation_catalog_structure():
    """Verify catalog coverage across Preparation, Distraction, Timing, and Routine Consistency."""
    assert len(RECOMMENDATION_CATALOG) >= 12
    categories = {item.category for item in RECOMMENDATION_CATALOG.values()}
    assert "preparation" in categories
    assert "distraction" in categories
    assert "timing" in categories
    assert "routine_consistency" in categories

    for cat_id, item in RECOMMENDATION_CATALOG.items():
        assert item.catalog_id == cat_id
        assert len(item.action) > 0
        assert len(item.description) > 0
        assert len(item.implementation_steps) >= 2
        assert item.base_delay_reduction_min > 0.0
        assert item.implementation_effort in ("low", "medium", "high")


def test_recommendation_eligibility_linking(test_db):
    """Verify that eligible interventions strictly link to empirical user behavioral problems."""
    db = test_db
    user = create_test_user(db, username="eligibility_user")

    # Generate logs with heavy item search overhead and prep underestimation
    for _ in range(5):
        create_test_log(
            db, user.id,
            total_delay_min=6.0,
            finding_items=True,
            delay_finding_items_min=4.5,
            phone_distraction=False,
            bag_prepared=True,
            actual_prep=40.0,
            estimated_prep=25.0  # +15m prep bias
        )

    user_logs = db.query(DepartureLog).filter(DepartureLog.user_id == user.id).all()
    eligible = RecommendationEligibilityEvaluator.evaluate_user_eligibility(user_logs)
    eligible_ids = {e.catalog_item.catalog_id for e in eligible}

    # Item search and prep buffer interventions MUST be eligible
    assert "prep_fixed_item_location" in eligible_ids
    assert "time_increase_prep_buffer" in eligible_ids

    # Unrelated phone distraction intervention should NOT be triggered
    assert "distract_prep_focus_mode" not in eligible_ids

    # Check supporting evidence contract
    item_search_rec = next(e for e in eligible if e.catalog_item.catalog_id == "prep_fixed_item_location")
    assert item_search_rec.supporting_evidence["sample_count"] == 5
    assert item_search_rec.supporting_evidence["problem_frequency_pct"] == 100.0
    assert item_search_rec.calibrated_delay_reduction_min >= 4.0
    assert "item searches" in item_search_rec.supporting_evidence["supporting_statement"].lower()


def test_prioritization_and_fatigue_cooling(test_db):
    """Verify adaptive scoring, ranking, feedback weighting, and dismissal fatigue cooling."""
    db = test_db
    user = create_test_user(db, username="prioritize_user")

    for _ in range(6):
        create_test_log(
            db, user.id,
            total_delay_min=5.0,
            finding_items=True,
            delay_finding_items_min=3.5,
            bag_prepared=False
        )

    user_logs = db.query(DepartureLog).filter(DepartureLog.user_id == user.id).all()
    eligible = RecommendationEligibilityEvaluator.evaluate_user_eligibility(user_logs)

    # Initial ranking without history
    ranked_initial = AdaptivePrioritizer.rank_interventions(eligible, [])
    assert len(ranked_initial) >= 2
    top_cat_id = ranked_initial[0][0].catalog_item.catalog_id

    # Simulate user dismissing the top recommendation today
    now = datetime.datetime.now(datetime.timezone.utc)
    dismissed_rec = UserInterventionRecord(
        intervention_id="intv_dismissed_1",
        user_id=user.id,
        recommendation_id=top_cat_id,
        category="preparation",
        action="Dismissed Action",
        description="Dismissed Description",
        implementation_steps="[]",
        behavioral_problem="some_problem",
        supporting_evidence="{}",
        status="dismissed",
        expected_outcome="{}",
        causation_disclaimer=INTERVENTION_CAUSATION_DISCLAIMER,
        generated_at=now.isoformat(),
        dismissed_at=now.isoformat()
    )
    db.add(dismissed_rec)
    db.commit()

    # Re-rank with dismissal history
    history = db.query(UserInterventionRecord).filter(UserInterventionRecord.user_id == user.id).all()
    ranked_after_dismiss = AdaptivePrioritizer.rank_interventions(eligible, history, now=now)

    # The dismissed item must be heavily penalized or demoted
    dismissed_score = next(r[1] for r in ranked_after_dismiss if r[0].catalog_item.catalog_id == top_cat_id)
    initial_score = next(r[1] for r in ranked_initial if r[0].catalog_item.catalog_id == top_cat_id)
    assert dismissed_score < (initial_score * 0.40)


def test_snooze_and_active_exclusion(test_db):
    """Verify that actively accepted or currently snoozed interventions are excluded from suggestions."""
    db = test_db
    user = create_test_user(db, username="snooze_user")

    for _ in range(4):
        create_test_log(db, user.id, bag_prepared=False)

    user_logs = db.query(DepartureLog).filter(DepartureLog.user_id == user.id).all()
    eligible = RecommendationEligibilityEvaluator.evaluate_user_eligibility(user_logs)

    now = datetime.datetime.now(datetime.timezone.utc)
    snooze_until = (now + datetime.timedelta(days=5)).isoformat()

    snoozed_rec = UserInterventionRecord(
        intervention_id="intv_snoozed_1",
        user_id=user.id,
        recommendation_id="prep_bag_previous_night",
        category="preparation",
        action="Snoozed Action",
        description="Desc",
        implementation_steps="[]",
        behavioral_problem="unprepared_bag",
        supporting_evidence="{}",
        status="snoozed",
        snoozed_until=snooze_until,
        expected_outcome="{}",
        causation_disclaimer=INTERVENTION_CAUSATION_DISCLAIMER,
        generated_at=now.isoformat()
    )
    db.add(snoozed_rec)
    db.commit()

    history = [snoozed_rec]
    ranked = AdaptivePrioritizer.rank_interventions(eligible, history, now=now)

    # Snoozed item should be excluded completely (score 0.0)
    snoozed_items = [r for r in ranked if r[0].catalog_item.catalog_id == "prep_bag_previous_night"]
    assert len(snoozed_items) == 0


def test_full_intervention_lifecycle(test_db):
    """Verify complete intervention lifecycle: suggest -> accept (baseline) -> complete (effectiveness)."""
    db = test_db
    user = create_test_user(db, username="lifecycle_user")

    # 1. Seed baseline departure logs
    base_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)
    for i in range(5):
        t = (base_time + datetime.timedelta(days=i)).isoformat()
        create_test_log(
            db, user.id,
            total_delay_min=8.0,
            finding_items=True,
            delay_finding_items_min=4.0,
            recorded_at=t
        )

    # 2. Generate personalized recommendations
    recs = InterventionService.get_personalized_recommendations(db, user.id, limit=2)
    assert len(recs) >= 1
    intv_id = recs[0]["intervention_id"]
    assert recs[0]["status"] == "suggested"

    # 3. Accept intervention
    accepted = InterventionService.accept_intervention(db, user.id, intv_id)
    assert accepted["status"] == "accepted"
    assert accepted["accepted_at"] is not None
    assert accepted["baseline_metrics"] is not None
    assert accepted["baseline_metrics"]["avg_departure_delay"] == 8.0
    assert accepted["baseline_metrics"]["sample_count"] == 5

    # Backdate started_at to 4 days ago so post-intervention checkouts fall within active trial period
    intv_obj = db.query(UserInterventionRecord).filter(UserInterventionRecord.intervention_id == intv_id).first()
    intv_obj.started_at = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=4)).isoformat()
    db.commit()

    # 4. Log subsequent post-intervention checkouts with improved outcomes
    post_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)
    for j in range(3):
        t = (post_time + datetime.timedelta(days=j)).isoformat()
        create_test_log(
            db, user.id,
            total_delay_min=1.5,  # Substantial improvement: down from 8.0m to 1.5m!
            finding_items=False,
            delay_finding_items_min=0.0,
            recorded_at=t
        )

    # 5. Mark intervention as completed with feedback
    completed = InterventionService.complete_intervention(
        db, user.id, intv_id,
        rating=5,
        notes="Placing keys in the front bowl completely solved my morning searching!",
        helpful=True
    )

    assert completed["status"] == "completed"
    assert completed["user_rating"] == 5
    assert completed["user_helpful"] is True
    assert completed["completion_status"] == "success"

    # 6. Verify measured outcome
    outcome = completed["measured_outcome"]
    assert outcome["status"] == "evaluated"
    assert outcome["is_improved"] is True
    assert outcome["baseline_avg_delay"] == 8.0
    assert outcome["post_avg_delay"] == 1.5
    assert outcome["delay_delta_min"] == -6.5  # Saved 6.5 minutes!
    assert outcome["baseline_sample_count"] == 5
    assert outcome["post_sample_count"] == 3


def test_insufficient_post_observations_handling(test_db):
    """Verify graceful handling when an intervention is completed with insufficient post logs."""
    db = test_db
    user = create_test_user(db, username="insufficient_user")

    # Baseline log
    create_test_log(db, user.id, total_delay_min=6.0)

    recs = InterventionService.get_personalized_recommendations(db, user.id, limit=1)
    intv_id = recs[0]["intervention_id"]
    InterventionService.accept_intervention(db, user.id, intv_id)

    # Complete immediately without logging post-intervention checkouts
    completed = InterventionService.complete_intervention(db, user.id, intv_id, rating=4)
    outcome = completed["measured_outcome"]
    assert outcome["status"] == "insufficient_post_observations"
    assert outcome["post_sample_count"] == 0
    assert "At least 2 departure routines" in outcome["message"]


def test_user_data_isolation(test_db):
    """Verify that User A cannot view, accept, dismiss, or modify User B's interventions."""
    db = test_db
    user_a = create_test_user(db, username="user_a")
    user_b = create_test_user(db, username="user_b")

    create_test_log(db, user_a.id, total_delay_min=5.0, bag_prepared=False)
    recs_a = InterventionService.get_personalized_recommendations(db, user_a.id, limit=1)
    intv_a_id = recs_a[0]["intervention_id"]

    # User B attempts to accept User A's intervention -> MUST FAIL
    with pytest.raises(ValueError, match="Intervention not found or access denied"):
        InterventionService.accept_intervention(db, user_b.id, intv_a_id)

    # User B attempts to complete User A's intervention -> MUST FAIL
    with pytest.raises(ValueError, match="Intervention not found or access denied"):
        InterventionService.complete_intervention(db, user_b.id, intv_a_id, rating=5)

    # User B querying their interventions should see zero records from User A
    interventions_b = InterventionService.get_user_interventions(db, user_b.id)
    assert len(interventions_b) == 0


def test_fastapi_intervention_endpoints(test_db):
    """Verify full FastAPI REST API intervention endpoints integration."""
    db = test_db
    user = create_test_user(db, username="api_user")
    token = create_access_token({"sub": user.username})
    headers = {"Authorization": f"Bearer {token}"}

    create_test_log(db, user.id, total_delay_min=7.0, finding_items=True, delay_finding_items_min=3.5)

    app.dependency_overrides[get_db] = lambda: db
    client = TestClient(app)

    try:
        # 1. GET /api/recommendations/personalized
        res_recs = client.get("/api/recommendations/personalized?limit=2", headers=headers)
        assert res_recs.status_code == 200
        recs_data = res_recs.json()
        assert isinstance(recs_data, list)
        assert len(recs_data) >= 1
        intv_id = recs_data[0]["intervention_id"]

        # 2. POST /api/interventions/accept
        res_accept = client.post("/api/interventions/accept", json={"intervention_id": intv_id}, headers=headers)
        assert res_accept.status_code == 200
        acc_data = res_accept.json()
        assert acc_data["status"] == "accepted"
        assert acc_data["baseline_metrics"] is not None

        # 3. GET /api/interventions
        res_list = client.get("/api/interventions?status=accepted", headers=headers)
        assert res_list.status_code == 200
        assert len(res_list.json()) == 1

        # 4. POST /api/interventions/feedback
        res_fb = client.post(
            "/api/interventions/feedback",
            json={"intervention_id": intv_id, "rating": 5, "notes": "Great guidance", "helpful": True},
            headers=headers
        )
        assert res_fb.status_code == 200
        assert res_fb.json()["user_rating"] == 5

        # 5. POST /api/interventions/snooze
        res_snooze = client.post("/api/interventions/snooze", json={"intervention_id": intv_id, "days": 5}, headers=headers)
        assert res_snooze.status_code == 200
        assert res_snooze.json()["status"] == "snoozed"

        # 6. GET /api/interventions/{id}/effectiveness
        res_eff = client.get(f"/api/interventions/{intv_id}/effectiveness", headers=headers)
        assert res_eff.status_code == 200
        eff_data = res_eff.json()
        assert "baseline_metrics" in eff_data
        assert "causation_disclaimer" in eff_data
    finally:
        app.dependency_overrides.clear()


def test_backward_compatibility_with_adaptive_engine():
    """Verify that existing AdaptiveRecommendationEngine and orchestrator contracts are preserved."""
    engine = AdaptiveRecommendationEngine()
    ctx = PredictionContext(
        user_id=1,
        event_name="Office Standup",
        day_of_week="Monday",
        event_importance="high",
        estimated_prep_duration=25.0,
        estimated_travel_duration=20.0,
        bag_prepared=False
    )

    recs = engine.generate_recommendations(
        context=ctx,
        predicted_delay_min=6.5,
        contributing_factors=[{"factor": "bag_unprepared", "impact_minutes": 3.8}],
        profile_data={"prep_bias_min": 4.2}
    )

    assert isinstance(recs, list)
    assert len(recs) >= 1
    # Check original contract keys
    for r in recs:
        assert "action" in r
        assert "description" in r
        assert "expected_delay_reduction_min" in r
        assert "category" in r
        assert "priority" in r
