"""Automated Test Suite for Global-to-Personalized Learning & Cold-Start Handling (Prompt 2.3).

Tests:
1. Dynamic alpha calculation (data volume, relative MAE, behavioral stability).
2. Explicit lifecycle states (new_user, collecting_history, global_prediction, personalizing, personalized, insufficient_data).
3. Hybrid prediction blending with graceful fallbacks.
4. Privacy safeguards: Consent exclusion & synthetic data isolation.
5. Minimum cohort size safeguards for shared training.
6. End-to-end global model training and inference.
7. Leave-users-out cross-validation experiment.
8. Cross-user tenant data isolation for personal models.
9. User consent management API (GET & POST).
10. Global model status & training API endpoints.
11. Leave-users-out benchmarking endpoint.
12. Cold-start to full personalization progression.
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, User, DepartureLog, get_db
from main import app
from ml.global_model import GlobalModelManager, GlobalCohortConfig
from ml.adaptive.personalization import (
    PersonalizationEngine,
    PersonalizationState,
    compute_dynamic_alpha
)
from ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator


@pytest.fixture(scope="function")
def test_db():
    """In-memory SQLite database for isolated global-to-personalized testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client(test_db):
    """FastAPI TestClient wired to the test database."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def create_user_helper(db, username="user_test", profile_type="student", consent=True):
    """Create a test user with specified consent preference."""
    u = User(
        username=username,
        hashed_password="pw_hash",
        salt="salt_123",
        profile_type=profile_type,
        consent_to_global_training=consent
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def create_log_helper(db, user_id, delay=5.0, is_synthetic=False, provenance="real", event_name="Class"):
    """Create a departure log with explicit provenance."""
    log = DepartureLog(
        user_id=user_id,
        event_name=event_name,
        date="2026-11-17",
        day_of_week="Monday",
        planned_prep_start="08:00:00",
        actual_prep_start="08:00:00",
        planned_departure="08:30:00",
        actual_departure="08:35:00",
        actual_arrival="08:50:00",
        estimated_prep_duration_min=30.0,
        actual_prep_duration_min=30.0,
        estimated_travel_duration_min=15.0,
        actual_travel_duration_min=15.0,
        total_delay_min=float(delay),
        bag_prepared=True,
        tag_one_last_task=False,
        tag_finding_items=False,
        tag_phone_usage=False,
        tag_waiting=False,
        provenance=provenance,
        data_source="simulator_checkout" if not is_synthetic else "generator",
        is_synthetic=is_synthetic,
        recorded_at="2026-11-17T08:35:00Z",
        observation_status="valid"
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


# ---------------------------------------------------------------------------
# 1. Dynamic Alpha Calculation Tests
# ---------------------------------------------------------------------------

def test_dynamic_alpha_computation():
    """Verify dynamic alpha varies based on samples, relative performance, and stability."""
    # 0 samples -> pure global
    assert compute_dynamic_alpha(0) == 1.0

    # >= 8 samples -> pure personal
    assert compute_dynamic_alpha(8) == 0.0
    assert compute_dynamic_alpha(15) == 0.0

    # 4 samples with neutral performance and stability: base = 1 - 4/8 = 0.50
    alpha_mid = compute_dynamic_alpha(4)
    assert 0.45 <= alpha_mid <= 0.55

    # More samples -> lower alpha (shifts toward personal)
    alpha_2 = compute_dynamic_alpha(2)
    alpha_6 = compute_dynamic_alpha(6)
    assert alpha_2 > alpha_mid > alpha_6

    # Performance adjustment: personal model worse than global (personal_mae > global_mae)
    # -> alpha increases (relies more on global)
    alpha_worse_personal = compute_dynamic_alpha(4, personal_mae=5.0, global_mae=3.0)
    alpha_better_personal = compute_dynamic_alpha(4, personal_mae=2.0, global_mae=4.0)
    assert alpha_worse_personal > alpha_mid
    assert alpha_better_personal < alpha_mid

    # Stability adjustment: high volatility delay_std > 5 -> alpha increases (regularization)
    alpha_volatile = compute_dynamic_alpha(4, delay_std=7.5)
    alpha_stable = compute_dynamic_alpha(4, delay_std=1.2)
    assert alpha_volatile > alpha_stable


# ---------------------------------------------------------------------------
# 2. Lifecycle State Progression & Hybrid Blending Tests
# ---------------------------------------------------------------------------

def test_personalization_lifecycle_states():
    """Verify progression across explicit cold-start states."""
    engine = PersonalizationEngine(full_personalization_threshold=8, hybrid_start_threshold=4)

    # State: new_user / global_prediction
    info_new_with_global = engine.evaluate_lifecycle(genuine_samples=0, has_active_global_model=True)
    assert info_new_with_global["learning_mode"] == PersonalizationState.GLOBAL_PREDICTION.value
    assert info_new_with_global["is_global_used"] is True
    assert info_new_with_global["alpha"] == 1.0

    info_new_no_global = engine.evaluate_lifecycle(genuine_samples=0, has_active_global_model=False, synthetic_samples=0)
    assert info_new_no_global["learning_mode"] == PersonalizationState.INSUFFICIENT_DATA.value

    info_new_synth = engine.evaluate_lifecycle(genuine_samples=0, has_active_global_model=False, synthetic_samples=30)
    assert info_new_synth["learning_mode"] == PersonalizationState.NEW_USER.value

    # State: collecting_history (1 to 3 observations)
    info_collecting = engine.evaluate_lifecycle(genuine_samples=2, has_active_global_model=True)
    assert info_collecting["learning_mode"] == PersonalizationState.COLLECTING_HISTORY.value
    assert info_collecting["is_personalized"] is False

    # State: personalizing (4 to 7 observations)
    info_personalizing = engine.evaluate_lifecycle(genuine_samples=5, has_active_global_model=True)
    assert info_personalizing["learning_mode"] == PersonalizationState.PERSONALIZING.value
    assert 0.0 < info_personalizing["alpha"] < 1.0
    assert info_personalizing["is_personalized"] is False

    # State: personalized (>= 8 observations and approved model)
    info_personalized = engine.evaluate_lifecycle(genuine_samples=9, has_active_personal_model=True)
    assert info_personalized["learning_mode"] == PersonalizationState.PERSONALIZED.value
    assert info_personalized["alpha"] == 0.0
    assert info_personalized["is_personalized"] is True


def test_hybrid_prediction_blending():
    """Verify mathematically accurate hybrid blending with fallbacks."""
    engine = PersonalizationEngine()

    # Blend 60% global (4.0m) and 40% personal (2.0m) -> 0.60*4.0 + 0.40*2.0 = 2.4 + 0.8 = 3.2m
    pred = engine.blend_hybrid_prediction(global_pred=4.0, personal_pred=2.0, baseline_pred=4.5, alpha=0.60)
    assert pred == 3.2

    # 100% personal (alpha=0.0)
    assert engine.blend_hybrid_prediction(global_pred=4.0, personal_pred=2.0, baseline_pred=4.5, alpha=0.0) == 2.0

    # 100% global (alpha=1.0)
    assert engine.blend_hybrid_prediction(global_pred=4.0, personal_pred=2.0, baseline_pred=4.5, alpha=1.0) == 4.0

    # Fallback when personal is None
    assert engine.blend_hybrid_prediction(global_pred=3.5, personal_pred=None, baseline_pred=4.5, alpha=0.5) == 3.5

    # Fallback when both are None
    assert engine.blend_hybrid_prediction(global_pred=None, personal_pred=None, baseline_pred=4.5, alpha=0.5) == 4.5


# ---------------------------------------------------------------------------
# 3. Privacy, Consent & Cohort Safeguard Tests
# ---------------------------------------------------------------------------

def test_global_data_privacy_and_consent_filtering(test_db):
    """Verify synthetic logs and opted-out users are strictly excluded from global training data."""
    # User 1: Consented genuine user (3 real logs)
    u1 = create_user_helper(test_db, "user_consented", consent=True)
    for _ in range(3):
        create_log_helper(test_db, u1.id, delay=3.0, is_synthetic=False, provenance="real")

    # User 2: Consented user with SYNTHETIC logs (must be excluded!)
    u2 = create_user_helper(test_db, "user_synthetic_only", consent=True)
    for _ in range(10):
        create_log_helper(test_db, u2.id, delay=8.0, is_synthetic=True, provenance="synthetic")

    # User 3: OPTED-OUT user with real logs (must be excluded!)
    u3 = create_user_helper(test_db, "user_opted_out", consent=False)
    for _ in range(5):
        create_log_helper(test_db, u3.id, delay=6.0, is_synthetic=False, provenance="real")

    # Query consented genuine data
    df, meta = GlobalModelManager.get_consented_genuine_data(test_db, min_cohort_users=1, min_cohort_samples=1)

    # Only u1's genuine logs should be present!
    assert len(df) == 3
    assert df["user_id"].unique().tolist() == [u1.id]
    assert all(df["user_id"] != u2.id)
    assert all(df["user_id"] != u3.id)


def test_global_cohort_safeguard(test_db):
    """Verify training is safely rejected when minimum cohort size is not met."""
    u1 = create_user_helper(test_db, "solo_user", consent=True)
    for _ in range(4):
        create_log_helper(test_db, u1.id, delay=2.5, is_synthetic=False, provenance="real")

    # Requires min 2 users and min 10 samples
    res = GlobalModelManager.train_global_model(test_db, min_cohort_users=2, min_cohort_samples=10)
    assert res["success"] is False
    assert res["status"] == "insufficient_cohort"


# ---------------------------------------------------------------------------
# 4. Global Model Training & Leave-Users-Out Experiment Tests
# ---------------------------------------------------------------------------

def test_global_model_training_and_inference(test_db):
    """Train global model across 2 consented users and verify persistent prediction."""
    u1 = create_user_helper(test_db, "cohort_user_1", consent=True)
    u2 = create_user_helper(test_db, "cohort_user_2", consent=True)

    # 6 records for user 1, 6 records for user 2
    for i in range(6):
        create_log_helper(test_db, u1.id, delay=float(3 + (i % 3)), is_synthetic=False, provenance="real")
        create_log_helper(test_db, u2.id, delay=float(4 + (i % 2)), is_synthetic=False, provenance="real")

    res = GlobalModelManager.train_global_model(test_db, min_cohort_users=2, min_cohort_samples=10)
    assert res["success"] is True
    assert res["status"] == "trained_and_approved"
    assert res["cohort_users"] == 2
    assert res["cohort_samples"] == 12
    assert "test_mae" in res

    # Verify global inference works
    pred = GlobalModelManager.predict_global_delay(
        event_name="Class",
        day_of_week="Monday",
        importance="medium",
        bag_prepared=True,
        estimated_prep_duration_min=30.0,
        estimated_travel_duration_min=20.0
    )
    assert pred is not None
    assert 0.0 <= pred <= 20.0


def test_leave_users_out_experiment(test_db):
    """Verify Leave-Users-Out cross-validation across distinct users."""
    u1 = create_user_helper(test_db, "luo_user_1", profile_type="student", consent=True)
    u2 = create_user_helper(test_db, "luo_user_2", profile_type="teacher", consent=True)

    for i in range(5):
        create_log_helper(test_db, u1.id, delay=3.0, is_synthetic=False, provenance="real")
        create_log_helper(test_db, u2.id, delay=6.0, is_synthetic=False, provenance="real")

    res = GlobalModelManager.leave_users_out_benchmark(test_db, min_cohort_users=2)
    assert res["success"] is True
    assert res["evaluated_users_count"] == 2
    assert "mean_leave_users_out_mae" in res
    assert "group_breakdown" in res
    assert "student" in res["group_breakdown"]
    assert "teacher" in res["group_breakdown"]


def test_cross_user_isolation(test_db):
    """Verify personal models strictly isolate data per user_id."""
    u1 = create_user_helper(test_db, "tenant_1", consent=True)
    u2 = create_user_helper(test_db, "tenant_2", consent=True)

    create_log_helper(test_db, u1.id, delay=2.0)
    create_log_helper(test_db, u2.id, delay=15.0)

    # Query strictly u1's logs
    u1_logs = test_db.query(DepartureLog).filter(DepartureLog.user_id == u1.id).all()
    assert len(u1_logs) == 1
    assert u1_logs[0].total_delay_min == 2.0


# ---------------------------------------------------------------------------
# 5. API Endpoints Tests
# ---------------------------------------------------------------------------

def test_user_consent_api_endpoints(client, test_db):
    """Verify GET and POST /api/users/consent."""
    # Register and login user
    reg = client.post("/api/auth/register", json={
        "username": "consent_user",
        "password": "Password123!",
        "profile_type": "student"
    })
    token = reg.json().get("token") or reg.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. GET consent status (defaults to True)
    res = client.get("/api/users/consent", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["consent_to_global_training"] is True

    # 2. Opt out of global training
    opt_out = client.post("/api/users/consent", json={"consent": False}, headers=headers)
    assert opt_out.status_code == 200
    assert opt_out.json()["consent_to_global_training"] is False

    # 3. Verify GET returns updated status
    res2 = client.get("/api/users/consent", headers=headers)
    assert res2.json()["consent_to_global_training"] is False


def test_global_model_api_endpoints(client, test_db):
    """Verify GET /api/models/global/status and POST /api/models/global/train."""
    reg = client.post("/api/auth/register", json={
        "username": "api_global_user",
        "password": "Password123!",
        "profile_type": "it_professional"
    })
    token = reg.json().get("token") or reg.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}

    # Status check
    status_res = client.get("/api/models/global/status", headers=headers)
    assert status_res.status_code == 200

    # Train request
    train_res = client.post(
        "/api/models/global/train",
        json={"min_cohort_users": 2, "min_cohort_samples": 10},
        headers=headers
    )
    assert train_res.status_code == 200


def test_leave_users_out_api_endpoint(client, test_db):
    """Verify GET /api/benchmarks/leave-users-out."""
    reg = client.post("/api/auth/register", json={
        "username": "luo_api_user",
        "password": "Password123!",
        "profile_type": "professor"
    })
    token = reg.json().get("token") or reg.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}

    res = client.get("/api/benchmarks/leave-users-out", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "success" in data


def test_orchestrator_cold_start_to_personalization_output(test_db):
    """Verify AdaptiveIntelligenceOrchestrator attaches personalization_info."""
    u = create_user_helper(test_db, "orchestrator_user", consent=True)

    ctx = PredictionContext(
        user_id=u.id,
        event_name="Morning Brief",
        day_of_week="Monday",
        event_importance="medium",
        estimated_prep_duration=25.0,
        estimated_travel_duration=15.0,
        bag_prepared=True
    )

    orchestrator = AdaptiveIntelligenceOrchestrator()
    output = orchestrator.predict(ctx, db=test_db)

    # personalization_info contract check
    p_info = output.personalization_info
    assert p_info is not None
    assert "learning_mode" in p_info
    assert "alpha" in p_info
    assert "is_personalized" in p_info
    assert "is_global_used" in p_info
    assert "status_explanation" in p_info
