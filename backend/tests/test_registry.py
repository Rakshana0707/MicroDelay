"""Tests for Model Registry, Versioning, and Safe Promotion (Prompt 1.9)."""

import os
import pytest
import threading
from sklearn.ensemble import RandomForestRegressor
from database import User, ModelVersionRecord
from ml.registry import ModelRegistry, ModelStatus, RetrainingPolicy, get_user_lock
from ml.preprocessing import MicroDelayPreprocessor
import models


def test_create_candidate_version_and_save_artifacts(db_session, test_user):
    """Verify candidate record creation and isolated directory persistence."""
    candidate = ModelRegistry.create_candidate_version(
        db=db_session,
        user_id=test_user.id,
        model_type="RandomForestRegressor",
        data_source="synthetic",
        training_samples=50
    )

    assert candidate.id is not None
    assert candidate.model_status == ModelStatus.TRAINING
    assert candidate.is_active is False
    assert f"user_{test_user.id}" in candidate.artifact_path
    assert os.path.exists(candidate.artifact_path)

    # Save mock artifacts
    dummy_model = RandomForestRegressor(n_estimators=5)
    dummy_prep = MicroDelayPreprocessor()
    dummy_feats = ["importance", "bag_prepared"]
    baseline_info = {"event_means": {"Gym": 2.0}, "global_mean": 3.0}
    metadata = {"user_id": test_user.id, "model_version": candidate.model_version}

    ModelRegistry.save_candidate_artifacts(
        record=candidate,
        model=dummy_model,
        preprocessor=dummy_prep,
        feature_columns=dummy_feats,
        baseline_info=baseline_info,
        metadata=metadata
    )

    assert os.path.exists(os.path.join(candidate.artifact_path, "model.pkl"))
    assert os.path.exists(os.path.join(candidate.artifact_path, "preprocessor.pkl"))
    assert os.path.exists(os.path.join(candidate.artifact_path, "features.pkl"))
    assert os.path.exists(os.path.join(candidate.artifact_path, "baseline.pkl"))
    assert os.path.exists(os.path.join(candidate.artifact_path, "metadata.json"))


def test_safe_promotion_workflow(db_session, test_user):
    """Verify safe promotion: initial promotes, better promotes, degraded is rejected."""
    # 1. Candidate 1: Initial model should promote directly
    c1 = ModelRegistry.create_candidate_version(db_session, test_user.id, training_samples=30)
    promoted_1, reason_1 = ModelRegistry.evaluate_and_promote(db_session, c1.id, val_mae=5.0)
    assert promoted_1 is True
    assert c1.is_active is True
    assert c1.model_status == ModelStatus.ACTIVE

    # 2. Candidate 2: Better MAE (3.5 < 5.0) should promote, archiving C1
    c2 = ModelRegistry.create_candidate_version(db_session, test_user.id, training_samples=40)
    promoted_2, reason_2 = ModelRegistry.evaluate_and_promote(db_session, c2.id, val_mae=3.5)
    assert promoted_2 is True
    assert c2.is_active is True
    assert c2.model_status == ModelStatus.ACTIVE
    
    db_session.refresh(c1)
    assert c1.is_active is False
    assert c1.model_status == ModelStatus.ARCHIVED

    # 3. Candidate 3: Degraded MAE (6.0 > 3.5 + 0.5) should be REJECTED, C2 remains active
    c3 = ModelRegistry.create_candidate_version(db_session, test_user.id, training_samples=50)
    promoted_3, reason_3 = ModelRegistry.evaluate_and_promote(db_session, c3.id, val_mae=6.0)
    assert promoted_3 is False
    assert c3.is_active is False
    assert c3.model_status == ModelStatus.REJECTED

    db_session.refresh(c2)
    assert c2.is_active is True
    assert c2.model_status == ModelStatus.ACTIVE


def test_rollback_mechanism(db_session, test_user):
    """Verify rollback restores previous active model and demotes current active."""
    # Setup: C1 active -> C2 active (C1 archived)
    c1 = ModelRegistry.create_candidate_version(db_session, test_user.id, training_samples=30)
    ModelRegistry.evaluate_and_promote(db_session, c1.id, val_mae=4.0)

    c2 = ModelRegistry.create_candidate_version(db_session, test_user.id, training_samples=40)
    ModelRegistry.evaluate_and_promote(db_session, c2.id, val_mae=3.0)

    # Rollback to C1
    success, msg = ModelRegistry.rollback_model(db_session, test_user.id)
    assert success is True
    assert c1.model_version in msg

    db_session.refresh(c1)
    db_session.refresh(c2)
    assert c1.is_active is True
    assert c1.model_status == ModelStatus.ACTIVE
    assert c2.is_active is False
    assert c2.model_status == ModelStatus.ARCHIVED

    # Second rollback rolls back to C2
    success2, msg2 = ModelRegistry.rollback_model(db_session, test_user.id)
    assert success2 is True
    db_session.refresh(c2)
    assert c2.is_active is True


def test_path_traversal_protection(test_user):
    """Verify validate_artifact_path blocks directory traversal and foreign user paths."""
    user_id = test_user.id
    valid_path = ModelRegistry.get_version_dir(user_id, "v20260101_000000")
    assert ModelRegistry.validate_artifact_path(user_id, valid_path) is True

    # Path to different user's directory
    foreign_path = ModelRegistry.get_version_dir(user_id + 999, "v20260101_000000")
    assert ModelRegistry.validate_artifact_path(user_id, foreign_path) is False

    # Directory traversal attempt
    traversal_path = os.path.join(valid_path, "..", "..", "..", "system32")
    assert ModelRegistry.validate_artifact_path(user_id, traversal_path) is False

    # Empty or None path
    assert ModelRegistry.validate_artifact_path(user_id, "") is False
    assert ModelRegistry.validate_artifact_path(user_id, None) is False


def test_user_training_concurrency_lock():
    """Verify get_user_lock provides per-user serialization."""
    lock_user_1a = get_user_lock(101)
    lock_user_1b = get_user_lock(101)
    lock_user_2 = get_user_lock(102)

    # Same lock instance for same user
    assert lock_user_1a is lock_user_1b
    # Different lock instance for different user
    assert lock_user_1a is not lock_user_2

    # Lock acquisition works non-blockingly
    acquired = lock_user_1a.acquire(blocking=False)
    assert acquired is True
    assert lock_user_1b.acquire(blocking=False) is False
    lock_user_1a.release()


def test_api_model_versions_and_rollback(client, auth_headers, db_session, test_user):
    """Test /api/models/versions and /api/models/rollback endpoints."""
    # Create two versions via registry
    c1 = ModelRegistry.create_candidate_version(db_session, test_user.id, training_samples=30)
    ModelRegistry.evaluate_and_promote(db_session, c1.id, val_mae=4.0)

    c2 = ModelRegistry.create_candidate_version(db_session, test_user.id, training_samples=40)
    ModelRegistry.evaluate_and_promote(db_session, c2.id, val_mae=3.2)

    # 1. Test GET /api/models/versions
    resp = client.get("/api/models/versions", headers=auth_headers)
    assert resp.status_code == 200
    versions = resp.json()
    assert len(versions) >= 2
    assert versions[0]["model_version"] == c2.model_version
    assert versions[0]["is_active"] is True
    assert versions[0]["validation_mae"] == 3.2

    # 2. Test POST /api/models/rollback
    resp_rb = client.post("/api/models/rollback", headers=auth_headers)
    assert resp_rb.status_code == 200
    assert resp_rb.json()["success"] is True

    # Check updated active version
    resp_after = client.get("/api/models/versions", headers=auth_headers)
    versions_after = resp_after.json()
    v1_rec = [v for v in versions_after if v["model_version"] == c1.model_version][0]
    assert v1_rec["is_active"] is True
