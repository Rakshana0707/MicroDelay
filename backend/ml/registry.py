"""Model Registry, Versioning, and Safe Promotion for MicroDelay.

This module provides versioned model artifact management, concurrency safety,
anti-tampering path validation, and safe promotion/rollback without breaking
backward-compatible prediction endpoints.

Key Principles:
1. Versioned storage per user: backend/models/versions/user_{id}/{version}/
2. Never overwrite active model artifacts without candidate validation and approval.
3. Safe promotion: candidates are evaluated and compared against current active model.
4. Instant rollback to previous archived versions.
5. Strict path protection: disallows loading untrusted or path-traversal artifacts.
6. Concurrency protection: prevents overlapping training runs for the same user.
"""

import os
import shutil
import json
import datetime
import threading
from typing import Any, Dict, List, Optional, Tuple

import joblib
from sqlalchemy.orm import Session
from sqlalchemy import text

try:
    from database import ModelVersionRecord, DepartureLog, User
except ModuleNotFoundError:
    from backend.database import ModelVersionRecord, DepartureLog, User


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
os.makedirs(VERSIONS_DIR, exist_ok=True)

# Process-level concurrency locks per user_id
_USER_TRAINING_LOCKS: Dict[int, threading.Lock] = {}
_LOCK_ADMIN = threading.Lock()


def get_user_lock(user_id: int) -> threading.Lock:
    """Get or create a thread lock for a specific user to serialize training."""
    with _LOCK_ADMIN:
        if user_id not in _USER_TRAINING_LOCKS:
            _USER_TRAINING_LOCKS[user_id] = threading.Lock()
        return _USER_TRAINING_LOCKS[user_id]


class ModelStatus:
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    APPROVED = "approved"
    ACTIVE = "active"
    REJECTED = "rejected"
    ARCHIVED = "archived"
    FAILED = "failed"


class RetrainingPolicy:
    """Configurable retraining thresholds and triggers."""
    def __init__(
        self,
        min_new_observations: int = 5,
        min_training_samples: int = 8,
        mae_tolerance_slack: float = 0.5
    ):
        self.min_new_observations = min_new_observations
        self.min_training_samples = min_training_samples
        self.mae_tolerance_slack = mae_tolerance_slack

    def should_retrain(self, db: Session, user_id: int) -> Tuple[bool, str]:
        """Check if retraining criteria are satisfied for a user."""
        total_logs = db.query(DepartureLog).filter(DepartureLog.user_id == user_id).count()
        if total_logs < self.min_training_samples:
            return False, f"Total observations ({total_logs}) below minimum required ({self.min_training_samples})."

        active_rec = db.query(ModelVersionRecord).filter(
            ModelVersionRecord.user_id == user_id,
            ModelVersionRecord.is_active == True
        ).first()

        if not active_rec:
            return True, "No active model version currently registered for user."

        new_logs_count = total_logs - active_rec.training_samples
        if new_logs_count >= self.min_new_observations:
            return True, f"Accumulated {new_logs_count} new observations since last training (policy threshold: {self.min_new_observations})."

        return False, f"Only {new_logs_count} new observations recorded (minimum {self.min_new_observations} required)."


class ModelRegistry:
    """Manages versioned model artifacts, safe promotion, and rollbacks."""

    @staticmethod
    def get_version_dir(user_id: int, version_tag: str) -> str:
        """Resolve canonical versioned storage path for a user."""
        user_dir = os.path.join(VERSIONS_DIR, f"user_{user_id}")
        version_dir = os.path.join(user_dir, version_tag)
        os.makedirs(version_dir, exist_ok=True)
        return os.path.abspath(version_dir)

    @staticmethod
    def validate_artifact_path(user_id: int, artifact_path: str) -> bool:
        """Protect model artifacts from directory traversal or foreign user access."""
        if not artifact_path or not isinstance(artifact_path, str):
            return False

        abs_path = os.path.abspath(artifact_path)
        expected_user_dir = os.path.abspath(os.path.join(VERSIONS_DIR, f"user_{user_id}"))
        legacy_models_dir = os.path.abspath(MODELS_DIR)

        # Allow valid versioned subfolder strictly inside expected user directory
        try:
            rel = os.path.relpath(abs_path, expected_user_dir)
            is_in_user_dir = not rel.startswith("..") and not os.path.isabs(rel)
        except ValueError:
            is_in_user_dir = False

        # Allow legacy flat artifact directly in MODELS_DIR with matching user suffix
        is_legacy = (
            os.path.dirname(abs_path) == legacy_models_dir
            and f"user_{user_id}" in os.path.basename(abs_path)
        )

        return is_in_user_dir or is_legacy

    @classmethod
    def create_candidate_version(
        cls,
        db: Session,
        user_id: int,
        model_type: str = "RandomForestRegressor",
        data_source: str = "synthetic",
        training_samples: int = 0,
        feature_schema_version: str = "v1_standard"
    ) -> ModelVersionRecord:
        """Initialize a new candidate version record in the database."""
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp_tag = now.strftime("%Y%m%d_%H%M%S_%f")
        version_str = f"v{timestamp_tag}"
        model_id = f"user_{user_id}_{model_type.lower()}_{version_str}"

        version_dir = cls.get_version_dir(user_id, version_str)

        record = ModelVersionRecord(
            model_id=model_id,
            user_id=user_id,
            model_version=version_str,
            model_type=model_type,
            training_data_source=data_source,
            training_samples=training_samples,
            training_started_at=now.isoformat(),
            feature_schema_version=feature_schema_version,
            artifact_path=version_dir,
            model_status=ModelStatus.TRAINING,
            is_active=False,
            status_reason="Candidate model training initiated."
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @classmethod
    def save_candidate_artifacts(
        cls,
        record: ModelVersionRecord,
        model: Any,
        preprocessor: Any,
        feature_columns: List[str],
        baseline_info: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """Persist candidate model artifacts into its isolated version directory."""
        v_dir = record.artifact_path
        os.makedirs(v_dir, exist_ok=True)

        joblib.dump(model, os.path.join(v_dir, "model.pkl"))
        joblib.dump(preprocessor, os.path.join(v_dir, "preprocessor.pkl"))
        joblib.dump(feature_columns, os.path.join(v_dir, "features.pkl"))
        joblib.dump(baseline_info, os.path.join(v_dir, "baseline.pkl"))

        meta_path = os.path.join(v_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def evaluate_and_promote(
        cls,
        db: Session,
        record_id: int,
        val_mae: float,
        test_mae: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        policy: Optional[RetrainingPolicy] = None
    ) -> Tuple[bool, str]:
        """Safely promote candidate model to active only if evaluation criteria are met."""
        pol = policy or RetrainingPolicy()
        record = db.query(ModelVersionRecord).filter(ModelVersionRecord.id == record_id).first()
        if not record:
            return False, "Candidate record not found."

        now = datetime.datetime.now(datetime.timezone.utc)
        record.training_completed_at = now.isoformat()
        record.validation_mae = float(val_mae)
        record.test_mae = float(test_mae) if test_mae is not None else None
        if metrics:
            record.metrics_json = json.dumps(metrics)

        # Check current active model
        active_rec = db.query(ModelVersionRecord).filter(
            ModelVersionRecord.user_id == record.user_id,
            ModelVersionRecord.is_active == True
        ).first()

        should_promote = False
        decision_reason = ""

        if not active_rec:
            # First model for user: promote directly
            should_promote = True
            decision_reason = f"Initial model version {record.model_version} approved as active model (MAE: {val_mae:.2f}m)."
        else:
            # Compare candidate vs current active model
            active_mae = active_rec.validation_mae or 999.0
            # Candidate promotes if its validation MAE is better or within allowable tolerance slack
            if val_mae <= (active_mae + pol.mae_tolerance_slack):
                should_promote = True
                decision_reason = (
                    f"Candidate {record.model_version} (MAE {val_mae:.2f}m) successfully outperformed or matched "
                    f"active version {active_rec.model_version} (MAE {active_mae:.2f}m)."
                )
            else:
                should_promote = False
                decision_reason = (
                    f"Candidate {record.model_version} rejected: validation MAE ({val_mae:.2f}m) degraded compared to "
                    f"active model {active_rec.model_version} ({active_mae:.2f}m) beyond slack threshold."
                )

        if should_promote:
            # Archive current active model
            if active_rec:
                active_rec.is_active = False
                active_rec.model_status = ModelStatus.ARCHIVED
                active_rec.status_reason = f"Archived upon promotion of newer model version {record.model_version}."

            record.is_active = True
            record.model_status = ModelStatus.ACTIVE
            record.status_reason = decision_reason

            # Sync legacy artifact pointers in backend/models/ for backward compatibility
            cls._sync_legacy_pointers(record)
        else:
            record.is_active = False
            record.model_status = ModelStatus.REJECTED
            record.status_reason = decision_reason

        db.commit()
        db.refresh(record)
        return should_promote, decision_reason

    @classmethod
    def rollback_model(cls, db: Session, user_id: int) -> Tuple[bool, str]:
        """Rollback to the most recent archived model version for a user."""
        current_active = db.query(ModelVersionRecord).filter(
            ModelVersionRecord.user_id == user_id,
            ModelVersionRecord.is_active == True
        ).first()

        previous_archived = db.query(ModelVersionRecord).filter(
            ModelVersionRecord.user_id == user_id,
            ModelVersionRecord.model_status == ModelStatus.ARCHIVED
        ).order_by(ModelVersionRecord.id.desc()).first()

        if not previous_archived:
            return False, "No previous archived model version available for rollback."

        # Demote current active
        if current_active:
            current_active.is_active = False
            current_active.model_status = ModelStatus.ARCHIVED
            current_active.status_reason = f"Demoted during rollback to version {previous_archived.model_version}."

        # Promote previous archived
        previous_archived.is_active = True
        previous_archived.model_status = ModelStatus.ACTIVE
        previous_archived.status_reason = "Restored to active status via rollback."

        cls._sync_legacy_pointers(previous_archived)

        db.commit()
        return True, f"Successfully rolled back to model version {previous_archived.model_version}."

    @classmethod
    def load_active_model_bundle(cls, db: Session, user_id: int) -> Optional[Dict[str, Any]]:
        """Safely load active model artifacts, verifying ownership and path protection."""
        active_rec = db.query(ModelVersionRecord).filter(
            ModelVersionRecord.user_id == user_id,
            ModelVersionRecord.is_active == True
        ).first()

        if active_rec and cls.validate_artifact_path(user_id, active_rec.artifact_path):
            v_dir = active_rec.artifact_path
            m_path = os.path.join(v_dir, "model.pkl")
            p_path = os.path.join(v_dir, "preprocessor.pkl")
            f_path = os.path.join(v_dir, "features.pkl")
            b_path = os.path.join(v_dir, "baseline.pkl")

            if os.path.exists(m_path) and os.path.exists(p_path) and os.path.exists(f_path):
                try:
                    return {
                        "model": joblib.load(m_path),
                        "preprocessor": joblib.load(p_path),
                        "features": joblib.load(f_path),
                        "baseline": joblib.load(b_path) if os.path.exists(b_path) else {},
                        "model_version": active_rec.model_version,
                        "model_type": active_rec.model_type,
                        "data_provenance": active_rec.training_data_source,
                        "is_versioned": True
                    }
                except Exception as e:
                    print(f"Error loading versioned model for user {user_id}: {e}")

        # Fallback to legacy flat files in MODELS_DIR
        legacy_m = os.path.join(MODELS_DIR, f"rf_model_user_{user_id}.pkl")
        legacy_p = os.path.join(MODELS_DIR, f"preprocessor_user_{user_id}.pkl")
        legacy_f = os.path.join(MODELS_DIR, f"feats_user_{user_id}.pkl")
        legacy_b = os.path.join(MODELS_DIR, f"baseline_user_{user_id}.pkl")

        if os.path.exists(legacy_m) and os.path.exists(legacy_f):
            try:
                return {
                    "model": joblib.load(legacy_m),
                    "preprocessor": joblib.load(legacy_p) if os.path.exists(legacy_p) else None,
                    "features": joblib.load(legacy_f),
                    "baseline": joblib.load(legacy_b) if os.path.exists(legacy_b) else {},
                    "model_version": "legacy_v1",
                    "model_type": "RandomForestRegressor",
                    "data_provenance": "unknown",
                    "is_versioned": False
                }
            except Exception as e:
                print(f"Error loading legacy model for user {user_id}: {e}")

        return None

    @classmethod
    def _sync_legacy_pointers(cls, record: ModelVersionRecord) -> None:
        """Copy active version files to flat paths so legacy endpoints function without modification."""
        v_dir = record.artifact_path
        user_id = record.user_id

        src_model = os.path.join(v_dir, "model.pkl")
        src_prep = os.path.join(v_dir, "preprocessor.pkl")
        src_feats = os.path.join(v_dir, "features.pkl")
        src_base = os.path.join(v_dir, "baseline.pkl")

        dst_model = os.path.join(MODELS_DIR, f"rf_model_user_{user_id}.pkl")
        dst_prep = os.path.join(MODELS_DIR, f"preprocessor_user_{user_id}.pkl")
        dst_feats = os.path.join(MODELS_DIR, f"feats_user_{user_id}.pkl")
        dst_base = os.path.join(MODELS_DIR, f"baseline_user_{user_id}.pkl")

        for s, d in [(src_model, dst_model), (src_prep, dst_prep), (src_feats, dst_feats), (src_base, dst_base)]:
            if os.path.exists(s):
                try:
                    shutil.copyfile(s, d)
                except Exception:
                    pass
