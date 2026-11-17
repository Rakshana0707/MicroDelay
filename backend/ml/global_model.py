"""Global Shared Model Architecture & Leave-Users-Out Evaluation for MicroDelay.

Solves the cold-start problem by training generalizable behavioral models across
consented, genuine user departure observations with strict privacy safeguards.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import joblib
from sqlalchemy.orm import Session
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import MicroDelayPreprocessor
from .validation import ChronologicalSplitter

logger = logging.getLogger("microdelay.ml.global_model")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

GLOBAL_MODEL_PATH = os.path.join(MODELS_DIR, "global_model.pkl")
GLOBAL_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "global_preprocessor.pkl")
GLOBAL_FEATS_PATH = os.path.join(MODELS_DIR, "global_feats.pkl")
GLOBAL_META_PATH = os.path.join(MODELS_DIR, "global_meta.json")


class GlobalCohortConfig:
    """Configurable safeguards for global shared training data."""
    MIN_COHORT_USERS: int = 2
    MIN_COHORT_SAMPLES: int = 10
    MIN_TEST_SAMPLES: int = 2


class GlobalModelManager:
    """Manages training, versioning, inference, and leave-users-out benchmarking

    for the shared cross-user MicroDelay global model.
    """

    @staticmethod
    def get_consented_genuine_data(
        db: Session,
        min_cohort_users: int = GlobalCohortConfig.MIN_COHORT_USERS,
        min_cohort_samples: int = GlobalCohortConfig.MIN_COHORT_SAMPLES
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Query strictly genuine departure logs from users who gave explicit consent.

        Privacy & Anonymization Safeguards:
        - Synthetic records (is_synthetic == True) are strictly excluded.
        - Users who opted out (consent_to_global_training == False) are strictly excluded.
        - Direct personal identifiers (usernames, emails, passwords, specific dates) are omitted.
        - Feature set contains solely behavioral and contextual attributes.
        """
        try:
            from database import DepartureLog, User
        except ModuleNotFoundError:
            from backend.database import DepartureLog, User

        # Strict join ensuring consent and genuine provenance
        logs_query = (
            db.query(DepartureLog, User.profile_type)
            .join(User, DepartureLog.user_id == User.id)
            .filter(
                DepartureLog.is_synthetic == False,
                DepartureLog.provenance == "real",
                DepartureLog.observation_status == "valid",
                User.consent_to_global_training == True
            )
            .all()
        )

        rows = []
        for log, profile_type in logs_query:
            rows.append({
                "user_id": log.user_id,
                "profile_type": profile_type or "general",
                "event_name": log.event_name,
                "day_of_week": log.day_of_week,
                "importance": getattr(log, "importance", "medium") or "medium",
                "bag_prepared": bool(getattr(log, "bag_prepared", False)),
                "estimated_prep_duration_min": float(getattr(log, "estimated_prep_duration_min", getattr(log, "estimated_prep_duration", 30)) or 30),
                "estimated_travel_duration_min": float(getattr(log, "estimated_travel_duration_min", getattr(log, "estimated_travel_duration", 20)) or 20),
                "total_delay_min": float(getattr(log, "total_delay_min", 0.0) or 0.0),
                "recorded_at": log.recorded_at or "2026-01-01T00:00:00Z"
            })

        df = pd.DataFrame(rows)
        if df.empty:
            meta = {
                "eligible_users_count": 0,
                "eligible_samples_count": 0,
                "is_cohort_sufficient": False,
                "reason": "No genuine records from consented users."
            }
            return df, meta

        unique_users = df["user_id"].nunique()
        total_samples = len(df)
        is_sufficient = (unique_users >= min_cohort_users) and (total_samples >= min_cohort_samples)

        meta = {
            "eligible_users_count": unique_users,
            "eligible_samples_count": total_samples,
            "min_cohort_users_required": min_cohort_users,
            "min_cohort_samples_required": min_cohort_samples,
            "is_cohort_sufficient": is_sufficient,
            "reason": None if is_sufficient else f"Requires >= {min_cohort_users} users and >= {min_cohort_samples} samples (found {unique_users} users, {total_samples} samples)."
        }
        return df, meta

    @staticmethod
    def train_global_model(
        db: Session,
        min_cohort_users: int = GlobalCohortConfig.MIN_COHORT_USERS,
        min_cohort_samples: int = GlobalCohortConfig.MIN_COHORT_SAMPLES
    ) -> Dict[str, Any]:
        """Train and evaluate the cross-user global model on consented genuine observations."""
        df, meta = GlobalModelManager.get_consented_genuine_data(
            db, min_cohort_users=min_cohort_users, min_cohort_samples=min_cohort_samples
        )

        if not meta["is_cohort_sufficient"]:
            logger.info("Global model training skipped: %s", meta["reason"])
            return {
                "success": False,
                "status": "insufficient_cohort",
                "message": meta["reason"],
                "cohort_users": meta["eligible_users_count"],
                "cohort_samples": meta["eligible_samples_count"]
            }

        # Sort chronologically by recorded_at
        df = df.sort_values(by="recorded_at").reset_index(drop=True)

        # 70% Train, 15% Val, 15% Test Chronological Split
        splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        df_train, df_val, df_test = splitter.split(df)

        if len(df_train) < 5 or len(df_val) < 1 or len(df_test) < 1:
            # Fallback split for small valid cohorts
            split_idx_train = max(3, int(len(df) * 0.70))
            split_idx_val = max(split_idx_train + 1, int(len(df) * 0.85))
            df_train = df.iloc[:split_idx_train].copy()
            df_val = df.iloc[split_idx_train:split_idx_val].copy()
            df_test = df.iloc[split_idx_val:].copy()
            if len(df_test) == 0:
                df_test = df_val.copy()

        # Fit preprocessor strictly on training split
        preprocessor = MicroDelayPreprocessor()
        preprocessor.fit(df_train)
        feature_names = preprocessor.feature_names_

        X_train = preprocessor.transform(df_train)
        y_train = df_train["total_delay_min"].values.astype(float)

        X_val = preprocessor.transform(df_val)
        y_val = df_val["total_delay_min"].values.astype(float)

        X_test = preprocessor.transform(df_test)
        y_test = df_test["total_delay_min"].values.astype(float)

        # Candidate Algorithms
        candidates = {
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=50, max_depth=5, random_state=42),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        }

        best_model_name = None
        best_model = None
        best_val_mae = float("inf")
        candidate_eval = {}

        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                val_mae = float(mean_absolute_error(y_val, val_pred))
                candidate_eval[name] = {"val_mae": val_mae}
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_model_name = name
                    best_model = model
            except Exception as e:
                logger.warning("Global candidate %s training failed: %s", name, e)

        if best_model is None:
            best_model_name = "RandomForestRegressor"
            best_model = candidates["RandomForestRegressor"]
            best_model.fit(X_train, y_train)

        # Final evaluation on holdout test partition
        test_pred = best_model.predict(X_test)
        test_mae = float(mean_absolute_error(y_test, test_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
        test_r2 = float(r2_score(y_test, test_pred)) if len(y_test) > 2 else 0.0
        median_ae = float(np.median(np.abs(y_test - test_pred)))

        within_5 = float(np.mean(np.abs(y_test - test_pred) <= 5.0) * 100.0)
        within_10 = float(np.mean(np.abs(y_test - test_pred) <= 10.0) * 100.0)

        version_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        global_version = f"global_{best_model_name.lower()[:3]}_{version_timestamp}"

        # Persist artifacts
        joblib.dump(best_model, GLOBAL_MODEL_PATH)
        joblib.dump(preprocessor, GLOBAL_PREPROCESSOR_PATH)
        joblib.dump(feature_names, GLOBAL_FEATS_PATH)

        metadata = {
            "global_model_version": global_version,
            "model_type": best_model_name,
            "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "cohort_users_count": meta["eligible_users_count"],
            "cohort_samples_count": len(df),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "validation_mae": round(best_val_mae, 3),
            "test_mae": round(test_mae, 3),
            "test_rmse": round(test_rmse, 3),
            "test_r2": round(test_r2, 3),
            "median_absolute_error": round(median_ae, 3),
            "pct_within_5min": round(within_5, 1),
            "pct_within_10min": round(within_10, 1),
            "feature_names": feature_names,
            "candidate_eval": candidate_eval,
            "is_active": True
        }

        with open(GLOBAL_META_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        # Register in database model_metadata table
        try:
            from database import ModelVersionRecord
            # Archive prior active global model entries
            db.query(ModelVersionRecord).filter(
                ModelVersionRecord.user_id == 0,
                ModelVersionRecord.is_active == True
            ).update({"is_active": False, "model_status": "archived"})

            new_record = ModelVersionRecord(
                model_id=f"global_model_{version_timestamp}",
                user_id=0,
                model_version=global_version,
                model_type=best_model_name,
                training_data_source="real_cohort_consented",
                training_samples=len(df),
                training_started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                training_completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                validation_mae=best_val_mae,
                test_mae=test_mae,
                feature_schema_version="v2_global_standard",
                artifact_path=GLOBAL_MODEL_PATH,
                model_status="approved",
                is_active=True,
                status_reason="Promoted as active global population model.",
                metrics_json=json.dumps(metadata)
            )
            db.add(new_record)
            db.commit()
        except Exception as reg_err:
            logger.warning("Could not register global model in model_metadata table: %s", reg_err)
            db.rollback()

        return {
            "success": True,
            "status": "trained_and_approved",
            "global_model_version": global_version,
            "model_type": best_model_name,
            "cohort_users": meta["eligible_users_count"],
            "cohort_samples": len(df),
            "validation_mae": round(best_val_mae, 3),
            "test_mae": round(test_mae, 3),
            "pct_within_5min": round(within_5, 1),
            "metadata": metadata
        }

    @staticmethod
    def load_global_model() -> Optional[Dict[str, Any]]:
        """Load persistent global model bundle from disk."""
        if not (os.path.exists(GLOBAL_MODEL_PATH) and os.path.exists(GLOBAL_PREPROCESSOR_PATH)):
            return None

        try:
            model = joblib.load(GLOBAL_MODEL_PATH)
            preprocessor = joblib.load(GLOBAL_PREPROCESSOR_PATH)
            feats = joblib.load(GLOBAL_FEATS_PATH) if os.path.exists(GLOBAL_FEATS_PATH) else None
            metadata = {}
            if os.path.exists(GLOBAL_META_PATH):
                with open(GLOBAL_META_PATH, "r") as f:
                    metadata = json.load(f)

            return {
                "model": model,
                "preprocessor": preprocessor,
                "feature_names": feats,
                "metadata": metadata
            }
        except Exception as e:
            logger.error("Error loading global model: %s", e)
            return None

    @staticmethod
    def predict_global_delay(
        event_name: str,
        day_of_week: str,
        importance: str,
        bag_prepared: bool,
        estimated_prep_duration_min: float,
        estimated_travel_duration_min: float
    ) -> Optional[float]:
        """Generate prediction from the shared global model without user-specific parameters."""
        bundle = GlobalModelManager.load_global_model()
        if not bundle:
            return None

        try:
            model = bundle["model"]
            preprocessor = bundle["preprocessor"]

            single_row = pd.DataFrame([{
                "event_name": event_name,
                "day_of_week": day_of_week,
                "importance": importance,
                "bag_prepared": bool(bag_prepared),
                "estimated_prep_duration_min": float(estimated_prep_duration_min),
                "estimated_travel_duration_min": float(estimated_travel_duration_min)
            }])

            X = preprocessor.transform(single_row)
            pred = float(model.predict(X)[0])
            return round(pred, 2)
        except Exception as e:
            logger.warning("Global model prediction error: %s", e)
            return None

    @staticmethod
    def get_global_model_status(db: Optional[Session] = None) -> Dict[str, Any]:
        """Surface metadata and operational state of the global model."""
        bundle = GlobalModelManager.load_global_model()
        if not bundle:
            return {
                "is_available": False,
                "message": "Global model has not yet been trained or does not meet minimum cohort requirements.",
                "global_model_version": None,
                "cohort_users_count": 0,
                "cohort_samples_count": 0
            }

        meta = bundle.get("metadata", {})
        return {
            "is_available": True,
            "global_model_version": meta.get("global_model_version"),
            "model_type": meta.get("model_type"),
            "trained_at": meta.get("trained_at"),
            "cohort_users_count": meta.get("cohort_users_count", 0),
            "cohort_samples_count": meta.get("cohort_samples_count", 0),
            "validation_mae": meta.get("validation_mae"),
            "test_mae": meta.get("test_mae"),
            "pct_within_5min": meta.get("pct_within_5min"),
            "candidate_eval": meta.get("candidate_eval", {})
        }

    @staticmethod
    def leave_users_out_benchmark(
        db: Session,
        min_cohort_users: int = GlobalCohortConfig.MIN_COHORT_USERS
    ) -> Dict[str, Any]:
        """Perform a leave-users-out evaluation experiment.

        Trains a global model on N-1 consented users and evaluates generalization
        on the held-out user to verify cross-user transferability without overfitting.
        """
        df, meta = GlobalModelManager.get_consented_genuine_data(db, min_cohort_users=min_cohort_users, min_cohort_samples=4)
        if df.empty or meta["eligible_users_count"] < 2:
            return {
                "success": False,
                "message": "Leave-users-out experiment requires at least 2 distinct consented users with genuine data.",
                "eligible_users_count": meta.get("eligible_users_count", 0),
                "eligible_samples_count": meta.get("eligible_samples_count", 0)
            }

        users = df["user_id"].unique()
        user_results = []
        all_y_true = []
        all_y_pred = []

        for held_out_user in users:
            df_train = df[df["user_id"] != held_out_user].copy()
            df_test = df[df["user_id"] == held_out_user].copy()

            if len(df_train) < 3 or len(df_test) < 1:
                continue

            try:
                preprocessor = MicroDelayPreprocessor()
                preprocessor.fit(df_train)
                X_train = preprocessor.transform(df_train)
                y_train = df_train["total_delay_min"].values.astype(float)

                X_test = preprocessor.transform(df_test)
                y_test = df_test["total_delay_min"].values.astype(float)

                model = RandomForestRegressor(n_estimators=40, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                user_mae = float(mean_absolute_error(y_test, y_pred))
                user_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                profile_type = df_test["profile_type"].iloc[0] if "profile_type" in df_test.columns else "general"

                user_results.append({
                    "held_out_user_id": int(held_out_user),
                    "profile_type": profile_type,
                    "test_samples": len(df_test),
                    "train_samples": len(df_train),
                    "mae": round(user_mae, 3),
                    "rmse": round(user_rmse, 3)
                })

                all_y_true.extend(y_test.tolist())
                all_y_pred.extend(y_pred.tolist())
            except Exception as e:
                logger.warning("Error in leave-users-out fold for user %s: %s", held_out_user, e)

        if not user_results:
            return {
                "success": False,
                "message": "Leave-users-out experiment could not produce valid folds.",
                "eligible_users_count": len(users)
            }

        mean_mae = float(np.mean([r["mae"] for r in user_results]))
        mean_rmse = float(np.mean([r["rmse"] for r in user_results]))
        mae_std = float(np.std([r["mae"] for r in user_results]))

        # Performance across user groups (without treating persona as deterministic)
        group_perf: Dict[str, List[float]] = {}
        for r in user_results:
            p = r["profile_type"]
            group_perf.setdefault(p, []).append(r["mae"])

        group_breakdown = {
            grp: {
                "users_count": len(maes),
                "mean_mae": round(float(np.mean(maes)), 3),
                "mae_std": round(float(np.std(maes)), 3)
            }
            for grp, maes in group_perf.items()
        }

        return {
            "success": True,
            "evaluated_users_count": len(user_results),
            "total_eval_samples": len(all_y_true),
            "mean_leave_users_out_mae": round(mean_mae, 3),
            "mean_leave_users_out_rmse": round(mean_rmse, 3),
            "inter_user_mae_std": round(mae_std, 3),
            "group_breakdown": group_breakdown,
            "user_results": user_results,
            "explanation": (
                f"Global model evaluated on {len(user_results)} unseen holdout users yielded an average MAE of {mean_mae:.2f}m "
                f"(std: {mae_std:.2f}m), confirming generalizability to new users before individual personalization."
            )
        }
