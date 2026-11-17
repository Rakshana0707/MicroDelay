import os
import datetime
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from database import DepartureLog, Event
try:
    from database import ModelVersionRecord
except ImportError:
    from backend.database import ModelVersionRecord

from ml.preprocessing import validate_observations_dataframe, MicroDelayPreprocessor, validate_single_observation
try:
    from ml.validation import ChronologicalEvaluator, ChronologicalSplitter, EvaluationReport, MIN_CHRONOLOGICAL_SAMPLES
    from ml.registry import ModelRegistry, ModelStatus, RetrainingPolicy, get_user_lock
except ModuleNotFoundError:
    from backend.ml.validation import ChronologicalEvaluator, ChronologicalSplitter, EvaluationReport, MIN_CHRONOLOGICAL_SAMPLES
    from backend.ml.registry import ModelRegistry, ModelStatus, RetrainingPolicy, get_user_lock
import joblib

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
EVENT_NAMES = [
    # General templates across all profiles
    "ML Lab", "Seminar", "Math Lecture", "Library Study", "Hardware Lab", "Theory Lecture", "Weekend Kickoff",
    "Morning Assembly", "Homeroom Class", "Staff Meeting", "Science Lab", "Grading Session",
    "Research Lecture", "Thesis Review", "Faculty Council", "Office Hours", "Symposium",
    "Daily Standup", "Code Review", "Sprint Planning", "Design Sync", "Deploy & Retro"
]
IMPORTANCE_MAPPING = {"low": 0, "medium": 1, "high": 2}

# Resolve model artifacts relative to backend module directory to avoid path collisions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model_paths(user_id: int) -> dict:
    """Generate dynamic model file paths per user_id using absolute paths."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return {
        "model": os.path.join(MODELS_DIR, f"rf_model_user_{user_id}.pkl"),
        "preprocessor": os.path.join(MODELS_DIR, f"preprocessor_user_{user_id}.pkl"),
        "features": os.path.join(MODELS_DIR, f"feats_user_{user_id}.pkl"),
        "baseline": os.path.join(MODELS_DIR, f"baseline_user_{user_id}.pkl"),
        "metadata": os.path.join(MODELS_DIR, f"metadata_user_{user_id}.pkl")
    }


def get_user_data_counts(db: Session, user_id: int) -> dict:
    """Get observation counts broken down by provenance for a user."""
    total = db.query(DepartureLog).filter(DepartureLog.user_id == user_id).count()
    real = db.query(DepartureLog).filter(DepartureLog.user_id == user_id, DepartureLog.is_synthetic == False).count()
    synthetic = db.query(DepartureLog).filter(DepartureLog.user_id == user_id, DepartureLog.is_synthetic == True).count()
    unknown = db.query(DepartureLog).filter(
        DepartureLog.user_id == user_id,
        DepartureLog.provenance == "unknown"
    ).count()
    return {
        "total_samples": total,
        "real_samples": real,
        "synthetic_samples": synthetic,
        "unknown_samples": unknown,
        "has_real_data": real > 0,
        "is_sufficient_real": real >= 8
    }


def get_logs_df(db: Session, user_id: int, training_mode: str = "real_only") -> pd.DataFrame:
    """Fetch departure logs from database for a specific user according to training_mode."""
    query = db.query(DepartureLog).filter(DepartureLog.user_id == user_id)

    if training_mode == "real_only":
        query = query.filter(DepartureLog.is_synthetic == False)
    elif training_mode == "synthetic_only":
        query = query.filter(DepartureLog.is_synthetic == True)
    elif training_mode in ["synthetic_assisted", "all"]:
        # Explicit synthetic-assisted development/experiment mode
        pass
    else:
        # Default fallback: real only
        query = query.filter(DepartureLog.is_synthetic == False)

    logs = query.all()
    if not logs:
        return pd.DataFrame()

    data = []
    for log in logs:
        # Find importance from the event template, fallback to medium
        importance = "medium"
        if log.event_id:
            event = db.query(Event).filter(Event.id == log.event_id).first()
            if event:
                importance = event.importance

        data.append({
            "id": log.id,
            "event_name": log.event_name,
            "day_of_week": log.day_of_week,
            "date": getattr(log, "date", ""),
            "planned_departure": getattr(log, "planned_departure", ""),
            "actual_departure": getattr(log, "actual_departure", ""),
            "importance": importance,
            "bag_prepared": 1 if log.bag_prepared else 0,
            "estimated_prep_duration_min": log.estimated_prep_duration_min,
            "estimated_travel_duration_min": log.estimated_travel_duration_min,
            "actual_prep_duration_min": getattr(log, "actual_prep_duration_min", log.estimated_prep_duration_min),
            "actual_travel_duration_min": getattr(log, "actual_travel_duration_min", log.estimated_travel_duration_min),
            "tag_phone_usage": 1 if getattr(log, "tag_phone_usage", False) else 0,
            "tag_finding_items": 1 if getattr(log, "tag_finding_items", False) else 0,
            "tag_one_last_task": 1 if getattr(log, "tag_one_last_task", False) else 0,
            "tag_waiting": 1 if getattr(log, "tag_waiting", False) else 0,
            "total_delay_min": log.total_delay_min,
            "is_synthetic": 1 if log.is_synthetic else 0,
            "provenance": getattr(log, "provenance", "unknown"),
            "data_source": getattr(log, "data_source", "unknown"),
            "recorded_at": getattr(log, "recorded_at", "")
        })

    return pd.DataFrame(data)


def preprocess_features(df: pd.DataFrame, fit_columns=None):
    """Encode categorical features consistently."""
    df["importance_code"] = df["importance"].map(IMPORTANCE_MAPPING).fillna(1)

    for day in DAYS_OF_WEEK:
        df[f"day_{day}"] = (df["day_of_week"] == day).astype(int)

    for event in EVENT_NAMES:
        clean_evt = event.lower().replace(" ", "_").replace("&", "and")
        df[f"event_{clean_evt}"] = (df["event_name"] == event).astype(int)

    X = df.drop(columns=[
        "id", "date", "planned_departure", "actual_departure",
        "actual_prep_duration_min", "actual_travel_duration_min",
        "tag_phone_usage", "tag_finding_items", "tag_one_last_task", "tag_waiting",
        "event_name", "day_of_week", "importance", "total_delay_min",
        "is_synthetic", "provenance", "data_source", "recorded_at"
    ], errors="ignore")

    if fit_columns is not None:
        for col in fit_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[fit_columns]

    return X


def train_and_evaluate(db: Session, user_id: int, training_mode: str = "auto") -> dict:
    """Train per-user statistical baseline and Random Forest models with explicit provenance tracking.
    
    Modes:
      - 'auto': trains real-only if >= 8 real logs exist; falls back to synthetic-only demo training if only synthetic logs exist.
      - 'real_only': trains strictly on genuine user checkout observations.
      - 'synthetic_only': trains strictly on synthetic demonstration records.
      - 'synthetic_assisted': trains on both for development experiments with explicit disclosure.
    """
    counts = get_user_data_counts(db, user_id)
    paths = get_model_paths(user_id)

    # Resolve training mode
    selected_mode = training_mode
    if training_mode == "auto":
        if counts["real_samples"] >= 8:
            selected_mode = "real_only"
        elif counts["synthetic_samples"] >= 8:
            selected_mode = "synthetic_only"
        else:
            selected_mode = "real_only"

    df = get_logs_df(db, user_id, training_mode=selected_mode)

    # Validate observations through centralized validation pipeline
    clean_df, val_report = validate_observations_dataframe(df, drop_invalid=True)

    if clean_df.empty or len(clean_df) < MIN_CHRONOLOGICAL_SAMPLES:
        reason = (
            f"Insufficient valid data for mode '{selected_mode}' after validation "
            f"(minimum {MIN_CHRONOLOGICAL_SAMPLES} required, found {len(clean_df)} valid). "
            f"Chronological evaluation requires separate train, validation, and untouched test partitions."
        )
        return {
            "success": False,
            "message": reason,
            "training_mode": selected_mode,
            "data_provenance": "none",
            "is_personalized_real": False,
            "real_samples": counts["real_samples"],
            "synthetic_samples": counts["synthetic_samples"],
            "total_samples": counts["total_samples"],
            "training_samples_used": len(clean_df),
            "evaluation_status": "insufficient_data",
            "validation_report": val_report.summary()
        }

    try:
        from ml.registry import ModelRegistry, ModelStatus, get_user_lock
    except ModuleNotFoundError:
        from backend.ml.registry import ModelRegistry, ModelStatus, get_user_lock

    user_lock = get_user_lock(user_id)
    if not user_lock.acquire(blocking=False):
        return {
            "success": False,
            "message": f"A training job is already in progress for user {user_id}. Concurrent runs are serialized.",
            "training_mode": selected_mode
        }

    candidate_rec = None
    try:
        is_personalized_real = (selected_mode == "real_only")
        data_provenance_label = "real" if is_personalized_real else ("synthetic" if selected_mode == "synthetic_only" else "synthetic_assisted")

        # Create candidate version entry in model_metadata table
        candidate_rec = ModelRegistry.create_candidate_version(
            db=db,
            user_id=user_id,
            model_type="RandomForestRegressor",
            data_source=data_provenance_label,
            training_samples=len(clean_df),
            feature_schema_version="v1_standard"
        )

        # Conduct chronological evaluation with 70/15/15 partitions and expanding-window CV
        evaluator = ChronologicalEvaluator(min_samples=MIN_CHRONOLOGICAL_SAMPLES)
        eval_report, final_rf, final_preprocessor = evaluator.evaluate(
            clean_df,
            provenance=data_provenance_label,
            model_builder=lambda: RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        )

        event_means = clean_df.groupby("event_name")["total_delay_min"].mean().to_dict()
        global_mean = float(clean_df["total_delay_min"].mean())
        feature_columns = final_preprocessor.feature_names_
        trained_at_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Detailed evaluation, residual analysis, and report generation
        try:
            from ml.reporting import (
                calculate_standardized_metrics,
                perform_residual_analysis,
                segment_performance,
                determine_model_status,
                EvaluationReporter
            )
        except ModuleNotFoundError:
            from backend.ml.reporting import (
                calculate_standardized_metrics,
                perform_residual_analysis,
                segment_performance,
                determine_model_status,
                EvaluationReporter
            )

        # Generate test predictions for holdout partition
        splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        _, _, test_df = splitter.split(clean_df)
        X_test = final_preprocessor.transform(test_df)
        y_test = test_df["total_delay_min"].values.astype(float)
        y_test_pred = final_rf.predict(X_test)

        std_metrics = calculate_standardized_metrics(y_test, y_test_pred, baseline_mae=eval_report.baseline_test_mae)
        residuals = perform_residual_analysis(y_test, y_test_pred, df=test_df)
        
        # Segment performance across full dataset
        full_preds = final_rf.predict(final_preprocessor.transform(clean_df))
        segmentation = segment_performance(clean_df, clean_df["total_delay_min"].values, full_preds)

        eval_status, status_reason = determine_model_status(
            train_count=len(clean_df),
            is_validated=eval_report.is_validated,
            baseline_mae=eval_report.baseline_test_mae,
            model_mae=std_metrics["mae"],
            min_samples=MIN_CHRONOLOGICAL_SAMPLES,
            tolerance_5m_pct=std_metrics["tolerance_5m_pct"],
            large_error_pct=residuals.get("large_errors", {}).get("percentage", 0.0)
        )

        reporter = EvaluationReporter(user_id=user_id, model_version=candidate_rec.model_version, provenance=data_provenance_label)
        json_report_path, md_report_path = reporter.generate_report(
            metrics=std_metrics,
            residual_analysis=residuals,
            segmentation=segmentation,
            model_name="RandomForestRegressor",
            hyperparameters={"n_estimators": 50, "max_depth": 5, "random_state": 42},
            training_metadata={
                "trained_at": trained_at_str,
                "training_samples_used": len(clean_df),
                "model_version": candidate_rec.model_version,
                "partitions": {
                    "train": eval_report.train_metadata,
                    "validation": eval_report.val_metadata,
                    "test": eval_report.test_metadata
                }
            },
            evaluation_status=eval_status,
            status_reason=status_reason
        )

        model_metadata = {
            "user_id": user_id,
            "training_mode": selected_mode,
            "data_provenance": data_provenance_label,
            "is_personalized_real": is_personalized_real,
            "real_samples": counts["real_samples"],
            "synthetic_samples": counts["synthetic_samples"],
            "total_samples": counts["total_samples"],
            "training_samples_used": len(clean_df),
            "trained_at": trained_at_str,
            "model_version": candidate_rec.model_version,
            "model_status": "active_real" if is_personalized_real else "demonstration_synthetic",
            "evaluation_status": eval_status,
            "status_reason": status_reason,
            "is_validated": eval_report.is_validated,
            "baseline_mae": eval_report.baseline_test_mae,
            "ml_mae": eval_report.test_mae,
            "error_reduction_pct": eval_report.error_reduction_pct,
            "standardized_metrics": std_metrics,
            "residual_analysis": residuals,
            "evaluation_report": eval_report.to_dict(),
            "validation_report": val_report.summary(),
            "report_artifacts": {
                "json": json_report_path,
                "markdown": md_report_path
            }
        }

        # Train and calibrate Personalized Lateness Classification Suite
        try:
            try:
                from ml.classifier import LatenessClassifierSuite, LatenessClassifierManager
            except ModuleNotFoundError:
                from backend.ml.classifier import LatenessClassifierSuite, LatenessClassifierManager

            clf_suite = LatenessClassifierSuite()
            clf_res = clf_suite.train_and_benchmark(clean_df)
            if clf_res.get("success"):
                clf_metadata = {
                    "user_id": user_id,
                    "selected_best_model": clf_res.get("selected_best_model"),
                    "test_metrics": clf_res.get("test_metrics"),
                    "models": clf_res.get("models"),
                    "model_version": candidate_rec.model_version,
                    "data_provenance": data_provenance_label,
                    "is_calibrated": any(m.get("is_calibrated", False) for m in clf_res.get("models", []))
                }
                LatenessClassifierManager.save_classifier_bundle(
                    user_id=user_id,
                    model=clf_res["fitted_model"],
                    preprocessor=clf_res.get("preprocessor"),
                    feature_columns=clf_res.get("fitted_columns", []),
                    metadata=clf_metadata
                )
                model_metadata["classifier_metrics"] = clf_res.get("test_metrics")
        except Exception as clf_err:
            print(f"Lateness classifier training skipped for user {user_id}: {clf_err}")

        # 1. Save candidate artifacts in versioned storage
        baseline_info = {
            "event_means": event_means,
            "global_mean": global_mean,
            "metadata": model_metadata
        }
        ModelRegistry.save_candidate_artifacts(
            record=candidate_rec,
            model=final_rf,
            preprocessor=final_preprocessor,
            feature_columns=feature_columns,
            baseline_info=baseline_info,
            metadata=model_metadata
        )

        # 2. Safe Promotion Evaluation
        is_promoted, promotion_reason = ModelRegistry.evaluate_and_promote(
            db=db,
            record_id=candidate_rec.id,
            val_mae=std_metrics["mae"],
            test_mae=std_metrics["mae"],
            metrics=std_metrics
        )

        # Extract Top Features
        importances = final_rf.feature_importances_
        feat_imp = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)[:5]
        feat_imp_list = [{"feature": f, "importance": round(float(imp), 4)} for f, imp in feat_imp]

        train_size = eval_report.train_metadata["count"] if eval_report.train_metadata else 0
        test_size = eval_report.test_metadata["count"] if eval_report.test_metadata else 0
        val_size = eval_report.val_metadata["count"] if eval_report.val_metadata else 0

        return {
            "success": True,
            "training_mode": selected_mode,
            "data_provenance": data_provenance_label,
            "is_personalized_real": is_personalized_real,
            "model_version": candidate_rec.model_version,
            "is_promoted_active": is_promoted,
            "promotion_reason": promotion_reason,
            "real_samples": counts["real_samples"],
            "synthetic_samples": counts["synthetic_samples"],
            "total_samples": counts["total_samples"],
            "training_samples_used": len(clean_df),
            "trained_at": trained_at_str,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "baseline_mae": eval_report.baseline_test_mae,
            "ml_mae": eval_report.test_mae,
            "error_reduction_pct": eval_report.error_reduction_pct,
            "evaluation_report": eval_report.to_dict(),
            "evaluation_status": eval_status,
            "status_reason": status_reason,
            "standardized_metrics": std_metrics,
            "report_artifacts": {
                "json": json_report_path,
                "markdown": md_report_path
            },
            "top_features": feat_imp_list
        }
    except Exception as e:
        if candidate_rec:
            candidate_rec.model_status = ModelStatus.FAILED
            candidate_rec.status_reason = f"Model training failed: {str(e)}"
            db.commit()
        print(f"Error training models for user {user_id}: {e}")
        return {
            "success": False,
            "message": f"Model training failed: {e}",
            "training_mode": selected_mode,
            "is_personalized_real": False,
            "real_samples": counts["real_samples"],
            "synthetic_samples": counts["synthetic_samples"]
        }
    finally:
        user_lock.release()


def get_model_status(user_id: int, db: Optional[Session] = None) -> dict:
    """Check if model is trained for the specific user and inspect metadata."""
    paths = get_model_paths(user_id)
    is_trained = os.path.exists(paths["model"]) and os.path.exists(paths["features"])
    metadata = {}
    if is_trained and os.path.exists(paths["metadata"]):
        try:
            metadata = joblib.load(paths["metadata"])
        except Exception:
            pass

    # Check active version record if db session is provided
    active_version = metadata.get("model_version", "legacy_v1")
    validation_mae = metadata.get("validation_mae", None)
    if db is not None:
        try:
            active_rec = db.query(ModelVersionRecord).filter(
                ModelVersionRecord.user_id == user_id,
                ModelVersionRecord.is_active == True
            ).first()
            if active_rec:
                is_trained = True
                active_version = active_rec.model_version
                validation_mae = active_rec.validation_mae
        except Exception:
            pass

    return {
        "trained": is_trained,
        "model_version": active_version,
        "validation_mae": validation_mae,
        "training_mode": metadata.get("training_mode", "unknown"),
        "data_provenance": metadata.get("data_provenance", "unknown"),
        "is_personalized_real": metadata.get("is_personalized_real", False),
        "real_samples": metadata.get("real_samples", 0),
        "synthetic_samples": metadata.get("synthetic_samples", 0),
        "training_samples_used": metadata.get("training_samples_used", 0),
        "trained_at": metadata.get("trained_at", "")
    }


def rollback_user_model(db: Session, user_id: int) -> dict:
    """Rollback user model to the most recent archived active version."""
    success, message = ModelRegistry.rollback_model(db, user_id)
    return {
        "success": success,
        "message": message
    }


def predict_micro_delay(
    user_id: int,
    event_name: str,
    day_of_week: str,
    importance: str,
    bag_prepared: bool,
    estimated_prep_duration_min: float,
    estimated_travel_duration_min: float,
    db: Optional[Session] = None
) -> dict:
    """Predict expected delay using the user's active versioned or legacy ML model."""
    rf = None
    feature_columns = None
    baseline = {}
    preprocessor = None
    model_version = "legacy_v1"
    data_provenance = "unknown"
    training_mode = "unknown"
    is_personalized_real = False

    # 1. Try loading versioned active bundle if db session is provided
    if db is not None:
        try:
            bundle = ModelRegistry.load_active_model_bundle(db, user_id)
            if bundle:
                rf = bundle.get("model")
                preprocessor = bundle.get("preprocessor")
                feature_columns = bundle.get("features")
                baseline = bundle.get("baseline", {})
                model_version = bundle.get("model_version", "legacy_v1")
                data_provenance = bundle.get("data_provenance", "unknown")
        except Exception as e:
            print(f"Failed to load versioned model bundle for user {user_id}: {e}")

    # 2. Fallback to legacy file paths or shared global model if needed
    if rf is None or feature_columns is None:
        paths = get_model_paths(user_id)
        if not (os.path.exists(paths["model"]) and os.path.exists(paths["features"])):
            if db is not None:
                try:
                    try:
                        from ml.global_model import GlobalModelManager
                    except ModuleNotFoundError:
                        from backend.ml.global_model import GlobalModelManager
                    global_pred = GlobalModelManager.predict_global_delay(
                        event_name=event_name,
                        day_of_week=day_of_week,
                        importance=importance,
                        bag_prepared=bag_prepared,
                        estimated_prep_duration_min=estimated_prep_duration_min,
                        estimated_travel_duration_min=estimated_travel_duration_min
                    )
                    global_status = GlobalModelManager.get_global_model_status(db)
                    if global_pred is not None:
                        return {
                            "predicted_delay_min": round(global_pred, 1),
                            "baseline_predicted_delay_min": 4.5,
                            "model_used": "global_model",
                            "model_version": global_status.get("global_model_version", "global_v1"),
                            "data_provenance": "global_cohort",
                            "is_personalized_real": False,
                            "model_availability": "available"
                        }
                except Exception as g_err:
                    print(f"Global model fallback lookup failed: {g_err}")

            return {
                "predicted_delay_min": 4.5,
                "baseline_predicted_delay_min": 4.5,
                "model_used": "default_fallback",
                "model_version": "default_fallback",
                "data_provenance": "none",
                "is_personalized_real": False,
                "model_availability": "insufficient_data"
            }
        try:
            rf = joblib.load(paths["model"])
            feature_columns = joblib.load(paths["features"])
            if os.path.exists(paths["baseline"]):
                baseline = joblib.load(paths["baseline"])
            if os.path.exists(paths.get("preprocessor", "")):
                preprocessor = joblib.load(paths["preprocessor"])
        except Exception as e:
            print(f"Error loading legacy artifacts for user {user_id}: {e}")
            return {
                "predicted_delay_min": 4.5,
                "baseline_predicted_delay_min": 4.5,
                "model_used": "error_fallback",
                "model_version": "error_fallback",
                "data_provenance": "none",
                "is_personalized_real": False
            }

    try:
        metadata = baseline.get("metadata", {})
        if data_provenance == "unknown":
            data_provenance = metadata.get("data_provenance", "unknown")
        training_mode = metadata.get("training_mode", "unknown")
        is_personalized_real = metadata.get("is_personalized_real", (data_provenance == "real"))
        if model_version == "legacy_v1" and "model_version" in metadata:
            model_version = metadata["model_version"]

        event_means = baseline.get("event_means", {})
        global_mean = baseline.get("global_mean", 4.5)

        # Use fitted preprocessor pipeline if available, fallback to legacy encoding
        if preprocessor is not None and hasattr(preprocessor, "transform_single_inference"):
            X_pred = preprocessor.transform_single_inference({
                "event_name": event_name,
                "day_of_week": day_of_week,
                "importance": importance,
                "bag_prepared": 1 if bag_prepared else 0,
                "estimated_prep_duration_min": estimated_prep_duration_min,
                "estimated_travel_duration_min": estimated_travel_duration_min
            })
        else:
            single_row = pd.DataFrame([{
                "event_name": event_name,
                "day_of_week": day_of_week,
                "importance": importance,
                "bag_prepared": 1 if bag_prepared else 0,
                "estimated_prep_duration_min": estimated_prep_duration_min,
                "estimated_travel_duration_min": estimated_travel_duration_min
            }])
            X_pred = preprocess_features(single_row, fit_columns=feature_columns)

        predicted_ml = rf.predict(X_pred)[0]
        predicted_baseline = event_means.get(event_name, global_mean)

        late_prob = None
        risk_cat = None
        try:
            try:
                from ml.adaptive.risk_classifier import LatenessRiskClassifier
            except ModuleNotFoundError:
                from backend.ml.adaptive.risk_classifier import LatenessRiskClassifier
            classifier = LatenessRiskClassifier()
            late_prob = classifier.compute_late_probability(float(predicted_ml))
            risk_cat = classifier.classify_risk(float(predicted_ml), late_prob).value
        except Exception:
            pass

        personalization_info = {}
        try:
            try:
                from ml.adaptive.personalization import PersonalizationEngine
            except ModuleNotFoundError:
                from backend.ml.adaptive.personalization import PersonalizationEngine
            engine = PersonalizationEngine()
            real_cnt = 0
            synth_cnt = 0
            if db is not None:
                try:
                    from database import DepartureLog
                except ModuleNotFoundError:
                    from backend.database import DepartureLog
                real_cnt = db.query(DepartureLog).filter(DepartureLog.user_id == user_id, DepartureLog.is_synthetic == False).count()
                synth_cnt = db.query(DepartureLog).filter(DepartureLog.user_id == user_id, DepartureLog.is_synthetic == True).count()
            personalization_info = engine.evaluate_lifecycle(
                genuine_samples=real_cnt,
                synthetic_samples=synth_cnt,
                has_active_personal_model=is_personalized_real,
                has_active_global_model=True
            )
        except Exception:
            pass

        return {
            "predicted_delay_min": round(float(predicted_ml), 1),
            "baseline_predicted_delay_min": round(float(predicted_baseline), 1),
            "model_used": "random_forest",
            "model_version": model_version,
            "data_provenance": data_provenance,
            "training_mode": training_mode,
            "is_personalized_real": is_personalized_real,
            "late_probability": round(late_prob, 2) if late_prob is not None else None,
            "risk_category": risk_cat,
            "model_availability": "available",
            "personalization_info": personalization_info
        }
    except Exception as e:
        print(f"Error predicting delay for user {user_id}: {e}")
        return {
            "predicted_delay_min": 4.5,
            "baseline_predicted_delay_min": 4.5,
            "model_used": "error_fallback",
            "model_version": model_version,
            "data_provenance": "none",
            "is_personalized_real": False,
            "late_probability": 0.5,
            "risk_category": "moderate",
            "model_availability": "failed"
        }


def benchmark_user_models(db: Session, user_id: int, training_mode: str = "auto") -> dict:
    """Benchmark multiple regression algorithms on historical records for a user."""
    counts = get_user_data_counts(db, user_id)

    selected_mode = training_mode
    if training_mode == "auto":
        if counts["real_samples"] >= 8:
            selected_mode = "real_only"
        elif counts["synthetic_samples"] >= 8:
            selected_mode = "synthetic_only"
        else:
            selected_mode = "real_only"

    df = get_logs_df(db, user_id, training_mode=selected_mode)
    clean_df, _ = validate_observations_dataframe(df, drop_invalid=True)

    data_provenance_label = "real" if (selected_mode == "real_only") else ("synthetic" if selected_mode == "synthetic_only" else "synthetic_assisted")

    try:
        from ml.benchmark import ModelBenchmarkSuite
    except ModuleNotFoundError:
        from backend.ml.benchmark import ModelBenchmarkSuite

    suite = ModelBenchmarkSuite()
    result = suite.run_benchmark(clean_df, provenance=data_provenance_label)
    return result.to_dict()


def predict_lateness_risk(
    user_id: int,
    event_name: str,
    day_of_week: str,
    importance: str,
    bag_prepared: bool,
    estimated_prep_duration_min: float,
    estimated_travel_duration_min: float
) -> dict:
    """Predict departure lateness risk probability and category using dedicated classifier."""
    try:
        from ml.classifier import LatenessClassifierManager
    except ModuleNotFoundError:
        from backend.ml.classifier import LatenessClassifierManager

    return LatenessClassifierManager.predict_lateness_risk(
        user_id=user_id,
        event_name=event_name,
        day_of_week=day_of_week,
        importance=importance,
        bag_prepared=bag_prepared,
        estimated_prep_duration_min=estimated_prep_duration_min,
        estimated_travel_duration_min=estimated_travel_duration_min
    )


def get_user_classifier_metrics(user_id: int) -> dict:
    """Fetch stored classification evaluation and calibration metrics for a user."""
    try:
        from ml.classifier import LatenessClassifierManager
    except ModuleNotFoundError:
        from backend.ml.classifier import LatenessClassifierManager

    bundle = LatenessClassifierManager.load_classifier_bundle(user_id)
    if not bundle:
        return {
            "has_classifier": False,
            "message": "No personalized classifier trained yet."
        }
    meta = bundle.get("metadata", {})
    return {
        "has_classifier": True,
        "selected_best_model": meta.get("selected_best_model"),
        "test_metrics": meta.get("test_metrics"),
        "models": meta.get("models"),
        "model_version": meta.get("model_version"),
        "data_provenance": meta.get("data_provenance"),
        "is_calibrated": meta.get("is_calibrated", True)
    }


def train_global_model(db: Session, min_cohort_users: int = 2, min_cohort_samples: int = 10) -> dict:
    """Train global shared model across consented genuine records."""
    try:
        from ml.global_model import GlobalModelManager
    except ModuleNotFoundError:
        from backend.ml.global_model import GlobalModelManager
    return GlobalModelManager.train_global_model(db, min_cohort_users=min_cohort_users, min_cohort_samples=min_cohort_samples)


def get_global_model_status(db: Optional[Session] = None) -> dict:
    """Get status and metrics of the global model."""
    try:
        from ml.global_model import GlobalModelManager
    except ModuleNotFoundError:
        from backend.ml.global_model import GlobalModelManager
    return GlobalModelManager.get_global_model_status(db)


def get_leave_users_out_benchmark(db: Session) -> dict:
    """Run or fetch leave-users-out evaluation across consented users."""
    try:
        from ml.global_model import GlobalModelManager
    except ModuleNotFoundError:
        from backend.ml.global_model import GlobalModelManager
    return GlobalModelManager.leave_users_out_benchmark(db)


def get_user_consent(db: Session, user_id: int) -> dict:
    """Get user's consent status for global shared training."""
    try:
        from database import User, DepartureLog
    except ModuleNotFoundError:
        from backend.database import User, DepartureLog

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"error": "User not found"}

    eligible_count = db.query(DepartureLog).filter(
        DepartureLog.user_id == user_id,
        DepartureLog.is_synthetic == False,
        DepartureLog.provenance == "real",
        DepartureLog.observation_status == "valid"
    ).count()

    return {
        "user_id": user_id,
        "consent_to_global_training": bool(getattr(user, "consent_to_global_training", True)),
        "eligible_genuine_samples": eligible_count,
        "anonymization_status": "Strict feature-only anonymization (direct identifiers omitted)",
        "policy_summary": (
            "Consented genuine departure patterns help improve initial estimates for new users. "
            "Synthetic data and personal identifiers are never included in shared global models. "
            "You may opt out at any time."
        )
    }


def update_user_consent(db: Session, user_id: int, consent: bool) -> dict:
    """Update user's consent for global shared training."""
    try:
        from database import User
    except ModuleNotFoundError:
        from backend.database import User

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"error": "User not found"}

    user.consent_to_global_training = bool(consent)
    db.commit()
    return {
        "success": True,
        "user_id": user_id,
        "consent_to_global_training": user.consent_to_global_training,
        "message": "Global model training consent preferences updated successfully."
    }


def get_user_cluster_profile(db: Session, user_id: int) -> dict:
    """Fetch or generate behavioral habit cluster profile for user."""
    try:
        from ml.habit_discovery import HabitDiscoveryService
    except ModuleNotFoundError:
        from backend.ml.habit_discovery import HabitDiscoveryService

    return HabitDiscoveryService.get_or_create_user_profile(user_id, db)


def train_cluster_models(db: Session) -> dict:
    """Run tournament clustering across genuine observations and persist active clusters."""
    try:
        from ml.habit_discovery import HabitDiscoveryService
    except ModuleNotFoundError:
        from backend.ml.habit_discovery import HabitDiscoveryService

    return HabitDiscoveryService.train_and_persist_clusters(db)


def get_clustering_benchmark(db: Session) -> dict:
    """Retrieve comparative benchmark report across K-Means, Agglomerative, and DBSCAN."""
    try:
        from ml.habit_discovery import HabitDiscoveryService
    except ModuleNotFoundError:
        from backend.ml.habit_discovery import HabitDiscoveryService

    return HabitDiscoveryService.get_clustering_benchmark_report(db)



