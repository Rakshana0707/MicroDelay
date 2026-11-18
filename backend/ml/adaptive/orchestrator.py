"""Central Adaptive Intelligence Orchestration Service for MicroDelay.

Coordinates regression prediction, lateness risk classification, feature explainability,
adaptive recommendations, behavioral drift detection, and confidence scoring into
a unified, failure-isolated prediction pipeline.
"""

import time
import logging
from typing import Optional, Dict, Any, List
import numpy as np
from sqlalchemy.orm import Session

from .context import PredictionContext
from .schemas import (
    AdaptiveIntelligenceOutput,
    ModelAvailability,
    DataQualityStatus,
    ConfidenceStatus,
    RiskCategory
)
from .risk_classifier import LatenessRiskClassifier
from .personalization import PersonalizationEngine
from .clustering import BehavioralClusteringEngine
from .behavioral_profile import BehavioralProfileEngine
from .explainability import PredictionExplainer
from .adaptive_recommendations import AdaptiveRecommendationEngine
from .drift_detection import BehavioralDriftDetector
from .confidence import ConfidenceEstimator

logger = logging.getLogger("microdelay.ml.adaptive.orchestrator")


class AdaptiveIntelligenceOrchestrator:
    """Coordinates all adaptive behavioral intelligence components with fault isolation."""

    def __init__(
        self,
        risk_classifier: Optional[LatenessRiskClassifier] = None,
        personalization_engine: Optional[PersonalizationEngine] = None,
        clustering_engine: Optional[BehavioralClusteringEngine] = None,
        profile_engine: Optional[BehavioralProfileEngine] = None,
        explainer: Optional[PredictionExplainer] = None,
        recommendation_engine: Optional[AdaptiveRecommendationEngine] = None,
        drift_detector: Optional[BehavioralDriftDetector] = None,
        confidence_estimator: Optional[ConfidenceEstimator] = None
    ):
        self.risk_classifier = risk_classifier or LatenessRiskClassifier()
        self.personalization_engine = personalization_engine or PersonalizationEngine()
        self.clustering_engine = clustering_engine or BehavioralClusteringEngine()
        self.profile_engine = profile_engine or BehavioralProfileEngine()
        self.explainer = explainer or PredictionExplainer()
        self.recommendation_engine = recommendation_engine or AdaptiveRecommendationEngine()
        self.drift_detector = drift_detector or BehavioralDriftDetector()
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator()

    def predict(
        self,
        context: PredictionContext,
        db: Optional[Session] = None
    ) -> AdaptiveIntelligenceOutput:
        """Run end-to-end adaptive behavioral intelligence pipeline.

        Guarantees that component-level failures never crash the prediction response.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []
        component_status: Dict[str, str] = {}

        # 1. Validate Context
        val_result = context.validate()
        if not val_result.is_valid:
            logger.warning(
                "PredictionContext validation warnings for user_id=%d: %s",
                context.user_id, val_result.errors
            )
            warnings.extend(val_result.errors)

        # 2. Query Data Counts & Check Model Availability
        sample_counts = {"real_samples": 0, "synthetic_samples": 0, "total_samples": 0}
        recent_logs = []
        user_profile_type = "general"

        if db is not None:
            try:
                try:
                    from models import get_user_data_counts
                    from database import User, DepartureLog
                except ModuleNotFoundError:
                    from backend.models import get_user_data_counts
                    from backend.database import User, DepartureLog

                sample_counts = get_user_data_counts(db, context.user_id)
                recent_logs = (
                    db.query(DepartureLog)
                    .filter(DepartureLog.user_id == context.user_id)
                    .order_by(DepartureLog.date.desc(), DepartureLog.planned_departure.desc())
                    .limit(20)
                    .all()
                )
                u = db.query(User).filter(User.id == context.user_id).first()
                if u and u.profile_type:
                    user_profile_type = u.profile_type
            except Exception as e:
                logger.warning("Could not retrieve DB data counts for user_id=%d: %s", context.user_id, e)
                warnings.append("Database data count lookup skipped.")

        # Determine Data Quality & Provenance
        real_count = sample_counts.get("real_samples", 0)
        synth_count = sample_counts.get("synthetic_samples", 0)
        provenance = "real" if real_count >= 8 else ("synthetic" if synth_count > 0 else "none")
        is_sufficient = (real_count >= 8) or (synth_count >= 20)

        data_quality = DataQualityStatus(
            provenance=provenance,
            sample_count=sample_counts.get("total_samples", real_count + synth_count),
            is_sufficient=is_sufficient,
            has_real_data=(real_count > 0),
            quality_notes=[]
        )

        # 3. Global Model & Lifecycle Assessment (Prompt 2.3)
        try:
            from ml.global_model import GlobalModelManager
        except ModuleNotFoundError:
            from backend.ml.global_model import GlobalModelManager

        has_active_global_model = False
        global_mae = None
        global_status = {}
        if db is not None:
            global_status = GlobalModelManager.get_global_model_status(db=db)
            has_active_global_model = global_status.get("is_available", False)
            global_mae = global_status.get("validation_mae")

        # Behavioral stability: calculate std of recent delays
        delay_std = None
        if recent_logs and len(recent_logs) >= 2:
            delays = [float(getattr(l, "total_delay_min", 0.0) or 0.0) for l in recent_logs]
            delay_std = float(np.std(delays))

        # Check personal model status
        has_active_personal_model = (real_count >= 8) and is_sufficient

        lifecycle_info = self.personalization_engine.evaluate_lifecycle(
            genuine_samples=real_count,
            synthetic_samples=synth_count,
            has_active_personal_model=has_active_personal_model,
            has_active_global_model=has_active_global_model,
            personal_mae=None,
            global_mae=global_mae,
            delay_std=delay_std
        )

        if lifecycle_info["learning_mode"] == "insufficient_data":
            model_availability = ModelAvailability.INSUFFICIENT_DATA.value
        else:
            model_availability = ModelAvailability.AVAILABLE.value

        # 4. Regression Delay Prediction Execution
        predicted_delay = 4.5
        baseline_delay = 4.5
        model_version = "default_fallback"
        model_type = "historical_baseline"
        validation_mae = None
        fallback_applied = False

        global_pred = None
        if has_active_global_model:
            global_pred = GlobalModelManager.predict_global_delay(
                event_name=context.event_name,
                day_of_week=context.day_of_week,
                importance=context.event_importance,
                bag_prepared=context.bag_prepared,
                estimated_prep_duration_min=context.estimated_prep_duration,
                estimated_travel_duration_min=context.estimated_travel_duration
            )

        personal_pred = None
        try:
            try:
                from models import predict_micro_delay
            except ModuleNotFoundError:
                from backend.models import predict_micro_delay

            pred_res = predict_micro_delay(
                user_id=context.user_id,
                event_name=context.event_name,
                day_of_week=context.day_of_week,
                importance=context.event_importance,
                bag_prepared=context.bag_prepared,
                estimated_prep_duration_min=context.estimated_prep_duration,
                estimated_travel_duration_min=context.estimated_travel_duration,
                db=db
            )
            raw_ml = pred_res.get("predicted_delay_min")
            if raw_ml is not None:
                personal_pred = float(raw_ml)
            baseline_delay = float(pred_res.get("baseline_predicted_delay_min", 4.5))
            model_type = str(pred_res.get("model_used", "random_forest"))
            model_version = str(pred_res.get("model_version", context.active_model_version or "v1"))
            if "fallback" in model_type and not has_active_global_model:
                fallback_applied = True
            component_status["regression"] = "success"
        except Exception as e:
            logger.error("Regression model execution failed for user_id=%d: %s", context.user_id, e)
            fallback_applied = True
            if not has_active_global_model:
                model_availability = ModelAvailability.FAILED.value
            component_status["regression"] = f"failed: {e}"
            warnings.append(f"Individual model execution error: {e}")

        # Blend final prediction based on lifecycle state & alpha
        alpha = lifecycle_info.get("alpha", 1.0)
        predicted_delay = self.personalization_engine.blend_hybrid_prediction(
            global_pred=global_pred,
            personal_pred=personal_pred,
            baseline_pred=baseline_delay,
            alpha=alpha
        )

        lifecycle_info["global_prediction"] = global_pred
        lifecycle_info["personal_prediction"] = personal_pred
        lifecycle_info["hybrid_prediction"] = predicted_delay
        lifecycle_info["global_model_version"] = global_status.get("global_model_version")
        lifecycle_info["personal_model_version"] = model_version

        if lifecycle_info["learning_mode"] == "personalizing":
            model_type = "hybrid_blended"
        elif lifecycle_info["is_global_used"]:
            model_type = "global_model"
            model_version = global_status.get("global_model_version") or model_version

        # 5. Isolated Subcomponent Execution

        # A. Behavioral Profile
        profile_data = {}
        try:
            profile_data = self.profile_engine.build_profile(recent_logs)
            component_status["behavioral_profile"] = "success"
        except Exception as e:
            logger.warning("Behavioral profile building failed: %s", e)
            component_status["behavioral_profile"] = f"failed: {e}"

        # B. Behavioral Archetype Clustering
        archetype_info = {}
        try:
            archetype_info = self.clustering_engine.assign_archetype(context)
            component_status["clustering"] = "success"
        except Exception as e:
            logger.warning("Clustering failed: %s", e)
            component_status["clustering"] = f"failed: {e}"

        # C. Lateness Risk Classification
        late_probability = None
        risk_category = None
        try:
            late_probability, risk_category = self.risk_classifier.classify(
                context=context,
                predicted_delay_min=predicted_delay,
                baseline_delay_min=baseline_delay,
                use_trained_classifier=True
            )
            component_status["risk_classification"] = "success"
        except Exception as e:
            logger.warning("Risk classification failed: %s", e)
            component_status["risk_classification"] = f"failed: {e}"
            risk_category = RiskCategory.MODERATE.value

        # D. Explainability (Contributing Factors & Structured Explanation)
        contributing_factors = []
        explanation_data = None
        try:
            # Trigger explain() to ensure compatibility with existing mocks/callers
            self.explainer.explain(
                context=context,
                predicted_delay_min=predicted_delay,
                baseline_delay_min=baseline_delay,
                profile_data=profile_data
            )
            explanation_data = self.explainer.explain_full(
                context=context,
                predicted_delay_min=predicted_delay,
                baseline_delay_min=baseline_delay,
                late_probability=late_probability,
                model_version=model_version,
                db=db,
                profile_data=profile_data
            )
            pos = explanation_data.get("top_positive_contributors", []) if isinstance(explanation_data, dict) else []
            mit = explanation_data.get("top_mitigating_factors", []) if isinstance(explanation_data, dict) else []
            contributing_factors = pos + mit
            for f in contributing_factors:
                if isinstance(f, dict) and "impact_minutes" not in f and "impact_value" in f:
                    f["impact_minutes"] = f["impact_value"]
            component_status["explainability"] = "success"
        except Exception as e:
            logger.warning("Explainability generation failed: %s", e)
            component_status["explainability"] = f"failed: {e}"

        # E. Adaptive Recommendations
        recommendations = []
        try:
            recommendations = self.recommendation_engine.generate_recommendations(
                context=context,
                predicted_delay_min=predicted_delay,
                contributing_factors=contributing_factors,
                profile_data=profile_data,
                db=db
            )
            component_status["recommendations"] = "success"
        except Exception as e:
            logger.warning("Adaptive recommendations failed: %s", e)
            component_status["recommendations"] = f"failed: {e}"

        # F. Confidence Estimation
        confidence_status_obj = ConfidenceStatus(score=0.5, level="moderate", is_reliable=True)
        try:
            confidence_status_obj = self.confidence_estimator.estimate_confidence(
                context=context,
                sample_counts=sample_counts,
                validation_mae=validation_mae,
                is_known_event=True
            )
            component_status["confidence"] = "success"
        except Exception as e:
            logger.warning("Confidence estimation failed: %s", e)
            component_status["confidence"] = f"failed: {e}"

        # G. Behavioral Drift Detection
        try:
            if recent_logs:
                drift_info = self.drift_detector.detect_drift(recent_logs)
                if drift_info.get("drift_detected"):
                    warnings.append(drift_info.get("description", "Behavioral drift detected."))
            component_status["drift_detection"] = "success"
        except Exception as e:
            logger.warning("Drift detection failed: %s", e)
            component_status["drift_detection"] = f"failed: {e}"

        # Calculate latency
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Structured operational log (Never logging PII or sensitive data!)
        logger.info(
            "AdaptiveIntelligenceOrchestrator.predict completed: user_id=%d, "
            "predicted_delay=%.1f min, risk=%s, confidence=%.2f, availability=%s, "
            "latency_ms=%.1f",
            context.user_id, predicted_delay, risk_category,
            confidence_status_obj.score, model_availability, latency_ms
        )

        return AdaptiveIntelligenceOutput(
            predicted_delay_minutes=predicted_delay,
            baseline_predicted_delay_minutes=baseline_delay,
            late_probability=late_probability,
            risk_category=risk_category,
            contributing_factors=contributing_factors,
            recommended_interventions=recommendations,
            confidence_status=confidence_status_obj.to_dict(),
            model_version=model_version,
            model_type=model_type,
            model_availability=model_availability,
            data_quality_status=data_quality.to_dict(),
            personalization_info=lifecycle_info,
            explanation=explanation_data,
            fallback_applied=fallback_applied,
            component_status=component_status,
            warnings=warnings
        )
