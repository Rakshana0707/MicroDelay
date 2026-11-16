"""Central Adaptive Intelligence Orchestration Service for MicroDelay.

Coordinates regression prediction, lateness risk classification, feature explainability,
adaptive recommendations, behavioral drift detection, and confidence scoring into
a unified, failure-isolated prediction pipeline.
"""

import time
import logging
from typing import Optional, Dict, Any, List
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

        # 3. Model Availability State Assessment
        if not is_sufficient and real_count == 0 and synth_count == 0:
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
            predicted_delay = float(pred_res.get("predicted_delay_min", 4.5))
            baseline_delay = float(pred_res.get("baseline_predicted_delay_min", 4.5))
            model_type = str(pred_res.get("model_used", "random_forest"))
            model_version = str(pred_res.get("model_version", context.active_model_version or "v1"))
            if "fallback" in model_type:
                fallback_applied = True
            component_status["regression"] = "success"
        except Exception as e:
            logger.error("Regression model execution failed for user_id=%d: %s", context.user_id, e)
            fallback_applied = True
            model_availability = ModelAvailability.FAILED.value
            predicted_delay = 4.5
            baseline_delay = 4.5
            component_status["regression"] = f"failed: {e}"
            warnings.append("Regression model prediction failed; empirical fallback applied.")

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
                baseline_delay_min=baseline_delay
            )
            component_status["risk_classification"] = "success"
        except Exception as e:
            logger.warning("Risk classification failed: %s", e)
            component_status["risk_classification"] = f"failed: {e}"
            risk_category = RiskCategory.MODERATE.value

        # D. Explainability (Contributing Factors)
        contributing_factors = []
        try:
            contributing_factors = self.explainer.explain(
                context=context,
                predicted_delay_min=predicted_delay,
                baseline_delay_min=baseline_delay,
                profile_data=profile_data
            )
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
                profile_data=profile_data
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
            fallback_applied=fallback_applied,
            component_status=component_status,
            warnings=warnings
        )
