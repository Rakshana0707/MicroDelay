"""Prediction Confidence & Reliability Foundation for MicroDelay (Prompt 2.9).

Evaluates how trustworthy an ML delay prediction is based on documented rule-based
data sufficiency, conformal error dispersion, calibration quality, and behavioral stability.
"""

from typing import Dict, Any, List, Optional
import logging
from .context import PredictionContext
from .schemas import ConfidenceStatus, ConfidenceLevel, ReliabilityStatus

logger = logging.getLogger("microdelay.ml.adaptive.confidence")


class ConfidenceEstimator:
    """Estimates prediction confidence score and reliability flags based on documented rules."""

    def estimate_confidence(
        self,
        context: PredictionContext,
        sample_counts: Dict[str, int],
        validation_mae: Optional[float] = None,
        is_known_event: bool = True,
        is_drift_detected: bool = False,
        is_model_degraded: bool = False
    ) -> ConfidenceStatus:
        """Calculate documented reliability level and evidence-grounded diagnostic reasons."""
        reasons: List[str] = []

        real_count = sample_counts.get("real_samples", 0)
        synthetic_count = sample_counts.get("synthetic_samples", 0)
        total_count = sample_counts.get("total_samples", real_count + synthetic_count)

        # 1. Sample Size Assessment
        if real_count >= 30:
            reasons.append(f"Robust genuine sample size ({real_count} departure checkout observations).")
        elif real_count >= 15:
            reasons.append(f"Moderate genuine sample size ({real_count} departure checkout observations).")
        elif real_count >= 8:
            reasons.append(f"Minimal genuine sample size ({real_count} departure observations; minimum personal threshold).")
        elif synthetic_count >= 20:
            reasons.append(f"Demonstration data profile with {synthetic_count} pre-seeded persona routines.")
        elif total_count > 0:
            reasons.append(f"Early data collection phase ({total_count} observations); preliminary estimates only.")
        else:
            reasons.append("Zero user history; relying on shared global prior.")

        # 2. Event Familiarity Assessment
        if not is_known_event:
            reasons.append(f"Event '{context.event_name}' has not been observed frequently in user history.")

        # 3. Model Accuracy Assessment
        if validation_mae is not None:
            if validation_mae <= 3.5:
                reasons.append(f"Model displays tight historical validation accuracy (MAE: {round(validation_mae, 1)} min).")
            elif validation_mae > 7.0:
                reasons.append(f"High historical error dispersion (MAE: {round(validation_mae, 1)} min).")

        # 4. Contextual Sanity Checks
        prep_dur = float(context.estimated_prep_duration)
        if prep_dur > 180.0 or prep_dur < 5.0:
            reasons.append(f"Estimated prep duration ({prep_dur} min) is atypical, requiring extrapolation.")

        # 5. Drift & Degradation Checks
        if is_model_degraded:
            reasons.append("Predictive model monitoring flagged concept drift / accuracy degradation.")
        elif is_drift_detected:
            reasons.append("Recent behavioral routine drift detected; intervals widened.")

        # Determine level according to documented rules
        if is_model_degraded:
            level = ReliabilityStatus.DEGRADED_MODEL.value
            score = 0.30
            is_reliable = False
        elif real_count < 8 and synthetic_count < 20:
            level = ReliabilityStatus.INSUFFICIENT_HISTORY.value
            score = 0.20
            is_reliable = False
        elif real_count >= 30 and (validation_mae is None or validation_mae <= 3.5) and not is_drift_detected:
            level = ReliabilityStatus.HIGH_RELIABILITY.value
            score = 0.85
            is_reliable = True
        elif (real_count >= 15 or synthetic_count >= 20) and (validation_mae is None or validation_mae <= 5.5):
            level = ReliabilityStatus.MODERATE_RELIABILITY.value
            score = 0.65
            is_reliable = True
        else:
            level = ReliabilityStatus.LOW_RELIABILITY.value
            score = 0.40
            is_reliable = True

        return ConfidenceStatus(
            score=score,
            level=level,
            is_reliable=is_reliable,
            reasons=reasons
        )
