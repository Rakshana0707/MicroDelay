"""Prediction Confidence & Reliability Foundation for MicroDelay.

Evaluates how trustworthy an ML delay prediction is based on sample size,
extrapolation distance, and historical model residual variance.
"""

from typing import Dict, Any, List, Optional
import logging
from .context import PredictionContext
from .schemas import ConfidenceStatus, ConfidenceLevel

logger = logging.getLogger("microdelay.ml.adaptive.confidence")


class ConfidenceEstimator:
    """Estimates prediction confidence score and reliability flags."""

    def estimate_confidence(
        self,
        context: PredictionContext,
        sample_counts: Dict[str, int],
        validation_mae: Optional[float] = None,
        is_known_event: bool = True
    ) -> ConfidenceStatus:
        """Calculate normalized confidence score [0.0, 1.0] and diagnostic reasons."""
        score = 0.50
        reasons: List[str] = []

        real_count = sample_counts.get("real_samples", 0)
        synthetic_count = sample_counts.get("synthetic_samples", 0)
        total_count = sample_counts.get("total_samples", real_count + synthetic_count)

        # 1. Sample Size Assessment
        if real_count >= 20:
            score += 0.35
            reasons.append(f"Strong real data sample size ({real_count} genuine departure observations).")
        elif real_count >= 8:
            score += 0.25
            reasons.append(f"Calibrated personal data sample size ({real_count} genuine departure observations).")
        elif synthetic_count >= 30:
            score += 0.15
            reasons.append(f"Demonstration data profile with {synthetic_count} pre-seeded persona routines.")
        elif total_count > 0:
            score += 0.05
            reasons.append("Early phase data collection; preliminary estimates only.")
        else:
            score -= 0.25
            reasons.append("Zero user history; relying on cold-start heuristic priors.")

        # 2. Event Familiarity Assessment
        if not is_known_event:
            score -= 0.15
            reasons.append(f"Event '{context.event_name}' has not been observed frequently in user history.")
        else:
            score += 0.05

        # 3. Model Accuracy Assessment
        if validation_mae is not None:
            if validation_mae <= 3.0:
                score += 0.10
                reasons.append(f"Model displays tight historical validation accuracy (MAE: {round(validation_mae, 1)} min).")
            elif validation_mae > 7.0:
                score -= 0.15
                reasons.append(f"High historical error dispersion (MAE: {round(validation_mae, 1)} min).")

        # 4. Contextual Sanity Checks
        prep_dur = float(context.estimated_prep_duration)
        if prep_dur > 120.0 or prep_dur < 5.0:
            score -= 0.10
            reasons.append(f"Estimated prep duration ({prep_dur} min) is atypical, requiring extrapolation.")

        # Clamp score to [0.05, 0.98]
        clamped_score = max(0.05, min(0.98, score))

        # Categorize level
        if clamped_score >= 0.70:
            level = ConfidenceLevel.HIGH.value
            is_reliable = True
        elif clamped_score >= 0.45:
            level = ConfidenceLevel.MODERATE.value
            is_reliable = True
        else:
            level = ConfidenceLevel.LOW.value
            is_reliable = False

        return ConfidenceStatus(
            score=round(clamped_score, 2),
            level=level,
            is_reliable=is_reliable,
            reasons=reasons
        )
