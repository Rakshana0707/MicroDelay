"""Lateness Risk Classification Foundation for MicroDelay.

Classifies continuous delay expectations into discrete risk categories and
calibrated departure lateness probabilities.
"""

import math
import logging
from typing import Tuple, Optional, Dict, Any
from .schemas import RiskCategory
from .context import PredictionContext

logger = logging.getLogger("microdelay.ml.adaptive.risk")


class LatenessRiskClassifier:
    """Classifies departure context into risk categories and late probabilities."""

    def __init__(
        self,
        low_threshold_min: float = 1.5,
        moderate_threshold_min: float = 5.0,
        high_threshold_min: float = 10.0,
        steepness: float = 0.35,
        midpoint_offset_min: float = 2.0
    ):
        self.low_threshold = low_threshold_min
        self.moderate_threshold = moderate_threshold_min
        self.high_threshold = high_threshold_min
        self.steepness = steepness
        self.midpoint_offset = midpoint_offset_min

    def compute_late_probability(
        self,
        predicted_delay_min: float,
        residual_std: float = 3.5
    ) -> float:
        """Compute calibrated probability of departure being delayed (> 0 min).

        Uses a calibrated logistic function anchored at the midpoint offset.
        """
        try:
            # Shifted sigmoid: P(delay > 0) increases smoothly with predicted delay
            scale = max(1.0, residual_std)
            k = self.steepness * (3.5 / scale)
            x = predicted_delay_min - self.midpoint_offset
            # Clamping to avoid numerical overflow
            x_clamped = max(-20.0, min(20.0, k * x))
            prob = 1.0 / (1.0 + math.exp(-x_clamped))
            return max(0.01, min(0.99, prob))
        except Exception as e:
            logger.warning("Error calculating late probability: %s. Using empirical fallback.", e)
            if predicted_delay_min <= 0:
                return 0.15
            elif predicted_delay_min < 5.0:
                return 0.50
            else:
                return 0.85

    def classify_risk(
        self,
        predicted_delay_min: float,
        late_probability: float
    ) -> RiskCategory:
        """Categorize into discrete risk level."""
        if predicted_delay_min <= self.low_threshold and late_probability < 0.40:
            return RiskCategory.LOW
        elif predicted_delay_min <= self.moderate_threshold and late_probability < 0.65:
            return RiskCategory.MODERATE
        elif predicted_delay_min <= self.high_threshold and late_probability < 0.85:
            return RiskCategory.HIGH
        else:
            return RiskCategory.EXTREME

    def classify(
        self,
        context: PredictionContext,
        predicted_delay_min: float,
        baseline_delay_min: float = 4.5,
        residual_std: float = 3.5
    ) -> Tuple[Optional[float], Optional[str]]:
        """Main classification entrypoint returning (late_probability, risk_category)."""
        try:
            prob = self.compute_late_probability(predicted_delay_min, residual_std=residual_std)
            category = self.classify_risk(predicted_delay_min, prob)
            return prob, category.value
        except Exception as e:
            logger.error("Risk classification failed: %s", e)
            return None, RiskCategory.MODERATE.value
