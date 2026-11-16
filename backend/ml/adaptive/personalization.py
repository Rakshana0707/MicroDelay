"""Global-to-Personalized Learning Foundation for MicroDelay.

Manages the transition from global population / persona priors to fully
individualized behavioral models based on user data maturity.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger("microdelay.ml.adaptive.personalization")


class PersonalizationTier:
    GLOBAL_COLD_START = "global_cold_start"
    PERSONA_PRIOR = "persona_prior"
    BAYESIAN_BLENDED = "bayesian_blended"
    PERSONALIZED_AUTONOMOUS = "personalized_autonomous"


class PersonalizationEngine:
    """Evaluates learning tier and calculates empirical blending weights."""

    def __init__(self, full_personalization_threshold: int = 8):
        self.full_threshold = full_personalization_threshold

    def evaluate_maturity(self, real_samples: int, synthetic_samples: int) -> Dict[str, Any]:
        """Assess user data maturity and determine algorithmic personalization tier."""
        if real_samples >= self.full_threshold:
            tier = PersonalizationTier.PERSONALIZED_AUTONOMOUS
            weight_real = 1.0
            weight_prior = 0.0
            description = "Individualized model: Powered strictly by your genuine real-world departure logs."
        elif real_samples > 0:
            tier = PersonalizationTier.BAYESIAN_BLENDED
            weight_real = min(0.9, real_samples / float(self.full_threshold))
            weight_prior = 1.0 - weight_real
            description = (
                f"Early Calibration: Blending {real_samples}/{self.full_threshold} genuine observations "
                "with initial behavioral baseline."
            )
        elif synthetic_samples > 0:
            tier = PersonalizationTier.PERSONA_PRIOR
            weight_real = 0.0
            weight_prior = 1.0
            description = "Demonstration mode: Based on pre-seeded persona archetypes until real routines are logged."
        else:
            tier = PersonalizationTier.GLOBAL_COLD_START
            weight_real = 0.0
            weight_prior = 1.0
            description = "Cold start: Using global conservative prior estimates."

        return {
            "tier": tier,
            "real_samples": real_samples,
            "synthetic_samples": synthetic_samples,
            "weight_real": round(weight_real, 2),
            "weight_prior": round(weight_prior, 2),
            "is_personalized": (tier == PersonalizationTier.PERSONALIZED_AUTONOMOUS),
            "is_calibrated": (real_samples > 0),
            "description": description
        }

    def blend_predictions(
        self,
        personalized_pred: float,
        prior_pred: float,
        maturity_info: Dict[str, Any]
    ) -> float:
        """Blend individual and prior predictions based on maturity weights."""
        w_real = maturity_info.get("weight_real", 0.0)
        w_prior = maturity_info.get("weight_prior", 1.0)
        return (w_real * personalized_pred) + (w_prior * prior_pred)
