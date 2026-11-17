"""Global-to-Personalized Learning Foundation for MicroDelay.

Manages the transition from global population / persona priors to fully
individualized behavioral models based on user data maturity, cold-start handling,
and dynamic hybrid prediction blending.
"""

import enum
import logging
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger("microdelay.ml.adaptive.personalization")


class PersonalizationState(str, enum.Enum):
    """Explicit lifecycle states for cold-start handling and model personalization."""
    NEW_USER = "new_user"
    COLLECTING_HISTORY = "collecting_history"
    GLOBAL_PREDICTION = "global_prediction"
    PERSONALIZING = "personalizing"
    PERSONALIZED = "personalized"
    INSUFFICIENT_DATA = "insufficient_data"


class PersonalizationTier:
    """Legacy tier definitions for backward compatibility."""
    GLOBAL_COLD_START = "global_cold_start"
    PERSONA_PRIOR = "persona_prior"
    BAYESIAN_BLENDED = "bayesian_blended"
    PERSONALIZED_AUTONOMOUS = "personalized_autonomous"


def compute_dynamic_alpha(
    genuine_samples: int,
    full_threshold: int = 8,
    personal_mae: Optional[float] = None,
    global_mae: Optional[float] = None,
    delay_std: Optional[float] = None
) -> float:
    """Compute adaptive blending weight alpha for:

        hybrid_prediction = alpha * global_prediction + (1 - alpha) * personal_prediction

    Where alpha in [0.0, 1.0]:
      - 1.0 = 100% global prediction (early cold-start)
      - 0.0 = 100% personalized prediction (fully autonomous)

    Alpha dynamically adapts based on:
      1. Number of genuine observations (logarithmic/linear transition).
      2. Historical relative prediction error (personal MAE vs global MAE).
      3. Individual behavioral stability (variance/std of delay).
    """
    if genuine_samples <= 0:
        return 1.0
    if genuine_samples >= full_threshold:
        return 0.0

    # 1. Base sample ratio: linear decay from 1.0 to 0.0
    base_alpha = 1.0 - (float(genuine_samples) / float(full_threshold))

    # 2. Performance-based adjustment
    perf_adj = 0.0
    if personal_mae is not None and global_mae is not None and global_mae > 0:
        # If personal model has higher error than global, lean more toward global (+alpha)
        # If personal model has lower error than global, lean faster toward personal (-alpha)
        rel_diff = (personal_mae - global_mae) / global_mae
        perf_adj = max(-0.20, min(0.20, rel_diff * 0.15))

    # 3. Behavioral stability adjustment
    stability_adj = 0.0
    if delay_std is not None:
        # High volatility (> 5 min std): increase regularization weight (+alpha)
        # High consistency (< 2.5 min std): trust individual patterns sooner (-alpha)
        if delay_std > 5.0:
            stability_adj = min(0.15, (delay_std - 5.0) * 0.03)
        elif delay_std < 2.5:
            stability_adj = max(-0.15, (delay_std - 2.5) * 0.04)

    final_alpha = base_alpha + perf_adj + stability_adj
    return round(max(0.05, min(0.95, final_alpha)), 3)


class PersonalizationEngine:
    """Evaluates cold-start lifecycle states and orchestrates dynamic hybrid prediction blending."""

    def __init__(self, full_personalization_threshold: int = 8, hybrid_start_threshold: int = 4):
        self.full_threshold = full_personalization_threshold
        self.hybrid_threshold = hybrid_start_threshold

    def evaluate_lifecycle(
        self,
        genuine_samples: int,
        synthetic_samples: int = 0,
        has_active_personal_model: bool = False,
        has_active_global_model: bool = False,
        personal_mae: Optional[float] = None,
        global_mae: Optional[float] = None,
        delay_std: Optional[float] = None
    ) -> Dict[str, Any]:
        """Assess the user's explicit lifecycle state, cold-start tier, and dynamic blending weight."""
        # A. Fully Personalized
        if genuine_samples >= self.full_threshold and has_active_personal_model:
            state = PersonalizationState.PERSONALIZED
            alpha = 0.0
            is_personalized = True
            is_global_used = False
            why_unavailable = None
            explanation = (
                f"Fully personalized individual model active. Trained strictly on your "
                f"{genuine_samples} genuine real-world departure records."
            )

        # B. Personalizing (Hybrid Blending Phase: 4-7 observations)
        elif genuine_samples >= self.hybrid_threshold:
            state = PersonalizationState.PERSONALIZING
            is_personalized = False
            is_global_used = has_active_global_model
            alpha = compute_dynamic_alpha(
                genuine_samples=genuine_samples,
                full_threshold=self.full_threshold,
                personal_mae=personal_mae,
                global_mae=global_mae,
                delay_std=delay_std
            )
            why_unavailable = (
                f"Requires {self.full_threshold - genuine_samples} more genuine routines to activate "
                f"fully independent personal validation."
            )
            explanation = (
                f"Hybrid intelligence active: Blending {int((1.0 - alpha) * 100)}% personal habit learning "
                f"with {int(alpha * 100)}% global cohort priors based on behavioral stability."
            )

        # C. Collecting History (1-3 observations)
        elif genuine_samples > 0:
            state = PersonalizationState.COLLECTING_HISTORY
            alpha = 1.0 if has_active_global_model else 0.0
            is_personalized = False
            is_global_used = has_active_global_model
            why_unavailable = (
                f"Gathering initial history ({genuine_samples}/{self.full_threshold} logged). "
                f"Hybrid blending activates at {self.hybrid_threshold} checkouts."
            )
            explanation = (
                f"Collecting genuine history ({genuine_samples}/{self.full_threshold}). "
                + ("Operating on shared global model baseline." if has_active_global_model else "Using statistical baseline.")
            )

        # D. New User (0 genuine observations)
        else:
            if has_active_global_model:
                state = PersonalizationState.GLOBAL_PREDICTION
                alpha = 1.0
                is_personalized = False
                is_global_used = True
                why_unavailable = (
                    f"No genuine departure routines recorded yet. Complete morning checkouts "
                    f"to build your personalized profile ({genuine_samples}/{self.full_threshold})."
                )
                explanation = "Cold-start protection: Serving predictions via the validated global cross-user model."
            elif synthetic_samples > 0:
                state = PersonalizationState.NEW_USER
                alpha = 1.0
                is_personalized = False
                is_global_used = False
                why_unavailable = "Operating on synthetic demonstration records until real routines are logged."
                explanation = "Demonstration mode: Synthetic persona prior active until genuine routines are tracked."
            else:
                state = PersonalizationState.INSUFFICIENT_DATA
                alpha = 1.0
                is_personalized = False
                is_global_used = False
                why_unavailable = "New user with neither personal history nor shared global model available."
                explanation = "Cold-start fallback: Operating on historical statistical baseline."

        return {
            "learning_mode": state.value,
            "genuine_observations_count": genuine_samples,
            "synthetic_observations_count": synthetic_samples,
            "full_personalization_threshold": self.full_threshold,
            "hybrid_start_threshold": self.hybrid_threshold,
            "is_personalized": is_personalized,
            "is_global_used": is_global_used,
            "alpha": alpha,
            "weight_global": round(alpha, 3),
            "weight_personal": round(1.0 - alpha, 3),
            "status_explanation": explanation,
            "why_personal_unavailable": why_unavailable
        }

    def blend_predictions(
        self,
        personalized_pred: float,
        prior_pred: float,
        maturity_info: Dict[str, Any]
    ) -> float:
        """Legacy helper for blending individual and prior predictions."""
        w_real = maturity_info.get("weight_real", maturity_info.get("weight_personal", 0.0))
        w_prior = maturity_info.get("weight_prior", maturity_info.get("weight_global", 1.0))
        return (w_real * personalized_pred) + (w_prior * prior_pred)

    def blend_hybrid_prediction(
        self,
        global_pred: Optional[float],
        personal_pred: Optional[float],
        baseline_pred: float,
        alpha: float
    ) -> float:
        """Perform hybrid blending between global, personal, and fallback baseline predictions."""
        # If alpha is 0.0 (pure personalized) and personal prediction is available
        if alpha <= 0.0 and personal_pred is not None:
            return round(personal_pred, 1)

        # If alpha is 1.0 (pure global) and global prediction is available
        if alpha >= 1.0 and global_pred is not None:
            return round(global_pred, 1)

        # Hybrid blending between global and personal
        if global_pred is not None and personal_pred is not None:
            blended = (alpha * global_pred) + ((1.0 - alpha) * personal_pred)
            return round(blended, 1)

        # Graceful fallbacks if either prediction is unavailable
        if personal_pred is not None:
            return round(personal_pred, 1)
        if global_pred is not None:
            return round(global_pred, 1)
        return round(baseline_pred, 1)

    def evaluate_maturity(self, real_samples: int, synthetic_samples: int) -> Dict[str, Any]:
        """Legacy evaluate_maturity method preserved for full backward compatibility."""
        if real_samples >= self.full_threshold:
            tier = PersonalizationTier.PERSONALIZED_AUTONOMOUS
            weight_real = 1.0
            weight_prior = 0.0
            desc = "Individualized model: Powered strictly by your genuine real-world departure logs."
        elif real_samples > 0:
            tier = PersonalizationTier.BAYESIAN_BLENDED
            weight_real = min(0.9, real_samples / float(self.full_threshold))
            weight_prior = 1.0 - weight_real
            desc = f"Early Calibration: Blending {real_samples}/{self.full_threshold} genuine observations with initial behavioral baseline."
        elif synthetic_samples > 0:
            tier = PersonalizationTier.PERSONA_PRIOR
            weight_real = 0.0
            weight_prior = 1.0
            desc = "Demonstration mode: Based on pre-seeded persona archetypes until real routines are logged."
        else:
            tier = PersonalizationTier.GLOBAL_COLD_START
            weight_real = 0.0
            weight_prior = 1.0
            desc = "Cold start: Using global conservative prior estimates."

        return {
            "tier": tier,
            "real_samples": real_samples,
            "synthetic_samples": synthetic_samples,
            "weight_real": round(weight_real, 2),
            "weight_prior": round(weight_prior, 2),
            "is_personalized": (tier == PersonalizationTier.PERSONALIZED_AUTONOMOUS),
            "is_calibrated": (real_samples > 0),
            "description": desc
        }
