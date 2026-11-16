"""Adaptive Behavioral Recommendations Foundation for MicroDelay.

Generates targeted, actionable behavioral interventions matched to identified
delay risk factors and personal habit patterns.
"""

from typing import List, Dict, Any, Optional
import logging
from .context import PredictionContext
from .schemas import RecommendedIntervention

logger = logging.getLogger("microdelay.ml.adaptive.recommendations")


class AdaptiveRecommendationEngine:
    """Recommends actionable habit adjustments based on context and risk factors."""

    def generate_recommendations(
        self,
        context: PredictionContext,
        predicted_delay_min: float,
        contributing_factors: List[Dict[str, Any]],
        profile_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Synthesize prioritized, high-impact behavioral interventions."""
        recommendations: List[RecommendedIntervention] = []

        try:
            factor_names = {f["factor"] for f in contributing_factors}

            # 1. Unprepared Bag Recommendation
            if not context.bag_prepared or "bag_unprepared" in factor_names:
                recommendations.append(RecommendedIntervention(
                    action="Evening Bag Packing Checklist",
                    description="Pack your bag, chargers, keys, and daily essentials the night before.",
                    expected_delay_reduction_min=3.8,
                    category="preparation",
                    priority="high"
                ))

            # 2. Preparation Estimation Bias
            prep_bias = profile_data.get("prep_bias_min", 0.0) if profile_data else 0.0
            if prep_bias > 2.0 or "personal_estimation_bias" in factor_names:
                buffer = round(max(3.0, prep_bias), 0)
                recommendations.append(RecommendedIntervention(
                    action=f"Add {int(buffer)}-Minute Buffer to Prep Timer",
                    description=f"Your preparation routinely takes ~{round(prep_bias, 1)} minutes longer than planned. Adjust your routine start time.",
                    expected_delay_reduction_min=prep_bias,
                    category="timing",
                    priority="high"
                ))

            # 3. Monday Inertia
            if context.day_of_week == "Monday":
                recommendations.append(RecommendedIntervention(
                    action="Monday Morning Buffer Rule",
                    description="Start your morning routine 10 minutes earlier on Mondays to counteract weekend-to-weekday schedule shifts.",
                    expected_delay_reduction_min=2.5,
                    category="timing",
                    priority="medium"
                ))

            # 4. Phone Distraction / Last Task Bottleneck
            bottleneck = profile_data.get("dominant_bottleneck") if profile_data else None
            if bottleneck in ("phone_distraction", "last_minute_tasks"):
                recommendations.append(RecommendedIntervention(
                    action="15-Minute Pre-Departure Screen Lock",
                    description="Switch phone to 'Do Not Disturb' mode 15 minutes before planned departure to eliminate notification rabbit holes.",
                    expected_delay_reduction_min=3.5,
                    category="distraction",
                    priority="high"
                ))

            # 5. Default Punctuality Reinforcement
            if len(recommendations) < 2 and predicted_delay_min > 2.0:
                recommendations.append(RecommendedIntervention(
                    action="Calibrate Routine Checklist",
                    description="Review your morning preparation milestones 30 minutes before leaving.",
                    expected_delay_reduction_min=2.0,
                    category="preparation",
                    priority="low"
                ))

            return [r.to_dict() for r in recommendations[:3]]

        except Exception as e:
            logger.warning("Error generating adaptive recommendations: %s", e)
            return []
