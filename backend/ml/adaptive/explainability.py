"""Prediction Explainability Foundation for MicroDelay.

Explains why the behavioral model predicted a specific delay by isolating
the impact of situational and behavioral context variables.
"""

from typing import List, Dict, Any, Optional
import logging
from .context import PredictionContext
from .schemas import ContributingFactor

logger = logging.getLogger("microdelay.ml.adaptive.explainability")


class PredictionExplainer:
    """Derives intuitive behavioral contributing factors for a prediction."""

    def explain(
        self,
        context: PredictionContext,
        predicted_delay_min: float,
        baseline_delay_min: float,
        profile_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Identify key contextual variables driving the predicted departure delay."""
        factors: List[ContributingFactor] = []

        try:
            # 1. Bag Preparation Factor
            if not context.bag_prepared:
                factors.append(ContributingFactor(
                    factor="bag_unprepared",
                    impact_minutes=3.8,
                    direction="increases_delay",
                    description="Leaving bag packing to morning of departure increases item search and packing delays."
                ))
            else:
                factors.append(ContributingFactor(
                    factor="bag_prepared",
                    impact_minutes=-2.5,
                    direction="decreases_delay",
                    description="Having work/study materials pre-packed saves crucial morning transition minutes."
                ))

            # 2. Day of Week Effect
            if context.day_of_week == "Monday":
                factors.append(ContributingFactor(
                    factor="monday_start_inertia",
                    impact_minutes=2.2,
                    direction="increases_delay",
                    description="Monday mornings exhibit elevated schedule re-adjustment and departure lag."
                ))
            elif context.day_of_week in ("Saturday", "Sunday"):
                factors.append(ContributingFactor(
                    factor="weekend_commute_flow",
                    impact_minutes=-1.5,
                    direction="decreases_delay",
                    description="Weekend departure schedules encounter fewer commuting bottlenecks."
                ))

            # 3. Preparation Duration Bias
            if profile_data and profile_data.get("prep_bias_min", 0.0) > 2.0:
                bias = profile_data["prep_bias_min"]
                factors.append(ContributingFactor(
                    factor="personal_estimation_bias",
                    impact_minutes=bias,
                    direction="increases_delay",
                    description=f"Your historical routines show an average {bias} min underestimation of actual prep time."
                ))

            # 4. Importance Level
            if context.event_importance in ("high", "critical"):
                factors.append(ContributingFactor(
                    factor="high_importance_diligence",
                    impact_minutes=-1.2,
                    direction="decreases_delay",
                    description="High priority events typically prompt earlier departure vigilance."
                ))

            # 5. Model vs Baseline Residual Difference
            model_delta = predicted_delay_min - baseline_delay_min
            if abs(model_delta) >= 1.5:
                direction = "increases_delay" if model_delta > 0 else "decreases_delay"
                factors.append(ContributingFactor(
                    factor="personalized_ml_adjustment",
                    impact_minutes=model_delta,
                    direction=direction,
                    description=f"Personal ML model adjusted historical baseline by {round(model_delta, 1)} min based on feature combination."
                ))

            # Sort by absolute impact descending
            factors.sort(key=lambda f: abs(f.impact_minutes), reverse=True)
            return [f.to_dict() for f in factors[:5]]

        except Exception as e:
            logger.warning("Error generating prediction explanation: %s", e)
            return []
