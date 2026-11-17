"""Prediction Explainability Foundation for MicroDelay.

Explains why the behavioral model predicted a specific delay by isolating
the impact of situational and behavioral context variables.
"""

from typing import List, Dict, Any, Optional
import logging
from .context import PredictionContext
from .schemas import ContributingFactor

try:
    from ml.explainability import (
        ExplainabilityService,
        LocalPredictionExplainer,
        HistoricalBehavioralComparator,
        CAUSATION_DISCLAIMER
    )
except ModuleNotFoundError:
    from backend.ml.explainability import (
        ExplainabilityService,
        LocalPredictionExplainer,
        HistoricalBehavioralComparator,
        CAUSATION_DISCLAIMER
    )

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
        try:
            explanation = LocalPredictionExplainer.explain_prediction(
                model=None,
                feature_names=[],
                x_vector=None,
                context=context,
                predicted_delay_min=predicted_delay_min,
                late_probability=None,
                model_version=getattr(context, "active_model_version", "v1") or "v1",
                historical_profile=profile_data
            )
            # Combine positive and mitigating factors for backward compatibility
            all_factors = explanation.top_positive_contributors + explanation.top_mitigating_factors
            all_factors.sort(key=lambda f: abs(f.get("impact_value", 0.0)), reverse=True)
            # Ensure impact_minutes key is present for backward compatibility
            for f in all_factors:
                if "impact_minutes" not in f and "impact_value" in f:
                    f["impact_minutes"] = f["impact_value"]
            return all_factors[:5]
        except Exception as e:
            logger.warning("Error generating prediction explanation via ExplainabilityService: %s", e)
            return []

    def explain_full(
        self,
        context: PredictionContext,
        predicted_delay_min: float,
        baseline_delay_min: float,
        late_probability: Optional[float] = None,
        model_version: str = "default_fallback",
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        x_vector: Optional[Any] = None,
        db: Optional[Any] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        persist: bool = True
    ) -> Dict[str, Any]:
        """Generate complete explanation contract with causation disclaimers and audit trail."""
        return ExplainabilityService.explain_and_record(
            db=db,
            user_id=context.user_id,
            context=context,
            predicted_delay_min=predicted_delay_min,
            late_probability=late_probability,
            model_version=model_version,
            model=model,
            feature_names=feature_names,
            x_vector=x_vector,
            historical_profile=profile_data,
            persist=persist
        )

