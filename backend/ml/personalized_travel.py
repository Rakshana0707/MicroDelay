"""Personalized Travel Estimation Engine (Prompt 3.4).

Bridges Phase 2 behavioral intelligence and external routing provider estimates.
Calculates personal travel-time bias, day-specific variations, rush-hour factors,
and applies empirical Bayesian shrinkage to avoid over-correcting on sparse user data.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np
import logging
from sqlalchemy.orm import Session

try:
    from database import DepartureLog
except ImportError:
    from backend.database import DepartureLog

logger = logging.getLogger(__name__)


class PersonalizedTravelEstimate:
    """Detailed breakdown of personalized travel duration adjustment."""

    def __init__(
        self,
        provider_duration_min: float,
        personalized_duration_min: float,
        bias_ratio: float,
        day_of_week_factor: float,
        time_of_day_factor: float,
        sample_count: int,
        confidence: float,
        adjustment_min: float,
        explanation: str
    ):
        self.provider_duration_min = round(provider_duration_min, 1)
        self.personalized_duration_min = round(personalized_duration_min, 1)
        self.bias_ratio = round(bias_ratio, 3)
        self.day_of_week_factor = round(day_of_week_factor, 3)
        self.time_of_day_factor = round(time_of_day_factor, 3)
        self.sample_count = sample_count
        self.confidence = round(confidence, 2)
        self.adjustment_min = round(adjustment_min, 1)
        self.explanation = explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_duration_min": self.provider_duration_min,
            "personalized_duration_min": self.personalized_duration_min,
            "bias_ratio": self.bias_ratio,
            "day_of_week_factor": self.day_of_week_factor,
            "time_of_day_factor": self.time_of_day_factor,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
            "adjustment_min": self.adjustment_min,
            "explanation": self.explanation
        }


class PersonalizedTravelEngine:
    """Calculates personalized travel durations based on historical journey logs."""

    # Rush hour windows (local military hours): 07:30 - 09:30 and 16:30 - 18:30
    RUSH_HOURS_MORNING = (7.5, 9.5)
    RUSH_HOURS_EVENING = (16.5, 18.5)

    @classmethod
    def _is_rush_hour(cls, dep_dt: datetime) -> bool:
        hr = dep_dt.hour + (dep_dt.minute / 60.0)
        return (cls.RUSH_HOURS_MORNING[0] <= hr <= cls.RUSH_HOURS_MORNING[1]) or \
               (cls.RUSH_HOURS_EVENING[0] <= hr <= cls.RUSH_HOURS_EVENING[1])

    @classmethod
    def estimate(
        cls,
        user_id: int,
        provider_duration_min: float,
        departure_time: Optional[datetime],
        travel_mode: str,
        db: Session
    ) -> PersonalizedTravelEstimate:
        """Calculate personalized travel estimate blending provider baseline with user behavioral bias."""
        if provider_duration_min <= 0:
            provider_duration_min = 15.0

        dep_dt = departure_time or datetime.now()
        dep_day = dep_dt.strftime("%A")
        is_rush = cls._is_rush_hour(dep_dt)

        # Query user's historical departure logs with valid actual travel durations
        logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id,
            DepartureLog.actual_travel_duration_min > 0,
            DepartureLog.estimated_travel_duration_min > 0
        ).all()

        n_samples = len(logs)

        # Baseline defaults if no history
        if n_samples < 3:
            rush_mult = 1.15 if (is_rush and travel_mode == "driving") else 1.0
            personalized = provider_duration_min * rush_mult
            return PersonalizedTravelEstimate(
                provider_duration_min=provider_duration_min,
                personalized_duration_min=personalized,
                bias_ratio=1.0,
                day_of_week_factor=1.0,
                time_of_day_factor=rush_mult,
                sample_count=n_samples,
                confidence=0.5,
                adjustment_min=personalized - provider_duration_min,
                explanation="Provider baseline used (gathering initial user journey observations)."
            )

        # 1. Compute overall travel ratio = actual / estimated
        ratios = []
        day_ratios = []
        for log in logs:
            est = float(log.estimated_travel_duration_min)
            act = float(log.actual_travel_duration_min)
            if est > 0:
                r = act / est
                # filter extreme outlier anomalies (> 4.0 or < 0.25)
                if 0.25 <= r <= 4.0:
                    ratios.append(r)
                    if log.day_of_week and log.day_of_week.lower() == dep_day.lower():
                        day_ratios.append(r)

        if not ratios:
            ratios = [1.0]

        mean_user_ratio = float(np.mean(ratios))

        # Day-of-week factor
        if len(day_ratios) >= 3:
            day_factor = float(np.mean(day_ratios)) / max(0.01, mean_user_ratio)
            # Clamp day factor between 0.85 and 1.25
            day_factor = max(0.85, min(1.25, day_factor))
        else:
            day_factor = 1.0

        # Time of day / Rush hour factor
        time_factor = 1.12 if (is_rush and travel_mode == "driving") else 1.0

        # Empirical Bayesian shrinkage towards 1.0 (provider anchor)
        # Weight increases with sample size: confidence = n / (n + 10)
        confidence = min(0.95, n_samples / (n_samples + 10.0))
        shrunk_ratio = (confidence * mean_user_ratio) + ((1.0 - confidence) * 1.0)

        raw_multiplier = shrunk_ratio * day_factor * time_factor
        # Clamp multiplier to realistic bounds [0.65, 1.85]
        bounded_multiplier = max(0.65, min(1.85, raw_multiplier))

        personalized_dur = provider_duration_min * bounded_multiplier
        adj_min = personalized_dur - provider_duration_min

        explanation_parts = []
        if abs(shrunk_ratio - 1.0) >= 0.05:
            pct = int(round((shrunk_ratio - 1.0) * 100))
            direction = "slower" if pct > 0 else "faster"
            explanation_parts.append(f"historical journeys average {abs(pct)}% {direction} than estimated")
        if day_factor != 1.0 and abs(day_factor - 1.0) >= 0.04:
            pct = int(round((day_factor - 1.0) * 100))
            explanation_parts.append(f"{dep_day} traffic pattern factor ({pct:+d}%)")
        if time_factor > 1.0:
            explanation_parts.append("peak rush-hour window factor (+12%)")

        if explanation_parts:
            explanation = f"Personalized adjustment: {', '.join(explanation_parts)}."
        else:
            explanation = "Aligned closely with provider estimates."

        return PersonalizedTravelEstimate(
            provider_duration_min=provider_duration_min,
            personalized_duration_min=personalized_dur,
            bias_ratio=shrunk_ratio,
            day_of_week_factor=day_factor,
            time_of_day_factor=time_factor,
            sample_count=n_samples,
            confidence=confidence,
            adjustment_min=adj_min,
            explanation=explanation
        )
