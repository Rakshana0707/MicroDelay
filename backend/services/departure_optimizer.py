"""Dynamic Departure-Time Optimizer for MicroDelay.

Synthesizes behavioral preparation predictions, real-time travel duration,
traffic congestion adjustments, and weather safety buffers into an optimal,
explainable departure timeline.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict

try:
    from services.context_service import DepartureContext
except ImportError:
    from backend.services.context_service import DepartureContext

logger = logging.getLogger("microdelay.services.departure_optimizer")


@dataclass
class DepartureTimelineSegment:
    """A distinct temporal phase in the departure timeline."""
    phase: str  # "preparation", "buffer", "transit", "arrival"
    start_time: str
    end_time: str
    duration_min: float
    description: str


@dataclass
class DepartureOptimizationResult:
    """Complete optimized departure decision and actionable schedule."""
    recommended_prep_start_time: str
    recommended_departure_time: str
    target_arrival_time: str
    event_start_time: str
    total_lead_time_min: float
    prep_duration_min: float
    travel_duration_min: float
    safety_buffer_min: float
    timeline: List[DepartureTimelineSegment] = field(default_factory=list)
    delay_risk_category: str = "low"
    optimization_notes: List[str] = field(default_factory=list)
    context_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class DepartureOptimizer:
    """Optimizes departure timelines by integrating contextual observations and behavioral profiles."""

    def optimize_departure(
        self,
        context: DepartureContext
    ) -> DepartureOptimizationResult:
        """Compute the recommended departure and prep start times from a DepartureContext."""
        # Parse target start time
        try:
            event_start_dt = datetime.fromisoformat(context.event_start_time)
            if event_start_dt.tzinfo is None:
                event_start_dt = event_start_dt.replace(tzinfo=timezone.utc)
        except Exception:
            event_start_dt = datetime.now(timezone.utc) + timedelta(hours=2)

        notes: List[str] = []

        # 1. Effective Travel Duration
        if context.live_travel_duration is not None and context.live_travel_duration > 0:
            effective_travel = context.live_travel_duration
            if context.traffic_conditions:
                congestion = context.traffic_conditions.get("congestion_level", "low")
                if congestion in ["moderate", "heavy", "severe"]:
                    notes.append(f"Adjusted for {congestion} traffic conditions along your route.")
        else:
            effective_travel = context.estimated_travel_duration
            notes.append("Using estimated travel duration (live traffic unavailable or route direct).")

        # 2. Personalized Prep Duration
        effective_prep = context.personalized_prep_duration or context.estimated_prep_duration
        if context.predicted_delay_min and context.predicted_delay_min > 0:
            notes.append(
                f"Incorporated +{round(context.predicted_delay_min, 1)}m behavioral prep buffer based on your profile history."
            )

        # 3. Weather Buffer
        weather_extra = 0.0
        if context.weather_conditions:
            weather_extra = float(context.weather_conditions.get("recommended_buffer_min", 0.0))
            if weather_extra > 0:
                notes.append(f"Added +{round(weather_extra, 1)}m weather safety buffer.")

        # 4. Total Safety Buffer
        effective_buffer = context.safety_buffer

        # 5. Calculate Exact Timeline
        # Target arrival is either right at event start or slightly early based on importance
        early_arrival_padding = 2.0 if context.event_importance in ["high", "critical"] else 0.0
        target_arrival_dt = event_start_dt - timedelta(minutes=early_arrival_padding)

        # Departure = Target Arrival - Travel - Buffer
        departure_dt = target_arrival_dt - timedelta(minutes=effective_travel + effective_buffer)

        # Prep Start = Departure - Prep
        prep_start_dt = departure_dt - timedelta(minutes=effective_prep)

        total_lead_time = effective_prep + effective_buffer + effective_travel + early_arrival_padding

        # Build detailed timeline segments
        timeline = [
            DepartureTimelineSegment(
                phase="preparation",
                start_time=prep_start_dt.isoformat(),
                end_time=departure_dt.isoformat(),
                duration_min=round(effective_prep, 1),
                description=f"Begin departure preparation ({round(effective_prep, 1)} min allocated)"
            ),
            DepartureTimelineSegment(
                phase="safety_buffer",
                start_time=departure_dt.isoformat(),
                end_time=(departure_dt + timedelta(minutes=effective_buffer)).isoformat(),
                duration_min=round(effective_buffer, 1),
                description=f"Safety buffer for behavioral and environmental variance ({round(effective_buffer, 1)} min)"
            ),
            DepartureTimelineSegment(
                phase="transit",
                start_time=(departure_dt + timedelta(minutes=effective_buffer)).isoformat(),
                end_time=target_arrival_dt.isoformat(),
                duration_min=round(effective_travel, 1),
                description=f"Active travel to {context.destination or 'destination'} ({round(effective_travel, 1)} min)"
            ),
            DepartureTimelineSegment(
                phase="arrival",
                start_time=target_arrival_dt.isoformat(),
                end_time=event_start_dt.isoformat(),
                duration_min=early_arrival_padding,
                description="Arrived and settling in prior to event start"
            )
        ]

        if context.is_destination_required:
            notes.append("Warning: Missing destination. Recommended times reflect estimated route template.")

        return DepartureOptimizationResult(
            recommended_prep_start_time=prep_start_dt.isoformat(),
            recommended_departure_time=departure_dt.isoformat(),
            target_arrival_time=target_arrival_dt.isoformat(),
            event_start_time=event_start_dt.isoformat(),
            total_lead_time_min=round(total_lead_time, 1),
            prep_duration_min=round(effective_prep, 1),
            travel_duration_min=round(effective_travel, 1),
            safety_buffer_min=round(effective_buffer, 1),
            timeline=timeline,
            delay_risk_category=context.behavioral_risk_category or "low",
            optimization_notes=notes,
            context_snapshot=context.to_dict()
        )
