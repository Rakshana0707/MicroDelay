"""Journey Tracking Service for managing active departure lifecycles.

Tracks transitions from preparation start to final arrival, providing real-time
progress monitoring and closing the feedback loop into Phase 1 & 2 model retraining.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
import logging
from sqlalchemy.orm import Session

try:
    from database import DepartureLog, Event, DataProvenance, DataSource, ObservationStatus
    from services.context_service import DepartureContext
except ImportError:
    from backend.database import DepartureLog, Event, DataProvenance, DataSource, ObservationStatus
    from backend.services.context_service import DepartureContext

logger = logging.getLogger("microdelay.services.journey")


class JourneyPhase(str, Enum):
    """Lifecycle phases of an active user journey."""
    SCHEDULED = "scheduled"
    PREP_STARTED = "prep_started"
    DEPARTED = "departed"
    IN_TRANSIT = "in_transit"
    ARRIVED = "arrived"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


VALID_PHASE_TRANSITIONS = {
    JourneyPhase.SCHEDULED: [JourneyPhase.PREP_STARTED, JourneyPhase.CANCELLED],
    JourneyPhase.PREP_STARTED: [JourneyPhase.DEPARTED, JourneyPhase.CANCELLED],
    JourneyPhase.DEPARTED: [JourneyPhase.IN_TRANSIT, JourneyPhase.ARRIVED, JourneyPhase.CANCELLED],
    JourneyPhase.IN_TRANSIT: [JourneyPhase.ARRIVED, JourneyPhase.CANCELLED],
    JourneyPhase.ARRIVED: [JourneyPhase.COMPLETED],
    JourneyPhase.COMPLETED: [],
    JourneyPhase.CANCELLED: []
}


class InvalidJourneyTransitionError(Exception):
    """Raised when attempting an illegal journey phase transition."""
    pass


@dataclass
class JourneyRecord:
    """In-memory or persisted representation of an active departure journey."""
    journey_id: str
    user_id: int
    event_id: Optional[str]
    event_name: str
    phase: JourneyPhase = JourneyPhase.SCHEDULED
    planned_prep_start: Optional[str] = None
    planned_departure: Optional[str] = None
    planned_arrival: Optional[str] = None
    actual_prep_start: Optional[str] = None
    actual_departure: Optional[str] = None
    actual_arrival: Optional[str] = None
    estimated_prep_duration_min: float = 30.0
    estimated_travel_duration_min: float = 20.0
    actual_prep_duration_min: Optional[float] = None
    actual_travel_duration_min: Optional[float] = None
    total_delay_min: float = 0.0
    bag_prepared: bool = False
    context_snapshot: Optional[Dict[str, Any]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.phase, JourneyPhase):
            data["phase"] = self.phase.value
        return data


class JourneyService:
    """Manages journey state machine and logs completed trips to the database."""

    def __init__(self):
        self._journeys: Dict[str, JourneyRecord] = {}

    def create_journey(
        self,
        user_id: int,
        context: DepartureContext,
        recommended_prep_start: str,
        recommended_departure: str,
        target_arrival: str
    ) -> JourneyRecord:
        """Initialize and track a scheduled departure journey."""
        now = datetime.now(timezone.utc)
        j_id = f"journey_{user_id}_{int(now.timestamp())}"

        journey = JourneyRecord(
            journey_id=j_id,
            user_id=user_id,
            event_id=context.event_id,
            event_name=context.event_name,
            phase=JourneyPhase.SCHEDULED,
            planned_prep_start=recommended_prep_start,
            planned_departure=recommended_departure,
            planned_arrival=target_arrival,
            estimated_prep_duration_min=context.personalized_prep_duration or context.estimated_prep_duration,
            estimated_travel_duration_min=context.live_travel_duration or context.estimated_travel_duration,
            context_snapshot=context.to_dict()
        )
        self._journeys[j_id] = journey
        logger.info("Created journey id='%s' for user=%s, event='%s'", j_id, user_id, context.event_name)
        return journey

    def get_journey(self, journey_id: str) -> Optional[JourneyRecord]:
        """Fetch journey by id."""
        return self._journeys.get(journey_id)

    def transition_phase(
        self,
        journey_id: str,
        target_phase: JourneyPhase,
        timestamp: Optional[datetime] = None,
        db: Optional[Session] = None
    ) -> JourneyRecord:
        """Advance journey phase with state machine validation."""
        journey = self._journeys.get(journey_id)
        if not journey:
            raise ValueError(f"Journey '{journey_id}' not found.")

        current = journey.phase
        allowed = VALID_PHASE_TRANSITIONS.get(current, [])

        if target_phase not in allowed:
            raise InvalidJourneyTransitionError(
                f"Cannot transition journey from '{current.value}' to '{target_phase.value}'. "
                f"Allowed transitions: {[p.value for p in allowed]}"
            )

        ts = timestamp or datetime.now(timezone.utc)
        ts_iso = ts.isoformat()
        journey.phase = target_phase
        journey.updated_at = ts_iso

        if target_phase == JourneyPhase.PREP_STARTED:
            journey.actual_prep_start = ts_iso

        elif target_phase == JourneyPhase.DEPARTED:
            journey.actual_departure = ts_iso
            if journey.actual_prep_start:
                try:
                    prep_start = datetime.fromisoformat(journey.actual_prep_start)
                    journey.actual_prep_duration_min = round((ts - prep_start).total_seconds() / 60.0, 1)
                except Exception:
                    pass

        elif target_phase == JourneyPhase.ARRIVED:
            journey.actual_arrival = ts_iso
            if journey.actual_departure:
                try:
                    dep = datetime.fromisoformat(journey.actual_departure)
                    journey.actual_travel_duration_min = round((ts - dep).total_seconds() / 60.0, 1)
                except Exception:
                    pass

            # Calculate total delay relative to planned arrival
            if journey.planned_arrival:
                try:
                    planned_arr = datetime.fromisoformat(journey.planned_arrival)
                    journey.total_delay_min = round((ts - planned_arr).total_seconds() / 60.0, 1)
                except Exception:
                    pass

        elif target_phase == JourneyPhase.COMPLETED:
            # Commit journey to MicroDelay departure_logs to close feedback loop
            if db is not None:
                self._persist_to_departure_log(journey, db)

        return journey

    def _persist_to_departure_log(self, journey: JourneyRecord, db: Session) -> None:
        """Record completed real journey into `departure_logs` for ML retraining."""
        try:
            today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            day_of_week = datetime.now(timezone.utc).strftime("%A")

            # Extract prep & travel actuals
            actual_prep = journey.actual_prep_duration_min or journey.estimated_prep_duration_min
            actual_travel = journey.actual_travel_duration_min or journey.estimated_travel_duration_min

            log_entry = DepartureLog(
                user_id=journey.user_id,
                event_id=int(journey.event_id) if (journey.event_id and str(journey.event_id).isdigit()) else None,
                event_name=journey.event_name,
                date=today_date,
                day_of_week=day_of_week,
                planned_prep_start=journey.planned_prep_start or "08:00:00",
                planned_departure=journey.planned_departure or "08:30:00",
                estimated_prep_duration_min=journey.estimated_prep_duration_min,
                estimated_travel_duration_min=journey.estimated_travel_duration_min,
                actual_prep_start=journey.actual_prep_start or journey.planned_prep_start or "08:00:00",
                actual_departure=journey.actual_departure or journey.planned_departure or "08:30:00",
                actual_arrival=journey.actual_arrival or journey.planned_arrival or "08:50:00",
                actual_prep_duration_min=actual_prep,
                actual_travel_duration_min=actual_travel,
                total_delay_min=journey.total_delay_min,
                bag_prepared=journey.bag_prepared,
                provenance=DataProvenance.REAL.value,
                data_source="journey_service_live",
                is_synthetic=False,
                observation_status=ObservationStatus.VALID.value
            )
            db.add(log_entry)
            db.commit()
            logger.info("Persisted completed journey '%s' to database logs", journey.journey_id)
        except Exception as e:
            db.rollback()
            logger.error("Failed to persist completed journey '%s' to DB: %s", journey.journey_id, e)
