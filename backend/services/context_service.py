"""Departure Context Aggregation Service for MicroDelay.

Aggregates personal behavioral intelligence, calendar schedules, route travel estimates,
traffic conditions, weather observations, and user constraints into a unified,
traceable DepartureContext object with strict graceful degradation.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from sqlalchemy.orm import Session

try:
    from integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationStatus
    )
    from integrations.registry import IntegrationRegistry
    from services.calendar_service import CalendarService
    from services.travel_service import TravelService
    from services.weather_service import WeatherService
    from ml.adaptive.context import PredictionContext
    from ml.adaptive.orchestrator import AdaptiveIntelligenceOrchestrator
except ImportError:
    from backend.integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationStatus
    )
    from backend.integrations.registry import IntegrationRegistry
    from backend.services.calendar_service import CalendarService
    from backend.services.travel_service import TravelService
    from backend.services.weather_service import WeatherService
    from backend.ml.adaptive.context import PredictionContext
    from backend.ml.adaptive.orchestrator import AdaptiveIntelligenceOrchestrator

logger = logging.getLogger("microdelay.services.context")


@dataclass
class DepartureContextValidationResult:
    """Validation report for DepartureContext."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DepartureContext:
    """Unified Context-Aware Departure Intelligence object.
    
    Contains all situational, environmental, scheduling, and behavioral
    attributes necessary for optimal departure recommendations.
    """
    user_id: int
    event_id: Optional[str] = None
    event_name: str = "Scheduled Departure"
    event_start_time: str = field(
        default_factory=lambda: (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    )
    event_location: Optional[str] = None
    current_location: Optional[str] = "Current Location"
    destination: Optional[str] = None
    event_importance: str = "medium"  # low, medium, high, critical
    estimated_prep_duration: float = 30.0  # base routine prep duration (minutes)
    personalized_prep_duration: float = 30.0  # Phase 2 ML behavioral prep duration (minutes)
    estimated_travel_duration: float = 20.0  # static/route travel estimate (minutes)
    live_travel_duration: Optional[float] = None  # dynamic/traffic-adjusted travel duration (minutes)
    weather_conditions: Optional[Dict[str, Any]] = None
    traffic_conditions: Optional[Dict[str, Any]] = None
    safety_buffer: float = 10.0  # minutes
    context_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    degradation_status: Dict[str, str] = field(default_factory=dict)
    is_destination_required: bool = False
    missing_fields: List[str] = field(default_factory=list)
    behavioral_risk_category: Optional[str] = None
    predicted_delay_min: Optional[float] = None
    ttl_seconds: int = 300

    def is_expired(self, at_time: Optional[datetime] = None) -> bool:
        """Check if context snapshot has exceeded its TTL."""
        check_time = at_time or datetime.now(timezone.utc)
        try:
            created_dt = datetime.fromisoformat(self.context_timestamp)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            if check_time.tzinfo is None:
                check_time = check_time.replace(tzinfo=timezone.utc)
            return (check_time - created_dt).total_seconds() > self.ttl_seconds
        except (ValueError, TypeError):
            return False

    def validate(self) -> DepartureContextValidationResult:
        """Validate context consistency and detect missing fields."""
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(self.user_id, int) or self.user_id <= 0:
            errors.append(f"Invalid user_id '{self.user_id}': must be a positive integer.")

        if not self.destination and self.is_destination_required:
            warnings.append("Destination is required for dynamic travel-time calculation.")

        if self.estimated_prep_duration < 0.0 or self.estimated_prep_duration > 360.0:
            errors.append(f"estimated_prep_duration {self.estimated_prep_duration} is out of sane bounds [0, 360].")

        if self.personalized_prep_duration < 0.0 or self.personalized_prep_duration > 360.0:
            errors.append(f"personalized_prep_duration {self.personalized_prep_duration} is out of sane bounds [0, 360].")

        if self.estimated_travel_duration < 0.0 or self.estimated_travel_duration > 720.0:
            errors.append(f"estimated_travel_duration {self.estimated_travel_duration} is out of sane bounds [0, 720].")

        if self.live_travel_duration is not None and (self.live_travel_duration < 0.0 or self.live_travel_duration > 720.0):
            errors.append(f"live_travel_duration {self.live_travel_duration} is out of sane bounds [0, 720].")

        if self.safety_buffer < 0.0 or self.safety_buffer > 180.0:
            errors.append(f"safety_buffer {self.safety_buffer} is out of sane bounds [0, 180].")

        return DepartureContextValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DepartureContext":
        """Reconstruct DepartureContext safely from dictionary."""
        return cls(
            user_id=int(data.get("user_id", 0)),
            event_id=str(data["event_id"]) if data.get("event_id") is not None else None,
            event_name=str(data.get("event_name", "Scheduled Departure")),
            event_start_time=str(data.get("event_start_time", datetime.now(timezone.utc).isoformat())),
            event_location=data.get("event_location"),
            current_location=data.get("current_location", "Current Location"),
            destination=data.get("destination"),
            event_importance=str(data.get("event_importance", "medium")),
            estimated_prep_duration=float(data.get("estimated_prep_duration", 30.0)),
            personalized_prep_duration=float(data.get("personalized_prep_duration", 30.0)),
            estimated_travel_duration=float(data.get("estimated_travel_duration", 20.0)),
            live_travel_duration=float(data["live_travel_duration"]) if data.get("live_travel_duration") is not None else None,
            weather_conditions=data.get("weather_conditions"),
            traffic_conditions=data.get("traffic_conditions"),
            safety_buffer=float(data.get("safety_buffer", 10.0)),
            context_timestamp=str(data.get("context_timestamp", datetime.now(timezone.utc).isoformat())),
            source_metadata=data.get("source_metadata", {}),
            degradation_status=data.get("degradation_status", {}),
            is_destination_required=bool(data.get("is_destination_required", False)),
            missing_fields=data.get("missing_fields", []),
            behavioral_risk_category=data.get("behavioral_risk_category"),
            predicted_delay_min=float(data["predicted_delay_min"]) if data.get("predicted_delay_min") is not None else None,
            ttl_seconds=int(data.get("ttl_seconds", 300))
        )


class ContextService:
    """Central service orchestrating context aggregation and graceful degradation."""

    def __init__(
        self,
        calendar_service: Optional[CalendarService] = None,
        travel_service: Optional[TravelService] = None,
        weather_service: Optional[WeatherService] = None,
        orchestrator: Optional[AdaptiveIntelligenceOrchestrator] = None,
        registry: Optional[IntegrationRegistry] = None
    ):
        self.registry = registry or IntegrationRegistry.get_instance()
        self.calendar_service = calendar_service or CalendarService(self.registry)
        self.travel_service = travel_service or TravelService(self.registry)
        self.weather_service = weather_service or WeatherService(self.registry)
        self.orchestrator = orchestrator or AdaptiveIntelligenceOrchestrator()

    def aggregate_context(
        self,
        user_id: int,
        event_id: Optional[str] = None,
        event_name: Optional[str] = None,
        current_location: Optional[str] = "Current Location",
        destination: Optional[str] = None,
        target_time: Optional[datetime] = None,
        user_constraints: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None
    ) -> DepartureContext:
        """Aggregate behavioral predictions, calendar, travel, traffic, and weather into DepartureContext.
        
        Graceful Degradation Guarantees:
        - If calendar unavailable -> use database routine template or default.
        - If traffic API unavailable -> use historical personal travel estimates.
        - If weather API unavailable -> omit weather adjustments (buffer=0).
        - If destination missing -> flag `is_destination_required = True`, request user input.
        - If insufficient behavioral history -> use validated baseline / global model.
        """
        now = datetime.now(timezone.utc)
        constraints = user_constraints or {}
        source_metadata: Dict[str, Dict[str, Any]] = {}
        degradation_status: Dict[str, str] = {}
        missing_fields: List[str] = []

        # 1. Resolve Calendar / Event
        cal_ctx = self.calendar_service.get_event_context(
            user_id=user_id,
            event_id=event_id,
            event_name=event_name,
            target_time=target_time,
            db=db
        )
        resolved_event_id = cal_ctx.get("event_id")
        resolved_event_name = cal_ctx.get("event_name", "Scheduled Departure")
        start_time_dt = cal_ctx.get("start_time") or (now + timedelta(hours=2))
        event_importance = cal_ctx.get("importance", "medium")
        base_prep = float(cal_ctx.get("estimated_prep_duration", 30.0))
        base_travel = float(cal_ctx.get("estimated_travel_duration", 20.0))

        if cal_ctx.get("is_fallback"):
            degradation_status["calendar"] = cal_ctx.get("source", "fallback_routine")
        else:
            degradation_status["calendar"] = "healthy"

        if cal_ctx.get("metadata"):
            source_metadata["calendar"] = cal_ctx["metadata"].to_dict()

        # 2. Resolve Destination & Location
        effective_dest = destination or cal_ctx.get("destination") or cal_ctx.get("location")
        if not effective_dest:
            missing_fields.append("destination")

        # 3. Resolve Travel & Traffic
        travel_ctx = self.travel_service.get_travel_context(
            user_id=user_id,
            origin=current_location,
            destination=effective_dest,
            departure_time=start_time_dt - timedelta(minutes=base_travel),
            event_id=resolved_event_id,
            db=db
        )
        est_travel = float(travel_ctx.get("estimated_travel_duration", base_travel))
        live_travel = travel_ctx.get("live_travel_duration")
        traffic_conditions = travel_ctx.get("traffic_conditions")
        is_dest_required = travel_ctx.get("is_destination_required", False)

        if travel_ctx.get("is_fallback"):
            degradation_status["travel"] = travel_ctx.get("degradation_reason", "fallback")
        else:
            degradation_status["travel"] = "healthy"

        if travel_ctx.get("travel_metadata"):
            source_metadata["travel"] = travel_ctx["travel_metadata"].to_dict()
        if travel_ctx.get("traffic_metadata"):
            source_metadata["traffic"] = travel_ctx["traffic_metadata"].to_dict()

        # 4. Resolve Weather
        weather_location = effective_dest or current_location
        weather_ctx = self.weather_service.get_weather_context(
            location=weather_location,
            target_time=start_time_dt,
            event_id=resolved_event_id
        )
        weather_conditions = weather_ctx.get("weather_conditions")
        weather_buffer = float(weather_ctx.get("recommended_buffer_min", 0.0))

        if weather_ctx.get("is_omitted"):
            degradation_status["weather"] = weather_ctx.get("source", "omitted")
        else:
            degradation_status["weather"] = "healthy"

        if weather_ctx.get("metadata"):
            source_metadata["weather"] = weather_ctx["metadata"].to_dict()

        # 5. Integrate Phase 2 Behavioral Intelligence
        day_of_week = start_time_dt.strftime("%A")
        pred_context = PredictionContext(
            user_id=user_id,
            event_name=resolved_event_name,
            day_of_week=day_of_week,
            event_importance=event_importance,
            estimated_prep_duration=base_prep,
            estimated_travel_duration=est_travel,
            bag_prepared=bool(constraints.get("bag_prepared", False)),
            event_id=int(resolved_event_id) if (resolved_event_id and str(resolved_event_id).isdigit()) else None
        )

        personalized_prep = base_prep
        predicted_delay = 0.0
        risk_cat = "low"
        try:
            intel_output = self.orchestrator.predict(pred_context, db=db)
            # Support both schema conventions
            predicted_delay = (
                getattr(intel_output, "predicted_delay_minutes", None)
                or getattr(intel_output, "predicted_delay_min", None)
                or 0.0
            )
            raw_risk = getattr(intel_output, "risk_category", None)
            if not raw_risk and hasattr(intel_output, "risk_classification"):
                rc = getattr(intel_output, "risk_classification")
                raw_risk = getattr(rc, "risk_category", "low")
            if hasattr(raw_risk, "value"):
                risk_cat = raw_risk.value
            elif raw_risk:
                risk_cat = str(raw_risk).lower()
            else:
                risk_cat = "low"

            # Personalization adjusts prep duration based on behavioral delay tendencies
            if predicted_delay > 0:
                personalized_prep = round(base_prep + (predicted_delay * 0.7), 1)

            # Confidence resolution
            conf_score = 0.85
            conf_status = "moderate"
            if hasattr(intel_output, "confidence_status") and isinstance(intel_output.confidence_status, dict):
                conf_score = intel_output.confidence_status.get("score", 0.85)
                conf_status = intel_output.confidence_status.get("status", "moderate")
            elif hasattr(intel_output, "confidence"):
                conf_score = getattr(intel_output.confidence, "score", 0.85)
                c_status = getattr(intel_output.confidence, "status", "moderate")
                conf_status = getattr(c_status, "value", str(c_status))

            # Provenance & Data Quality resolution
            prov = "real"
            is_suff = True
            if hasattr(intel_output, "data_quality_status") and isinstance(intel_output.data_quality_status, dict):
                prov = intel_output.data_quality_status.get("provenance", "real")
                is_suff = intel_output.data_quality_status.get("is_sufficient", True)
            elif hasattr(intel_output, "data_quality"):
                prov = getattr(intel_output.data_quality, "provenance", "real")
                is_suff = getattr(intel_output.data_quality, "is_sufficient", True)

            source_metadata["behavioral"] = {
                "source": "adaptive_behavioral_intelligence_phase_2",
                "retrieval_timestamp": now.isoformat(),
                "observation_type": ObservationType.ESTIMATED.value,
                "confidence": conf_score,
                "is_fallback": (prov != "real"),
                "details": {
                    "risk_category": risk_cat,
                    "predicted_delay_min": predicted_delay,
                    "confidence_status": conf_status,
                    "data_provenance": prov
                }
            }
            if not is_suff:
                degradation_status["behavioral"] = "baseline_fallback"
            else:
                degradation_status["behavioral"] = "personalized"
        except Exception as e:
            logger.warning("Phase 2 behavioral intelligence orchestration failed: %s. Using base routine.", e)
            degradation_status["behavioral"] = "error_fallback"
            source_metadata["behavioral"] = {
                "source": "heuristic_fallback",
                "retrieval_timestamp": now.isoformat(),
                "observation_type": ObservationType.ESTIMATED.value,
                "confidence": 0.5,
                "is_fallback": True,
                "details": {"error": str(e)}
            }

        # 6. Safety Buffer Calculation
        # Combines behavioral risk category, importance, weather impact, and user constraints
        importance_buffers = {"low": 3.0, "medium": 7.0, "high": 12.0, "critical": 20.0}
        risk_multipliers = {"low": 1.0, "moderate": 1.3, "high": 1.6, "severe": 2.0}

        base_buffer = importance_buffers.get(event_importance.lower(), 7.0)
        multiplier = risk_multipliers.get(risk_cat.lower(), 1.0)
        computed_buffer = (base_buffer * multiplier) + weather_buffer

        user_min_buffer = float(constraints.get("min_buffer_min", 5.0))
        user_max_buffer = float(constraints.get("max_buffer_min", 45.0))
        final_safety_buffer = round(max(user_min_buffer, min(computed_buffer, user_max_buffer)), 1)

        # 7. Construct DepartureContext
        start_time_iso = (
            start_time_dt.isoformat()
            if isinstance(start_time_dt, datetime)
            else str(start_time_dt)
        )

        return DepartureContext(
            user_id=user_id,
            event_id=str(resolved_event_id) if resolved_event_id else None,
            event_name=resolved_event_name,
            event_start_time=start_time_iso,
            event_location=effective_dest,
            current_location=current_location,
            destination=effective_dest,
            event_importance=event_importance,
            estimated_prep_duration=base_prep,
            personalized_prep_duration=personalized_prep,
            estimated_travel_duration=est_travel,
            live_travel_duration=live_travel,
            weather_conditions=weather_conditions,
            traffic_conditions=traffic_conditions,
            safety_buffer=final_safety_buffer,
            context_timestamp=now.isoformat(),
            source_metadata=source_metadata,
            degradation_status=degradation_status,
            is_destination_required=is_dest_required,
            missing_fields=missing_fields,
            behavioral_risk_category=risk_cat,
            predicted_delay_min=predicted_delay,
            ttl_seconds=300
        )
