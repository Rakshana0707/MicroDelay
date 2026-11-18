"""Travel and Traffic Service coordinating route estimation and traffic degradation."""

import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

try:
    from integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationError,
        IntegrationTimeoutError, IntegrationDataError
    )
    from integrations.registry import IntegrationRegistry
    from integrations.travel.base import TravelIntegrationBase, TravelEstimate, TravelMode
    from integrations.traffic.base import (
        TrafficIntegrationBase, TrafficObservation, TrafficCongestionLevel
    )
    from database import DepartureLog, Event
except ImportError:
    from backend.integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationError,
        IntegrationTimeoutError, IntegrationDataError
    )
    from backend.integrations.registry import IntegrationRegistry
    from backend.integrations.travel.base import TravelIntegrationBase, TravelEstimate, TravelMode
    from backend.integrations.traffic.base import (
        TrafficIntegrationBase, TrafficObservation, TrafficCongestionLevel
    )
    from backend.database import DepartureLog, Event

logger = logging.getLogger("microdelay.services.travel")


class TravelService:
    """Manages travel-time estimation, live traffic adjustments, and graceful degradation."""

    def __init__(self, registry: Optional[IntegrationRegistry] = None):
        self.registry = registry or IntegrationRegistry.get_instance()

    def get_travel_context(
        self,
        user_id: int,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        departure_time: Optional[datetime] = None,
        event_id: Optional[str] = None,
        mode: TravelMode = TravelMode.DRIVING,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Compute estimated and live travel duration, handling missing destinations and outages.
        
        Graceful degradation levels:
        1. Live Route Travel Estimation + Live Traffic Conditions
        2. Live Route Travel Estimation (Traffic API unavailable -> multiplier 1.0)
        3. Historical Personal Travel Average (from user's `DepartureLog` records)
        4. Static Event Routine Estimate or Sane Fallback (20.0 minutes)
        """
        now = datetime.now(timezone.utc)
        dep_time = departure_time or now
        clean_origin = origin.strip() if origin else "Current Location"
        clean_dest = destination.strip() if destination else None

        # Check for missing destination
        if not clean_dest:
            logger.info("Destination missing for user_id=%s; requesting user input and falling back to routine estimate", user_id)
            fallback_duration = self._get_historical_or_routine_travel(user_id, event_id, db)
            meta = ExternalObservationMetadata(
                source="missing_destination_fallback",
                applicable_id=str(event_id or "unknown"),
                observation_type=ObservationType.ESTIMATED,
                confidence=0.5,
                is_fallback=True,
                details={"reason": "Destination not provided; user input recommended"}
            )
            return {
                "origin": clean_origin,
                "destination": None,
                "is_destination_required": True,
                "estimated_travel_duration": fallback_duration,
                "live_travel_duration": None,
                "traffic_conditions": None,
                "source": "missing_destination_fallback",
                "is_fallback": True,
                "degradation_reason": "destination_missing",
                "travel_metadata": meta,
                "traffic_metadata": None
            }

        # 1. Attempt Live Travel Estimation
        travel_integration: Optional[TravelIntegrationBase] = self.registry.get("travel")
        live_travel_est: Optional[TravelEstimate] = None
        travel_meta: Optional[ExternalObservationMetadata] = None

        if travel_integration and travel_integration.is_enabled():
            try:
                live_travel_est = travel_integration.estimate_travel(
                    origin=clean_origin,
                    destination=clean_dest,
                    departure_time=dep_time,
                    mode=mode,
                    applicable_id=str(event_id or "")
                )
                # Sanity validation
                if live_travel_est.duration_min <= 0.0 or live_travel_est.duration_min > 720.0:
                    raise IntegrationDataError(
                        f"Unrealistic travel duration received: {live_travel_est.duration_min} min",
                        service=travel_integration.provider_name
                    )
                travel_meta = live_travel_est.metadata
            except (IntegrationTimeoutError, IntegrationDataError, IntegrationError) as e:
                logger.warning("Travel integration failed for user_id=%s: %s. Using historical/routine fallback.", user_id, e)
                live_travel_est = None

        # 2. Attempt Live Traffic Estimation
        traffic_integration: Optional[TrafficIntegrationBase] = self.registry.get("traffic")
        traffic_obs: Optional[TrafficObservation] = None
        traffic_meta: Optional[ExternalObservationMetadata] = None

        if traffic_integration and traffic_integration.is_enabled():
            try:
                traffic_obs = traffic_integration.get_traffic_conditions(
                    origin=clean_origin,
                    destination=clean_dest,
                    departure_time=dep_time,
                    applicable_id=str(event_id or "")
                )
                traffic_meta = traffic_obs.metadata
            except (IntegrationTimeoutError, IntegrationError) as e:
                logger.warning("Traffic integration unavailable for user_id=%s: %s. Omitting traffic adjustments.", user_id, e)
                traffic_obs = None

        # 3. Resolve Durations and Apply Traffic Multiplier
        if live_travel_est is not None:
            base_duration = live_travel_est.duration_min
            if traffic_obs is not None:
                # Combine live travel with traffic congestion delay
                live_duration = base_duration * max(1.0, traffic_obs.delay_multiplier) + traffic_obs.delay_minutes
            else:
                live_duration = base_duration
                traffic_obs = TrafficObservation(
                    origin=clean_origin,
                    destination=clean_dest,
                    congestion_level=TrafficCongestionLevel.UNKNOWN,
                    delay_multiplier=1.0,
                    delay_minutes=0.0,
                    metadata=ExternalObservationMetadata(
                        source="traffic_unavailable_fallback",
                        observation_type=ObservationType.ESTIMATED,
                        is_fallback=True
                    )
                )

            return {
                "origin": clean_origin,
                "destination": clean_dest,
                "is_destination_required": False,
                "estimated_travel_duration": base_duration,
                "live_travel_duration": round(live_duration, 1),
                "distance_km": live_travel_est.distance_km,
                "traffic_conditions": traffic_obs.to_dict(),
                "source": f"live_travel:{travel_integration.provider_name}",
                "is_fallback": False,
                "travel_metadata": travel_meta,
                "traffic_metadata": traffic_meta
            }

        # 4. Fallback Path: Historical Personal Travel or Event Routine
        fallback_travel_min = self._get_historical_or_routine_travel(user_id, event_id, db)
        meta = ExternalObservationMetadata(
            source="historical_personal_travel",
            applicable_id=str(event_id or "unknown"),
            observation_type=ObservationType.HISTORICAL,
            confidence=0.8,
            is_fallback=True,
            details={"note": "External travel API unavailable; using personal historical travel duration"}
        )

        return {
            "origin": clean_origin,
            "destination": clean_dest,
            "is_destination_required": False,
            "estimated_travel_duration": fallback_travel_min,
            "live_travel_duration": fallback_travel_min,  # Historical best estimate used as live fallback
            "distance_km": None,
            "traffic_conditions": None,
            "source": "historical_personal_travel",
            "is_fallback": True,
            "degradation_reason": "travel_api_unavailable",
            "travel_metadata": meta,
            "traffic_metadata": None
        }

    def _get_historical_or_routine_travel(
        self,
        user_id: int,
        event_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> float:
        """Fetch historical actual travel mean from `departure_logs` or routine template."""
        if db is not None:
            try:
                # 1. User event-specific historical mean
                if event_id and str(event_id).isdigit():
                    avg_travel = db.query(
                        func.avg(DepartureLog.actual_travel_duration_min)
                    ).filter(
                        DepartureLog.user_id == user_id,
                        DepartureLog.event_id == int(event_id),
                        DepartureLog.actual_travel_duration_min > 0
                    ).scalar()
                    if avg_travel is not None and avg_travel > 0:
                        return float(round(avg_travel, 1))

                    # 2. Check Event routine template
                    ev = db.query(Event).filter(Event.id == int(event_id)).first()
                    if ev and ev.estimated_travel_duration > 0:
                        return float(ev.estimated_travel_duration)

                # 3. Overall user historical mean travel
                user_avg = db.query(
                    func.avg(DepartureLog.actual_travel_duration_min)
                ).filter(
                    DepartureLog.user_id == user_id,
                    DepartureLog.actual_travel_duration_min > 0
                ).scalar()
                if user_avg is not None and user_avg > 0:
                    return float(round(user_avg, 1))

            except Exception as e:
                logger.warning("Failed to calculate historical travel for user_id=%s: %s", user_id, e)

        return 20.0  # Heuristic fallback
