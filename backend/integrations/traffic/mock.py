"""Configurable Mock Traffic Integration for testing and development."""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

try:
    from integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationError
    )
    from integrations.traffic.base import (
        TrafficIntegrationBase, TrafficObservation, TrafficCongestionLevel
    )
except ImportError:
    from backend.integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationError
    )
    from backend.integrations.traffic.base import (
        TrafficIntegrationBase, TrafficObservation, TrafficCongestionLevel
    )


class MockTrafficIntegration(TrafficIntegrationBase):
    """Simulates real-time traffic conditions, incidents, and congestion delays."""

    def __init__(
        self,
        provider_name: str = "mock_traffic",
        enabled: bool = True,
        default_congestion: TrafficCongestionLevel = TrafficCongestionLevel.LOW,
        simulate_timeout: bool = False,
        simulate_error: bool = False
    ):
        super().__init__(
            provider_name=provider_name,
            enabled=enabled,
            timeout_seconds=3.0,
            cache_ttl_seconds=180
        )
        self.default_congestion = default_congestion
        self.simulate_timeout = simulate_timeout
        self.simulate_error = simulate_error
        self.route_conditions: Dict[str, TrafficObservation] = {}

    def set_route_condition(
        self,
        origin: str,
        destination: str,
        congestion: TrafficCongestionLevel,
        delay_multiplier: float,
        delay_minutes: float = 0.0,
        jam_factor: float = 0.0,
        incidents: Optional[List[str]] = None
    ) -> None:
        """Register specific conditions for a route."""
        key = f"{origin.strip().lower()}->{destination.strip().lower()}"
        meta = self.create_metadata(
            observation_type=ObservationType.LIVE,
            details={"configured": True}
        )
        self.route_conditions[key] = TrafficObservation(
            origin=origin,
            destination=destination,
            congestion_level=congestion,
            delay_multiplier=delay_multiplier,
            delay_minutes=delay_minutes,
            jam_factor=jam_factor,
            incident_reports=incidents or [],
            metadata=meta
        )

    def get_traffic_conditions(
        self,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        applicable_id: Optional[str] = None
    ) -> TrafficObservation:
        if not self.enabled:
            # When disabled, callers degrade gracefully
            meta = self.create_metadata(
                applicable_id=applicable_id,
                observation_type=ObservationType.ESTIMATED,
                is_fallback=True
            )
            return TrafficObservation(
                origin=origin,
                destination=destination,
                congestion_level=TrafficCongestionLevel.UNKNOWN,
                delay_multiplier=1.0,
                delay_minutes=0.0,
                metadata=meta
            )

        if self.simulate_timeout:
            raise IntegrationTimeoutError("Traffic provider simulated timeout", service=self.provider_name)

        if self.simulate_error:
            raise IntegrationError("Traffic provider unavailable", service=self.provider_name)

        key = f"{origin.strip().lower()}->{destination.strip().lower()}"
        if key in self.route_conditions:
            obs = self.route_conditions[key]
            if obs.metadata:
                obs.metadata.applicable_id = applicable_id
            return obs

        # Default mapping from congestion level
        congestion_multipliers = {
            TrafficCongestionLevel.LOW: (1.0, 0.0, 1.2),
            TrafficCongestionLevel.MODERATE: (1.25, 5.0, 4.5),
            TrafficCongestionLevel.HEAVY: (1.5, 12.0, 7.5),
            TrafficCongestionLevel.SEVERE: (2.0, 25.0, 9.2),
            TrafficCongestionLevel.UNKNOWN: (1.0, 0.0, 0.0)
        }
        mult, extra_min, jam = congestion_multipliers.get(
            self.default_congestion, (1.0, 0.0, 0.0)
        )

        meta = self.create_metadata(
            applicable_id=applicable_id,
            observation_type=ObservationType.LIVE,
            confidence=0.9,
            details={"source": "mock_flow_engine"}
        )

        return TrafficObservation(
            origin=origin,
            destination=destination,
            congestion_level=self.default_congestion,
            delay_multiplier=mult,
            delay_minutes=extra_min,
            jam_factor=jam,
            incident_reports=["Road construction on main arterial"] if mult >= 1.5 else [],
            metadata=meta
        )
