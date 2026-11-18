"""Deterministic Mock Travel Integration for testing and development."""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import hashlib

try:
    from integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationDataError
    )
    from integrations.travel.base import (
        TravelIntegrationBase, TravelEstimate, TravelMode
    )
except ImportError:
    from backend.integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationDataError
    )
    from backend.integrations.travel.base import (
        TravelIntegrationBase, TravelEstimate, TravelMode
    )


class MockTravelIntegration(TravelIntegrationBase):
    """Calculates deterministic or custom travel durations for testing."""

    def __init__(
        self,
        provider_name: str = "mock_travel",
        enabled: bool = True,
        default_duration_min: float = 25.0,
        default_distance_km: float = 12.5,
        simulate_timeout: bool = False,
        simulate_invalid_data: bool = False
    ):
        super().__init__(
            provider_name=provider_name,
            enabled=enabled,
            timeout_seconds=3.0,
            cache_ttl_seconds=300
        )
        self.default_duration_min = default_duration_min
        self.default_distance_km = default_distance_km
        self.simulate_timeout = simulate_timeout
        self.simulate_invalid_data = simulate_invalid_data
        self.route_overrides: Dict[str, float] = {}

    def set_route_duration(self, origin: str, destination: str, duration_min: float) -> None:
        """Set an explicit duration for an origin->destination route."""
        key = f"{origin.strip().lower()}->{destination.strip().lower()}"
        self.route_overrides[key] = duration_min

    def estimate_travel(
        self,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        mode: TravelMode = TravelMode.DRIVING,
        applicable_id: Optional[str] = None
    ) -> TravelEstimate:
        if self.simulate_timeout:
            raise IntegrationTimeoutError("Travel API simulated timeout", service=self.provider_name)

        if not origin or not destination:
            raise IntegrationDataError("Origin and destination must be non-empty", service=self.provider_name)

        if self.simulate_invalid_data:
            # Return corrupt negative duration to test validation
            duration = -15.0
            distance = -5.0
        else:
            key = f"{origin.strip().lower()}->{destination.strip().lower()}"
            if key in self.route_overrides:
                duration = self.route_overrides[key]
                distance = max(1.0, duration * 0.5)
            else:
                # Deterministic duration based on hash of endpoints
                h = int(hashlib.md5(key.encode()).hexdigest()[:6], 16)
                duration = 15.0 + (h % 30)  # 15 to 44 minutes
                distance = duration * 0.6

        meta = self.create_metadata(
            applicable_id=applicable_id,
            observation_type=ObservationType.ESTIMATED,
            confidence=0.95 if not self.simulate_invalid_data else 0.1,
            details={"mode": mode.value if isinstance(mode, TravelMode) else str(mode)}
        )

        return TravelEstimate(
            origin=origin,
            destination=destination,
            duration_min=duration,
            distance_km=distance,
            mode=mode,
            departure_time=departure_time or datetime.now(timezone.utc),
            metadata=meta
        )
