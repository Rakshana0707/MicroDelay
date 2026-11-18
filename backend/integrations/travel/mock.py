"""Deterministic Mock Travel Integration for testing and development."""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import hashlib

try:
    from backend.integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationDataError
    )
    from backend.integrations.travel.base import (
        TravelIntegrationBase, TravelEstimate, TravelMode, RouteOption
    )
except ImportError:
    from integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationDataError
    )
    from integrations.travel.base import (
        TravelIntegrationBase, TravelEstimate, TravelMode, RouteOption
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

    def _mode_multiplier(self, mode: Any) -> float:
        """Return relative duration multiplier for different travel modes compared to driving."""
        val = getattr(mode, "value", str(mode)).lower()
        if "walking" in val:
            return 3.5
        elif "bicycling" in val:
            return 1.8
        elif "transit" in val:
            return 1.4
        return 1.0

    def estimate_travel(
        self,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        mode: TravelMode = TravelMode.DRIVING,
        applicable_id: Optional[str] = None
    ) -> TravelEstimate:
        routes = self.calculate_routes(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            mode=mode,
            alternatives=True,
            applicable_id=applicable_id
        )

        selected = routes[0] if routes else None
        duration = selected.duration_min if selected else self.default_duration_min
        duration_traffic = selected.duration_in_traffic_min if selected else duration
        distance = selected.distance_km if selected else self.default_distance_km

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
            duration_in_traffic_min=duration_traffic,
            distance_km=distance,
            mode=mode,
            departure_time=departure_time or datetime.now(timezone.utc),
            selected_route=selected.summary if selected else "Primary Route",
            route_options=routes,
            metadata=meta
        )

    def calculate_routes(
        self,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        mode: TravelMode = TravelMode.DRIVING,
        alternatives: bool = True,
        applicable_id: Optional[str] = None
    ) -> List[RouteOption]:
        if self.simulate_timeout:
            raise IntegrationTimeoutError("Travel API simulated timeout", service=self.provider_name)

        if not origin or not destination:
            raise IntegrationDataError("Origin and destination must be non-empty", service=self.provider_name)

        now_str = (departure_time or datetime.now(timezone.utc)).isoformat()

        if self.simulate_invalid_data:
            return [
                RouteOption(
                    route_id="err_route_1",
                    summary="Invalid Test Route",
                    origin=origin,
                    destination=destination,
                    travel_mode=mode,
                    duration_min=-15.0,
                    duration_in_traffic_min=-12.0,
                    distance_km=-5.0,
                    retrieval_timestamp=now_str,
                    provider=self.provider_name,
                    is_recommended=True
                )
            ]

        key = f"{origin.strip().lower()}->{destination.strip().lower()}"
        mult = self._mode_multiplier(mode)

        if key in self.route_overrides:
            base_duration = self.route_overrides[key] * mult
            base_dist = max(1.0, base_duration * 0.5)
            traffic_dur = base_duration
        else:
            h = int(hashlib.md5(key.encode()).hexdigest()[:6], 16)
            base_duration = (15.0 + (h % 30)) * mult
            base_dist = max(2.0, base_duration * 0.45)
            traffic_dur = round(base_duration * 1.15, 1) if mode == TravelMode.DRIVING else round(base_duration, 1)

        # Primary route
        routes = [
            RouteOption(
                route_id=f"route_fastest_{hash(origin + destination) % 10000}",
                summary="Primary / Fastest Route",
                origin=origin,
                destination=destination,
                travel_mode=mode,
                duration_min=round(base_duration, 1),
                duration_in_traffic_min=round(traffic_dur, 1),
                distance_km=round(base_dist, 2),
                retrieval_timestamp=now_str,
                provider=self.provider_name,
                is_recommended=True
            )
        ]

        if alternatives:
            # Route 2: Alternative corridor (slightly longer distance, less congestion)
            routes.append(
                RouteOption(
                    route_id=f"route_alt1_{hash(origin + destination) % 10000}",
                    summary="Alternative via Arterial Corridor",
                    origin=origin,
                    destination=destination,
                    travel_mode=mode,
                    duration_min=round(base_duration * 1.18, 1),
                    duration_in_traffic_min=round(base_duration * 1.22, 1) if mode == TravelMode.DRIVING else round(base_duration * 1.18, 1),
                    distance_km=round(base_dist * 1.12, 2),
                    retrieval_timestamp=now_str,
                    provider=self.provider_name,
                    is_recommended=False
                )
            )

            # Route 3: Scenic or Local route
            routes.append(
                RouteOption(
                    route_id=f"route_alt2_{hash(origin + destination) % 10000}",
                    summary="Local Streets (Avoid Tolls)",
                    origin=origin,
                    destination=destination,
                    travel_mode=mode,
                    duration_min=round(base_duration * 1.30, 1),
                    duration_in_traffic_min=round(base_duration * 1.35, 1) if mode == TravelMode.DRIVING else round(base_duration * 1.30, 1),
                    distance_km=round(base_dist * 1.05, 2),
                    retrieval_timestamp=now_str,
                    provider=self.provider_name,
                    is_recommended=False
                )
            )

        return routes
