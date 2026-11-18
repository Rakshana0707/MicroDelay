"""Google Maps Routes and Directions API Integration (Prompt 3.4).

Supports multi-modal routing (driving, transit, walking, bicycling),
traffic-aware duration estimation, and alternative route comparison.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import logging
import httpx

try:
    from backend.integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationDataError,
        IntegrationAuthError, ExternalObservationMetadata
    )
    from backend.integrations.travel.base import (
        TravelIntegrationBase, TravelEstimate, TravelMode, RouteOption
    )
except ImportError:
    from integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationDataError,
        IntegrationAuthError, ExternalObservationMetadata
    )
    from integrations.travel.base import (
        TravelIntegrationBase, TravelEstimate, TravelMode, RouteOption
    )

logger = logging.getLogger(__name__)


class GoogleRoutesIntegration(TravelIntegrationBase):
    """Google Maps Directions & Routes API client."""

    DIRECTIONS_API_URL = "https://maps.googleapis.com/maps/api/directions/json"

    def __init__(
        self,
        api_key: Optional[str] = None,
        enabled: bool = True,
        timeout_seconds: float = 6.0,
        cache_ttl_seconds: int = 300
    ):
        super().__init__(
            provider_name="google_routes",
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds
        )
        self.api_key = api_key

    def _map_travel_mode(self, mode: TravelMode) -> str:
        """Map internal TravelMode enum to Google Directions API mode string."""
        mapping = {
            TravelMode.DRIVING: "driving",
            TravelMode.TRANSIT: "transit",
            TravelMode.WALKING: "walking",
            TravelMode.BICYCLING: "bicycling"
        }
        return mapping.get(mode, "driving")

    def calculate_routes(
        self,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        mode: TravelMode = TravelMode.DRIVING,
        alternatives: bool = True,
        applicable_id: Optional[str] = None
    ) -> List[RouteOption]:
        """Call Google Directions API and parse routes with traffic conditions."""
        if not self.enabled:
            raise IntegrationDataError("Google Routes integration is disabled", service=self.provider_name)

        if not self.api_key:
            raise IntegrationAuthError("Google Maps API key is not configured", service=self.provider_name)

        if not origin or not destination:
            raise IntegrationDataError("Origin and destination must be non-empty", service=self.provider_name)

        google_mode = self._map_travel_mode(mode)
        dep_time = departure_time or datetime.now(timezone.utc)
        dep_timestamp = int(dep_time.timestamp())

        params: Dict[str, Any] = {
            "origin": origin,
            "destination": destination,
            "mode": google_mode,
            "alternatives": "true" if alternatives else "false",
            "key": self.api_key
        }

        # Google only supports departure_time for driving and transit
        if google_mode in ("driving", "transit"):
            params["departure_time"] = dep_timestamp

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(self.DIRECTIONS_API_URL, params=params)
        except httpx.TimeoutException as e:
            raise IntegrationTimeoutError(f"Google Maps Directions API timed out after {self.timeout_seconds}s", service=self.provider_name) from e
        except Exception as e:
            raise IntegrationDataError(f"HTTP request to Google Directions API failed: {e}", service=self.provider_name) from e

        if response.status_code != 200:
            raise IntegrationDataError(
                f"Google Directions API returned HTTP {response.status_code}: {response.text[:200]}",
                service=self.provider_name
            )

        data = response.json()
        status = data.get("status")

        if status == "REQUEST_DENIED":
            err_msg = data.get("error_message", "Request denied by Google Maps API")
            raise IntegrationAuthError(err_msg, service=self.provider_name)
        elif status in ("NOT_FOUND", "ZERO_RESULTS"):
            raise IntegrationDataError(f"No routes found between '{origin}' and '{destination}' (status={status})", service=self.provider_name)
        elif status != "OK":
            err_msg = data.get("error_message", f"Directions API status: {status}")
            raise IntegrationDataError(err_msg, service=self.provider_name)

        routes_data = data.get("routes", [])
        if not routes_data:
            raise IntegrationDataError("No route options in Directions response", service=self.provider_name)

        retrieval_time = datetime.now(timezone.utc).isoformat()
        route_options: List[RouteOption] = []

        for idx, r in enumerate(routes_data):
            summary = r.get("summary") or f"Route {idx + 1}"
            legs = r.get("legs", [])
            if not legs:
                continue

            # Aggregate distance and duration across legs
            total_distance_m = sum(leg.get("distance", {}).get("value", 0) for leg in legs)
            total_duration_sec = sum(leg.get("duration", {}).get("value", 0) for leg in legs)

            # Traffic duration if available (driving mode with departure_time)
            traffic_duration_sec = sum(
                leg.get("duration_in_traffic", {}).get("value", leg.get("duration", {}).get("value", 0))
                for leg in legs
            ) if google_mode == "driving" else None

            warnings = r.get("warnings", [])
            overview_polyline = r.get("overview_polyline", {}).get("points")

            route_options.append(
                RouteOption(
                    route_id=f"google_route_{idx}_{int(dep_timestamp)}",
                    summary=summary,
                    origin=legs[0].get("start_address", origin),
                    destination=legs[-1].get("end_address", destination),
                    travel_mode=mode,
                    duration_min=round(total_duration_sec / 60.0, 1),
                    duration_in_traffic_min=round(traffic_duration_sec / 60.0, 1) if traffic_duration_sec else None,
                    distance_km=round(total_distance_m / 1000.0, 2),
                    retrieval_timestamp=retrieval_time,
                    provider=self.provider_name,
                    is_recommended=(idx == 0),
                    warnings=warnings,
                    polyline=overview_polyline
                )
            )

        return route_options

    def estimate_travel(
        self,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        mode: TravelMode = TravelMode.DRIVING,
        applicable_id: Optional[str] = None
    ) -> TravelEstimate:
        """Estimate travel duration and retrieve route options."""
        routes = self.calculate_routes(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            mode=mode,
            alternatives=True,
            applicable_id=applicable_id
        )

        primary = routes[0]
        eff_duration = primary.duration_in_traffic_min or primary.duration_min

        meta = self.create_metadata(
            applicable_id=applicable_id,
            observation_type=ObservationType.LIVE if primary.duration_in_traffic_min else ObservationType.ESTIMATED,
            confidence=0.95,
            details={
                "provider": self.provider_name,
                "summary": primary.summary,
                "mode": mode.value if isinstance(mode, TravelMode) else str(mode),
                "has_traffic": primary.duration_in_traffic_min is not None
            }
        )

        return TravelEstimate(
            origin=primary.origin,
            destination=primary.destination,
            duration_min=primary.duration_min,
            duration_in_traffic_min=primary.duration_in_traffic_min,
            distance_km=primary.distance_km,
            mode=mode,
            departure_time=departure_time or datetime.now(timezone.utc),
            selected_route=primary.summary,
            route_options=routes,
            metadata=meta
        )
