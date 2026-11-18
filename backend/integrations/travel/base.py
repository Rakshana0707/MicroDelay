"""Travel time and distance estimation interfaces and data contracts."""

from abc import abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass, field

try:
    from backend.integrations.base import (
        BaseIntegration, ExternalObservationMetadata, ObservationType,
        IntegrationStatus, IntegrationError
    )
except ImportError:
    from integrations.base import (
        BaseIntegration, ExternalObservationMetadata, ObservationType,
        IntegrationStatus, IntegrationError
    )


class TravelMode(str, Enum):
    """Modes of transport for departure intelligence."""
    DRIVING = "driving"
    TRANSIT = "transit"
    WALKING = "walking"
    BICYCLING = "bicycling"


@dataclass
class RouteOption:
    """Individual alternative route option returned by routing providers (Prompt 3.4)."""
    route_id: str
    summary: str
    origin: str
    destination: str
    travel_mode: TravelMode
    duration_min: float
    distance_km: float
    retrieval_timestamp: str
    provider: str
    duration_in_traffic_min: Optional[float] = None
    is_recommended: bool = False
    warnings: List[str] = field(default_factory=list)
    polyline: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "summary": self.summary,
            "origin": self.origin,
            "destination": self.destination,
            "travel_mode": self.travel_mode.value if isinstance(self.travel_mode, TravelMode) else str(self.travel_mode),
            "duration_min": round(self.duration_min, 1),
            "duration_in_traffic_min": round(self.duration_in_traffic_min, 1) if self.duration_in_traffic_min is not None else None,
            "distance_km": round(self.distance_km, 2),
            "retrieval_timestamp": self.retrieval_timestamp,
            "provider": self.provider,
            "is_recommended": self.is_recommended,
            "warnings": self.warnings,
            "polyline": self.polyline
        }


@dataclass
class TravelEstimate:
    """Estimated travel duration, distance, and routing details."""
    origin: str
    destination: str
    duration_min: float
    distance_km: float
    mode: TravelMode = TravelMode.DRIVING
    departure_time: Optional[datetime] = None
    duration_in_traffic_min: Optional[float] = None
    selected_route: Optional[str] = None
    route_options: List[RouteOption] = field(default_factory=list)
    metadata: Optional[ExternalObservationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "duration_min": round(self.duration_min, 1),
            "duration_in_traffic_min": round(self.duration_in_traffic_min, 1) if self.duration_in_traffic_min is not None else None,
            "distance_km": round(self.distance_km, 2),
            "mode": self.mode.value if isinstance(self.mode, TravelMode) else str(self.mode),
            "departure_time": self.departure_time.isoformat() if self.departure_time else None,
            "selected_route": self.selected_route,
            "route_options": [opt.to_dict() for opt in self.route_options],
            "metadata": self.metadata.to_dict() if self.metadata else None
        }


class TravelIntegrationBase(BaseIntegration):
    """Abstract base class for travel time estimation integrations."""

    def __init__(
        self,
        provider_name: str,
        enabled: bool = True,
        timeout_seconds: float = 5.0,
        cache_ttl_seconds: int = 300
    ):
        super().__init__(
            service_name="travel",
            provider_name=provider_name,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds
        )

    @abstractmethod
    def estimate_travel(
        self,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        mode: TravelMode = TravelMode.DRIVING,
        applicable_id: Optional[str] = None
    ) -> TravelEstimate:
        """Estimate travel duration and distance between origin and destination."""
        pass

    @abstractmethod
    def calculate_routes(
        self,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        mode: TravelMode = TravelMode.DRIVING,
        alternatives: bool = True,
        applicable_id: Optional[str] = None
    ) -> List[RouteOption]:
        """Calculate and compare multiple route options between origin and destination."""
        pass
