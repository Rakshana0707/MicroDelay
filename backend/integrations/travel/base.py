"""Travel time and distance estimation interfaces and data contracts."""

from abc import abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass

try:
    from integrations.base import (
        BaseIntegration, ExternalObservationMetadata, ObservationType,
        IntegrationStatus, IntegrationError
    )
except ImportError:
    from backend.integrations.base import (
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
class TravelEstimate:
    """Estimated travel duration, distance, and routing details."""
    origin: str
    destination: str
    duration_min: float
    distance_km: float
    mode: TravelMode = TravelMode.DRIVING
    departure_time: Optional[datetime] = None
    metadata: Optional[ExternalObservationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "duration_min": round(self.duration_min, 1),
            "distance_km": round(self.distance_km, 2),
            "mode": self.mode.value if isinstance(self.mode, TravelMode) else str(self.mode),
            "departure_time": self.departure_time.isoformat() if self.departure_time else None,
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
