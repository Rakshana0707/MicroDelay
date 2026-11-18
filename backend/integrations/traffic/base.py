"""Traffic condition integration interfaces and data contracts."""

from abc import abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field

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


class TrafficCongestionLevel(str, Enum):
    """Traffic congestion severity levels."""
    LOW = "low"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"
    UNKNOWN = "unknown"


@dataclass
class TrafficObservation:
    """Current or forecast traffic congestion observation along a route."""
    origin: str
    destination: str
    congestion_level: TrafficCongestionLevel = TrafficCongestionLevel.LOW
    delay_multiplier: float = 1.0
    delay_minutes: float = 0.0
    jam_factor: float = 0.0  # 0.0 (free flow) to 10.0 (standstill)
    incident_reports: List[str] = field(default_factory=list)
    metadata: Optional[ExternalObservationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "congestion_level": (
                self.congestion_level.value
                if isinstance(self.congestion_level, TrafficCongestionLevel)
                else str(self.congestion_level)
            ),
            "delay_multiplier": round(self.delay_multiplier, 2),
            "delay_minutes": round(self.delay_minutes, 1),
            "jam_factor": round(self.jam_factor, 1),
            "incident_reports": self.incident_reports,
            "metadata": self.metadata.to_dict() if self.metadata else None
        }


class TrafficIntegrationBase(BaseIntegration):
    """Abstract base class for live traffic condition providers."""

    def __init__(
        self,
        provider_name: str,
        enabled: bool = True,
        timeout_seconds: float = 4.0,
        cache_ttl_seconds: int = 180
    ):
        super().__init__(
            service_name="traffic",
            provider_name=provider_name,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds
        )

    @abstractmethod
    def get_traffic_conditions(
        self,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        applicable_id: Optional[str] = None
    ) -> TrafficObservation:
        """Fetch real-time traffic observations or congestion index along a route."""
        pass
