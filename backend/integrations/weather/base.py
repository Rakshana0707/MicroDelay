"""Weather condition integration interfaces and data contracts."""

from abc import abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
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


class WeatherSeverity(str, Enum):
    """Weather severity classification for departure safety impact."""
    BENIGN = "benign"
    ADVERSE = "adverse"
    SEVERE = "severe"
    HAZARDOUS = "hazardous"


@dataclass
class WeatherObservation:
    """Current or forecast atmospheric conditions affecting departure."""
    location: str
    condition: str = "Clear"
    temperature_c: float = 20.0
    precipitation_probability: float = 0.0
    wind_speed_kmh: float = 10.0
    visibility_km: float = 10.0
    severity: WeatherSeverity = WeatherSeverity.BENIGN
    recommended_buffer_min: float = 0.0
    metadata: Optional[ExternalObservationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "condition": self.condition,
            "temperature_c": round(self.temperature_c, 1),
            "precipitation_probability": round(self.precipitation_probability, 2),
            "wind_speed_kmh": round(self.wind_speed_kmh, 1),
            "visibility_km": round(self.visibility_km, 1),
            "severity": self.severity.value if isinstance(self.severity, WeatherSeverity) else str(self.severity),
            "recommended_buffer_min": round(self.recommended_buffer_min, 1),
            "metadata": self.metadata.to_dict() if self.metadata else None
        }


class WeatherIntegrationBase(BaseIntegration):
    """Abstract base class for weather condition integrations."""

    def __init__(
        self,
        provider_name: str,
        enabled: bool = True,
        timeout_seconds: float = 4.0,
        cache_ttl_seconds: int = 600
    ):
        super().__init__(
            service_name="weather",
            provider_name=provider_name,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds
        )

    @abstractmethod
    def get_weather_conditions(
        self,
        location: str,
        target_time: Optional[datetime] = None,
        applicable_id: Optional[str] = None
    ) -> WeatherObservation:
        """Fetch weather observations or forecast for a given location and time."""
        pass
