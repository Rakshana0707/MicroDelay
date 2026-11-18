"""Weather Service managing weather observations and departure buffer impact."""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

try:
    from integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationError,
        IntegrationTimeoutError
    )
    from integrations.registry import IntegrationRegistry
    from integrations.weather.base import (
        WeatherIntegrationBase, WeatherObservation, WeatherSeverity
    )
except ImportError:
    from backend.integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationError,
        IntegrationTimeoutError
    )
    from backend.integrations.registry import IntegrationRegistry
    from backend.integrations.weather.base import (
        WeatherIntegrationBase, WeatherObservation, WeatherSeverity
    )

logger = logging.getLogger("microdelay.services.weather")


class WeatherService:
    """Coordinates atmospheric condition observations with graceful omission on failure."""

    def __init__(self, registry: Optional[IntegrationRegistry] = None):
        self.registry = registry or IntegrationRegistry.get_instance()

    def get_weather_context(
        self,
        location: Optional[str] = None,
        target_time: Optional[datetime] = None,
        event_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch weather observations. If unavailable or disabled, omits weather adjustments gracefully."""
        now = datetime.now(timezone.utc)
        target = target_time or now
        clean_loc = location.strip() if location else None

        if not clean_loc:
            # Missing location -> gracefully omit weather
            meta = ExternalObservationMetadata(
                source="weather_omitted:no_location",
                applicable_id=str(event_id or ""),
                observation_type=ObservationType.ESTIMATED,
                confidence=1.0,
                is_fallback=True,
                details={"reason": "Location not specified; weather adjustments omitted"}
            )
            return {
                "weather_conditions": None,
                "recommended_buffer_min": 0.0,
                "is_omitted": True,
                "source": "weather_omitted",
                "metadata": meta
            }

        weather_integration: Optional[WeatherIntegrationBase] = self.registry.get("weather")
        if not weather_integration or not weather_integration.is_enabled():
            meta = ExternalObservationMetadata(
                source="weather_disabled",
                applicable_id=str(event_id or ""),
                observation_type=ObservationType.ESTIMATED,
                is_fallback=True,
                details={"reason": "Weather integration disabled"}
            )
            return {
                "weather_conditions": None,
                "recommended_buffer_min": 0.0,
                "is_omitted": True,
                "source": "weather_disabled",
                "metadata": meta
            }

        try:
            obs: WeatherObservation = weather_integration.get_weather_conditions(
                location=clean_loc,
                target_time=target,
                applicable_id=str(event_id or "")
            )
            return {
                "weather_conditions": obs.to_dict(),
                "recommended_buffer_min": obs.recommended_buffer_min,
                "is_omitted": False,
                "source": f"live_weather:{weather_integration.provider_name}",
                "metadata": obs.metadata
            }
        except (IntegrationTimeoutError, IntegrationError) as e:
            logger.warning("Weather integration failed for location='%s': %s. Omitting weather buffer.", clean_loc, e)
            meta = ExternalObservationMetadata(
                source=f"weather_failed:{weather_integration.provider_name}",
                applicable_id=str(event_id or ""),
                observation_type=ObservationType.ESTIMATED,
                is_fallback=True,
                details={"error": str(e), "note": "Gracefully omitted weather adjustments"}
            )
            return {
                "weather_conditions": None,
                "recommended_buffer_min": 0.0,
                "is_omitted": True,
                "source": "weather_failed_fallback",
                "metadata": meta
            }
        except Exception as e:
            logger.error("Unexpected error in weather integration: %s", e)
            return {
                "weather_conditions": None,
                "recommended_buffer_min": 0.0,
                "is_omitted": True,
                "source": "weather_error_fallback",
                "metadata": None
            }
