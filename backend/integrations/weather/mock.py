"""Deterministic and configurable Mock Weather Integration for testing."""

from typing import Optional, Dict, Any
from datetime import datetime, timezone

try:
    from integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationError
    )
    from integrations.weather.base import (
        WeatherIntegrationBase, WeatherObservation, WeatherSeverity
    )
except ImportError:
    from backend.integrations.base import (
        ObservationType, IntegrationTimeoutError, IntegrationError
    )
    from backend.integrations.weather.base import (
        WeatherIntegrationBase, WeatherObservation, WeatherSeverity
    )


class MockWeatherIntegration(WeatherIntegrationBase):
    """Simulates weather observations and departure buffer recommendations."""

    def __init__(
        self,
        provider_name: str = "mock_weather",
        enabled: bool = True,
        default_condition: str = "Clear",
        default_severity: WeatherSeverity = WeatherSeverity.BENIGN,
        default_buffer_min: float = 0.0,
        simulate_timeout: bool = False,
        simulate_error: bool = False
    ):
        super().__init__(
            provider_name=provider_name,
            enabled=enabled,
            timeout_seconds=3.0,
            cache_ttl_seconds=600
        )
        self.default_condition = default_condition
        self.default_severity = default_severity
        self.default_buffer_min = default_buffer_min
        self.simulate_timeout = simulate_timeout
        self.simulate_error = simulate_error
        self.location_overrides: Dict[str, WeatherObservation] = {}

    def set_location_weather(
        self,
        location: str,
        condition: str,
        temperature_c: float,
        severity: WeatherSeverity,
        recommended_buffer_min: float,
        precipitation_probability: float = 0.0
    ) -> None:
        """Set explicit weather conditions for a location."""
        loc_key = location.strip().lower()
        meta = self.create_metadata(
            observation_type=ObservationType.FORECAST,
            details={"forecast": True}
        )
        self.location_overrides[loc_key] = WeatherObservation(
            location=location,
            condition=condition,
            temperature_c=temperature_c,
            precipitation_probability=precipitation_probability,
            severity=severity,
            recommended_buffer_min=recommended_buffer_min,
            metadata=meta
        )

    def get_weather_conditions(
        self,
        location: str,
        target_time: Optional[datetime] = None,
        applicable_id: Optional[str] = None
    ) -> WeatherObservation:
        if not self.enabled:
            # Graceful omission: benign default with is_fallback=True
            meta = self.create_metadata(
                applicable_id=applicable_id,
                observation_type=ObservationType.ESTIMATED,
                is_fallback=True,
                details={"omitted": "Weather integration disabled"}
            )
            return WeatherObservation(
                location=location,
                condition="Unknown",
                severity=WeatherSeverity.BENIGN,
                recommended_buffer_min=0.0,
                metadata=meta
            )

        if self.simulate_timeout:
            raise IntegrationTimeoutError("Weather provider timeout", service=self.provider_name)

        if self.simulate_error:
            raise IntegrationError("Weather provider connection failure", service=self.provider_name)

        loc_key = location.strip().lower() if location else "default"
        if loc_key in self.location_overrides:
            obs = self.location_overrides[loc_key]
            if obs.metadata:
                obs.metadata.applicable_id = applicable_id
            return obs

        # Default heuristic based on severity
        buffer_map = {
            WeatherSeverity.BENIGN: 0.0,
            WeatherSeverity.ADVERSE: 7.0,
            WeatherSeverity.SEVERE: 15.0,
            WeatherSeverity.HAZARDOUS: 25.0
        }
        rec_buffer = self.default_buffer_min or buffer_map.get(self.default_severity, 0.0)

        meta = self.create_metadata(
            applicable_id=applicable_id,
            observation_type=ObservationType.FORECAST,
            confidence=0.88,
            details={"provider": "mock_weather_engine"}
        )

        return WeatherObservation(
            location=location or "Current Location",
            condition=self.default_condition,
            temperature_c=21.0,
            precipitation_probability=0.2 if self.default_severity != WeatherSeverity.BENIGN else 0.0,
            severity=self.default_severity,
            recommended_buffer_min=rec_buffer,
            metadata=meta
        )
