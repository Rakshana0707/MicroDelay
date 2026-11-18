"""Base data contracts and interfaces for MicroDelay external integrations.

Every external observation adheres to a consistent data contract recording:
- Data source
- Retrieval timestamp
- Applicable event or journey
- Validity period (valid_from, valid_until)
- Observation type (live, forecast, historical, estimated)
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger("microdelay.integrations.base")


class ObservationType(str, Enum):
    """Categorization of how an observation was derived."""
    LIVE = "live"
    FORECAST = "forecast"
    HISTORICAL = "historical"
    ESTIMATED = "estimated"


class IntegrationStatus(str, Enum):
    """Operational status of an integration provider."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class IntegrationError(Exception):
    """Base exception for external integration failures."""
    def __init__(self, message: str, service: str = "unknown", is_transient: bool = False):
        super().__init__(message)
        self.service = service
        self.is_transient = is_transient


class IntegrationTimeoutError(IntegrationError):
    """Raised when an external API times out."""
    def __init__(self, message: str, service: str = "unknown"):
        super().__init__(message, service=service, is_transient=True)


class IntegrationAuthError(IntegrationError):
    """Raised when authentication with an external provider fails."""
    def __init__(self, message: str, service: str = "unknown"):
        super().__init__(message, service=service, is_transient=False)


class IntegrationDataError(IntegrationError):
    """Raised when external data is malformed or invalid."""
    def __init__(self, message: str, service: str = "unknown"):
        super().__init__(message, service=service, is_transient=False)


@dataclass
class ExternalObservationMetadata:
    """Standardized metadata attached to every external observation.
    
    Guarantees full provenance, temporal bounds, and source traceability.
    """
    source: str
    retrieval_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    applicable_id: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    observation_type: ObservationType = ObservationType.ESTIMATED
    confidence: float = 1.0
    is_fallback: bool = False
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self, at_time: Optional[datetime] = None) -> bool:
        """Check whether the observation is currently within its validity period."""
        if not self.valid_until:
            return True
        check_time = at_time or datetime.now(timezone.utc)
        try:
            valid_until_dt = datetime.fromisoformat(self.valid_until)
            if valid_until_dt.tzinfo is None:
                valid_until_dt = valid_until_dt.replace(tzinfo=timezone.utc)
            if check_time.tzinfo is None:
                check_time = check_time.replace(tzinfo=timezone.utc)
            return check_time <= valid_until_dt
        except (ValueError, TypeError):
            return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to standard dictionary."""
        data = asdict(self)
        if isinstance(self.observation_type, ObservationType):
            data["observation_type"] = self.observation_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExternalObservationMetadata":
        """Reconstruct metadata from dictionary."""
        obs_type = data.get("observation_type", ObservationType.ESTIMATED.value)
        if isinstance(obs_type, str):
            try:
                obs_type = ObservationType(obs_type)
            except ValueError:
                obs_type = ObservationType.ESTIMATED
        return cls(
            source=data.get("source", "unknown"),
            retrieval_timestamp=data.get("retrieval_timestamp", datetime.now(timezone.utc).isoformat()),
            applicable_id=data.get("applicable_id"),
            valid_from=data.get("valid_from"),
            valid_until=data.get("valid_until"),
            observation_type=obs_type,
            confidence=float(data.get("confidence", 1.0)),
            is_fallback=bool(data.get("is_fallback", False)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            details=data.get("details", {})
        )


@dataclass
class IntegrationHealth:
    """Health check outcome for an integration adapter."""
    service_name: str
    provider_name: str
    status: IntegrationStatus
    message: str = "Operating normally"
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class BaseIntegration:
    """Abstract base class for all external service integrations."""

    def __init__(
        self,
        service_name: str,
        provider_name: str,
        enabled: bool = True,
        timeout_seconds: float = 5.0,
        cache_ttl_seconds: int = 300
    ):
        self.service_name = service_name
        self.provider_name = provider_name
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._last_health = IntegrationHealth(
            service_name=self.service_name,
            provider_name=self.provider_name,
            status=IntegrationStatus.HEALTHY if enabled else IntegrationStatus.DISABLED,
            message="Initialized"
        )

    def is_enabled(self) -> bool:
        """Check if integration is active and enabled."""
        return self.enabled

    def enable(self) -> None:
        """Enable integration."""
        self.enabled = True
        self._last_health.status = IntegrationStatus.HEALTHY
        self._last_health.message = "Enabled"

    def disable(self) -> None:
        """Disable integration."""
        self.enabled = False
        self._last_health.status = IntegrationStatus.DISABLED
        self._last_health.message = "Disabled by configuration"

    def health_check(self) -> IntegrationHealth:
        """Perform health check. Subclasses can override for live probing."""
        if not self.enabled:
            return IntegrationHealth(
                service_name=self.service_name,
                provider_name=self.provider_name,
                status=IntegrationStatus.DISABLED,
                message="Integration is disabled"
            )
        return self._last_health

    def create_metadata(
        self,
        applicable_id: Optional[str] = None,
        observation_type: ObservationType = ObservationType.LIVE,
        ttl_seconds: Optional[int] = None,
        confidence: float = 1.0,
        is_fallback: bool = False,
        latency_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None
    ) -> ExternalObservationMetadata:
        """Helper to create standardized observation metadata."""
        now = datetime.now(timezone.utc)
        ttl = ttl_seconds if ttl_seconds is not None else self.cache_ttl_seconds
        valid_until = (now + timedelta(seconds=ttl)).isoformat() if ttl > 0 else None

        return ExternalObservationMetadata(
            source=f"{self.service_name}:{self.provider_name}",
            retrieval_timestamp=now.isoformat(),
            applicable_id=applicable_id,
            valid_from=now.isoformat(),
            valid_until=valid_until,
            observation_type=observation_type,
            confidence=confidence,
            is_fallback=is_fallback,
            latency_ms=latency_ms,
            details=details or {}
        )
