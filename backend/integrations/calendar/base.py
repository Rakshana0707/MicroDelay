"""Calendar integration interfaces and data contracts."""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
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


@dataclass
class CalendarEvent:
    """Standardized event representation from external calendar providers."""
    event_id: str
    title: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    destination: Optional[str] = None
    description: Optional[str] = None
    importance: str = "medium"
    is_all_day: bool = False
    metadata: Optional[ExternalObservationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "location": self.location or self.destination,
            "destination": self.destination or self.location,
            "description": self.description,
            "importance": self.importance,
            "is_all_day": self.is_all_day,
            "metadata": self.metadata.to_dict() if self.metadata else None
        }


class CalendarIntegrationBase(BaseIntegration):
    """Abstract base class for Calendar integrations (Google, Outlook, etc.)."""

    def __init__(
        self,
        provider_name: str,
        enabled: bool = True,
        timeout_seconds: float = 5.0,
        cache_ttl_seconds: int = 300
    ):
        super().__init__(
            service_name="calendar",
            provider_name=provider_name,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds
        )

    @abstractmethod
    def get_upcoming_events(
        self,
        user_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10
    ) -> List[CalendarEvent]:
        """Fetch upcoming events for a user within a time window."""
        pass

    @abstractmethod
    def get_event_by_id(
        self,
        user_id: int,
        event_id: str
    ) -> Optional[CalendarEvent]:
        """Fetch a specific event by ID."""
        pass
