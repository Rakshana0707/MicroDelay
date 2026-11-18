"""Configurable Mock Calendar Integration for unit testing and offline development."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

try:
    from integrations.base import (
        ObservationType, IntegrationStatus, IntegrationHealth,
        IntegrationTimeoutError, IntegrationAuthError
    )
    from integrations.calendar.base import CalendarIntegrationBase, CalendarEvent
except ImportError:
    from backend.integrations.base import (
        ObservationType, IntegrationStatus, IntegrationHealth,
        IntegrationTimeoutError, IntegrationAuthError
    )
    from backend.integrations.calendar.base import CalendarIntegrationBase, CalendarEvent


class MockCalendarIntegration(CalendarIntegrationBase):
    """Deterministic mock calendar for testing degradation, timeouts, and edge cases."""

    def __init__(
        self,
        provider_name: str = "mock_calendar",
        enabled: bool = True,
        simulate_timeout: bool = False,
        simulate_auth_error: bool = False,
        timeout_seconds: float = 2.0
    ):
        super().__init__(
            provider_name=provider_name,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=300
        )
        self.simulate_timeout = simulate_timeout
        self.simulate_auth_error = simulate_auth_error
        self.events: List[CalendarEvent] = []

    def set_mock_events(self, events: List[CalendarEvent]) -> None:
        """Seed mock events."""
        self.events = list(events)

    def add_event(
        self,
        event_id: str,
        title: str,
        start_time: datetime,
        end_time: datetime,
        location: Optional[str] = "Campus Engineering Building",
        importance: str = "high"
    ) -> CalendarEvent:
        """Convenience method to construct and register a mock event."""
        meta = self.create_metadata(
            applicable_id=event_id,
            observation_type=ObservationType.ESTIMATED,
            confidence=1.0,
            details={"mock": True}
        )
        ev = CalendarEvent(
            event_id=event_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            location=location,
            destination=location,
            importance=importance,
            metadata=meta
        )
        self.events.append(ev)
        return ev

    def get_upcoming_events(
        self,
        user_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10
    ) -> List[CalendarEvent]:
        if not self.enabled:
            return []
        if self.simulate_timeout:
            raise IntegrationTimeoutError(
                f"Mock calendar request timed out after {self.timeout_seconds}s",
                service=self.provider_name
            )
        if self.simulate_auth_error:
            raise IntegrationAuthError(
                "Mock calendar authentication rejected",
                service=self.provider_name
            )

        now = datetime.now(timezone.utc)
        start = start_time or now
        end = end_time or (now + timedelta(days=7))

        matching = []
        for ev in self.events:
            ev_start = ev.start_time if ev.start_time.tzinfo else ev.start_time.replace(tzinfo=timezone.utc)
            if start <= ev_start <= end:
                matching.append(ev)
        return matching[:limit]

    def get_event_by_id(
        self,
        user_id: int,
        event_id: str
    ) -> Optional[CalendarEvent]:
        if not self.enabled:
            return None
        if self.simulate_timeout:
            raise IntegrationTimeoutError("Mock calendar timed out", service=self.provider_name)
        if self.simulate_auth_error:
            raise IntegrationAuthError("Mock calendar auth failed", service=self.provider_name)

        for ev in self.events:
            if ev.event_id == str(event_id):
                return ev
        return None
