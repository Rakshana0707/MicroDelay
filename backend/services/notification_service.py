"""Smart Notification Service for Context-Aware Departure Intelligence.

Schedules, evaluates, and dispatches proactive departure alerts based on
behavioral preparation milestones, travel-time changes, and traffic spikes.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
import logging

try:
    from services.departure_optimizer import DepartureOptimizationResult
except ImportError:
    from backend.services.departure_optimizer import DepartureOptimizationResult

logger = logging.getLogger("microdelay.services.notification")


class NotificationMilestone(str, Enum):
    """Notification trigger milestones in departure workflow."""
    PREP_HEADS_UP = "prep_heads_up"
    PREP_START = "prep_start"
    DEPARTURE_COUNTDOWN = "departure_countdown"
    DEPART_NOW = "depart_now"
    EN_ROUTE_DELAY = "en_route_delay"


class NotificationChannel(str, Enum):
    """Supported delivery mechanisms."""
    IN_APP = "in_app"
    PUSH = "push"
    WEBHOOK = "webhook"
    MOCK = "mock"


@dataclass
class ScheduledNotification:
    """A scheduled or triggered notification event."""
    notification_id: str
    user_id: int
    event_id: Optional[str]
    milestone: NotificationMilestone
    scheduled_for: str  # ISO timestamp
    title: str
    body: str
    channel: NotificationChannel = NotificationChannel.IN_APP
    is_dispatched: bool = False
    dispatched_at: Optional[str] = None
    priority: str = "normal"  # low, normal, high, urgent
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.milestone, NotificationMilestone):
            data["milestone"] = self.milestone.value
        if isinstance(self.channel, NotificationChannel):
            data["channel"] = self.channel.value
        return data


class NotificationService:
    """Coordinates smart notification scheduling and evaluation."""

    def __init__(self):
        # In-memory store for active notifications: Dict[notification_id, ScheduledNotification]
        self._scheduled_notifications: Dict[str, ScheduledNotification] = {}

    def schedule_departure_notifications(
        self,
        user_id: int,
        optimization: DepartureOptimizationResult,
        event_name: str = "Upcoming Event",
        channel: NotificationChannel = NotificationChannel.IN_APP
    ) -> List[ScheduledNotification]:
        """Generate smart notification milestones for an optimized departure plan."""
        notifications: List[ScheduledNotification] = []
        now = datetime.now(timezone.utc)

        prep_start_dt = datetime.fromisoformat(optimization.recommended_prep_start_time)
        departure_dt = datetime.fromisoformat(optimization.recommended_departure_time)

        # 1. Prep Heads-Up (15 minutes prior to prep start, if in future)
        heads_up_dt = prep_start_dt - timedelta(minutes=15)
        if heads_up_dt > now:
            n1 = ScheduledNotification(
                notification_id=f"notif_{user_id}_{int(heads_up_dt.timestamp())}_heads_up",
                user_id=user_id,
                event_id=optimization.context_snapshot.get("event_id") if optimization.context_snapshot else None,
                milestone=NotificationMilestone.PREP_HEADS_UP,
                scheduled_for=heads_up_dt.isoformat(),
                title=f"Upcoming: {event_name}",
                body=f"Time to begin getting ready in 15 minutes. Planned prep duration: {optimization.prep_duration_min}m.",
                channel=channel,
                priority="normal"
            )
            notifications.append(n1)

        # 2. Prep Start Alert
        if prep_start_dt > now:
            n2 = ScheduledNotification(
                notification_id=f"notif_{user_id}_{int(prep_start_dt.timestamp())}_prep_start",
                user_id=user_id,
                event_id=optimization.context_snapshot.get("event_id") if optimization.context_snapshot else None,
                milestone=NotificationMilestone.PREP_START,
                scheduled_for=prep_start_dt.isoformat(),
                title=f"Start Preparing: {event_name}",
                body=f"Begin departure preparation now to stay on schedule. Recommended departure at {departure_dt.strftime('%H:%M')}.",
                channel=channel,
                priority="high"
            )
            notifications.append(n2)

        # 3. Departure Countdown (5 min prior to departure)
        countdown_dt = departure_dt - timedelta(minutes=5)
        if countdown_dt > now:
            n3 = ScheduledNotification(
                notification_id=f"notif_{user_id}_{int(countdown_dt.timestamp())}_countdown",
                user_id=user_id,
                event_id=optimization.context_snapshot.get("event_id") if optimization.context_snapshot else None,
                milestone=NotificationMilestone.DEPARTURE_COUNTDOWN,
                scheduled_for=countdown_dt.isoformat(),
                title="5 Minutes to Departure",
                body=f"Wrap up final tasks and gather your items. Travel time: {optimization.travel_duration_min}m.",
                channel=channel,
                priority="high"
            )
            notifications.append(n3)

        # 4. Depart Now Alert
        if departure_dt > now:
            n4 = ScheduledNotification(
                notification_id=f"notif_{user_id}_{int(departure_dt.timestamp())}_depart",
                user_id=user_id,
                event_id=optimization.context_snapshot.get("event_id") if optimization.context_snapshot else None,
                milestone=NotificationMilestone.DEPART_NOW,
                scheduled_for=departure_dt.isoformat(),
                title="Time to Depart Now",
                body=f"Depart now for {event_name} to ensure timely arrival at {optimization.target_arrival_time}.",
                channel=channel,
                priority="urgent"
            )
            notifications.append(n4)

        for n in notifications:
            self._scheduled_notifications[n.notification_id] = n

        return notifications

    def get_due_notifications(
        self,
        user_id: int,
        current_time: Optional[datetime] = None
    ) -> List[ScheduledNotification]:
        """Fetch and mark pending notifications due up to current_time."""
        now = current_time or datetime.now(timezone.utc)
        due: List[ScheduledNotification] = []

        for notif in self._scheduled_notifications.values():
            if notif.user_id == user_id and not notif.is_dispatched:
                try:
                    sched_dt = datetime.fromisoformat(notif.scheduled_for)
                    if sched_dt.tzinfo is None:
                        sched_dt = sched_dt.replace(tzinfo=timezone.utc)
                    if now.tzinfo is None:
                        now = now.replace(tzinfo=timezone.utc)
                    if now >= sched_dt:
                        due.append(notif)
                except Exception:
                    pass

        return due

    def dispatch(self, notification_id: str) -> bool:
        """Mark a notification as dispatched."""
        if notification_id in self._scheduled_notifications:
            n = self._scheduled_notifications[notification_id]
            n.is_dispatched = True
            n.dispatched_at = datetime.now(timezone.utc).isoformat()
            logger.info("Dispatched notification id='%s' to user=%s", notification_id, n.user_id)
            return True
        return False
