"""Calendar Service for managing synchronized calendar events, classifications, and routine fallbacks."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

try:
    from database import Event, CalendarConnection, CalendarEventRecord
    from integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationError,
        IntegrationTimeoutError, IntegrationAuthError
    )
    from integrations.registry import IntegrationRegistry
    from integrations.calendar.base import CalendarIntegrationBase, CalendarEvent
    from integrations.calendar.google import GoogleCalendarIntegration
    from integrations.calendar.outlook import OutlookCalendarIntegration
except ImportError:
    from backend.database import Event, CalendarConnection, CalendarEventRecord
    from backend.integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationError,
        IntegrationTimeoutError, IntegrationAuthError
    )
    from backend.integrations.registry import IntegrationRegistry
    from backend.integrations.calendar.base import CalendarIntegrationBase, CalendarEvent
    from backend.integrations.calendar.google import GoogleCalendarIntegration
    from backend.integrations.calendar.outlook import OutlookCalendarIntegration

logger = logging.getLogger("microdelay.services.calendar")


class CalendarService:
    """Coordinates calendar schedule ingestion, event synchronization, and routine fallbacks."""

    def __init__(
        self,
        registry: Optional[IntegrationRegistry] = None,
        google_integration: Optional[GoogleCalendarIntegration] = None,
        outlook_integration: Optional[OutlookCalendarIntegration] = None
    ):
        self.registry = registry or IntegrationRegistry.get_instance()
        self.google_integration = google_integration or GoogleCalendarIntegration()
        self.outlook_integration = outlook_integration or OutlookCalendarIntegration()

    def get_event_context(
        self,
        user_id: int,
        event_id: Optional[str] = None,
        event_name: Optional[str] = None,
        target_time: Optional[datetime] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Resolve event details from synchronized external calendar, database routine templates, or heuristics.
        
        Resolution Priority:
        1. Synchronized `CalendarEventRecord` in DB
        2. Live External Calendar Integration Adapter
        3. Database Routine Template (`Event` table)
        4. Static Heuristic Default
        """
        now = datetime.now(timezone.utc)
        target = target_time or (now + timedelta(hours=2))

        # 1. Check Synchronized CalendarEventRecord in Database
        if db is not None:
            try:
                sync_query = db.query(CalendarEventRecord).filter(
                    CalendarEventRecord.user_id == user_id,
                    CalendarEventRecord.status == "confirmed"
                )
                if event_id and str(event_id).isdigit():
                    cal_rec = sync_query.filter(CalendarEventRecord.id == int(event_id)).first()
                elif event_name:
                    cal_rec = sync_query.filter(CalendarEventRecord.title.ilike(f"%{event_name}%")).first()
                else:
                    # Next upcoming event requiring departure planning
                    now_iso = now.isoformat()
                    cal_rec = sync_query.filter(
                        CalendarEventRecord.start_time >= now_iso,
                        CalendarEventRecord.requires_departure_planning == True
                    ).order_by(CalendarEventRecord.start_time.asc()).first()

                if cal_rec:
                    start_dt = datetime.fromisoformat(cal_rec.start_time)
                    prep_duration = 30.0
                    travel_duration = 20.0
                    importance = "medium"

                    # If user associated a routine template, inherit prep & travel baselines
                    if cal_rec.associated_routine_id:
                        routine = db.query(Event).filter(Event.id == cal_rec.associated_routine_id).first()
                        if routine:
                            prep_duration = float(routine.estimated_prep_duration)
                            travel_duration = float(routine.estimated_travel_duration)
                            importance = routine.importance

                    if cal_rec.prep_buffer_override_min is not None:
                        prep_duration = float(cal_rec.prep_buffer_override_min)

                    provider_name = cal_rec.connection.provider if cal_rec.connection else "calendar"
                    meta = ExternalObservationMetadata(
                        source=f"synced_calendar:{provider_name}",
                        applicable_id=str(cal_rec.id),
                        observation_type=ObservationType.LIVE,
                        confidence=0.95,
                        is_fallback=False,
                        details={
                            "provider": provider_name,
                            "travel_requirement": cal_rec.travel_requirement,
                            "missing_location_warning": cal_rec.missing_location_warning,
                            "provider_event_id": cal_rec.provider_event_id
                        }
                    )
                    return {
                        "event_id": str(cal_rec.id),
                        "event_name": cal_rec.title,
                        "start_time": start_dt,
                        "location": cal_rec.location,
                        "destination": cal_rec.location,
                        "importance": importance,
                        "estimated_prep_duration": prep_duration,
                        "estimated_travel_duration": travel_duration,
                        "travel_mode": cal_rec.travel_mode_override or "driving",
                        "source": f"synced_calendar:{provider_name}",
                        "is_fallback": False,
                        "metadata": meta
                    }
            except Exception as e:
                logger.warning("Error querying CalendarEventRecord for user_id=%s: %s", user_id, e)

        # 2. Live Calendar Integration Adapter
        calendar_integration: Optional[CalendarIntegrationBase] = self.registry.get("calendar")
        if calendar_integration and calendar_integration.is_enabled():
            try:
                if event_id:
                    ext_event = calendar_integration.get_event_by_id(user_id, str(event_id))
                    if ext_event:
                        return {
                            "event_id": str(ext_event.event_id),
                            "event_name": ext_event.title,
                            "start_time": ext_event.start_time,
                            "location": ext_event.location,
                            "destination": ext_event.destination or ext_event.location,
                            "importance": ext_event.importance,
                            "source": f"external_calendar:{calendar_integration.provider_name}",
                            "is_fallback": False,
                            "metadata": ext_event.metadata
                        }
                # Search upcoming
                upcoming = calendar_integration.get_upcoming_events(
                    user_id=user_id,
                    start_time=target - timedelta(hours=1),
                    end_time=target + timedelta(hours=6),
                    limit=5
                )
                if upcoming:
                    selected = upcoming[0]
                    if event_name:
                        for ev in upcoming:
                            if event_name.lower() in ev.title.lower():
                                selected = ev
                                break
                    return {
                        "event_id": str(selected.event_id),
                        "event_name": selected.title,
                        "start_time": selected.start_time,
                        "location": selected.location,
                        "destination": selected.destination or selected.location,
                        "importance": selected.importance,
                        "source": f"external_calendar:{calendar_integration.provider_name}",
                        "is_fallback": False,
                        "metadata": selected.metadata
                    }
            except (IntegrationTimeoutError, IntegrationAuthError, IntegrationError) as e:
                logger.warning("External calendar integration failed for user_id=%s: %s", user_id, e)

        # 3. Graceful Fallback: Routine Template (`Event` table)
        if db is not None:
            try:
                query = db.query(Event).filter(Event.user_id == user_id)
                if event_id and str(event_id).isdigit():
                    db_event = query.filter(Event.id == int(event_id)).first()
                elif event_name:
                    db_event = query.filter(Event.name.ilike(f"%{event_name}%")).first()
                else:
                    db_event = query.first()

                if db_event:
                    start_dt = self._combine_date_and_time(target, db_event.start_time)
                    meta = ExternalObservationMetadata(
                        source="database_routine_template",
                        applicable_id=str(db_event.id),
                        observation_type=ObservationType.HISTORICAL,
                        confidence=0.85,
                        is_fallback=True,
                        details={"routine_name": db_event.name, "day_of_week": db_event.day_of_week}
                    )
                    return {
                        "event_id": str(db_event.id),
                        "event_name": db_event.name,
                        "start_time": start_dt,
                        "location": None,
                        "destination": None,
                        "importance": db_event.importance,
                        "estimated_prep_duration": float(db_event.estimated_prep_duration),
                        "estimated_travel_duration": float(db_event.estimated_travel_duration),
                        "source": "database_routine_template",
                        "is_fallback": True,
                        "metadata": meta
                    }
            except Exception as e:
                logger.warning("Error querying DB routine templates for user_id=%s: %s", user_id, e)

        # 4. Static Heuristic Default
        meta = ExternalObservationMetadata(
            source="heuristic_default",
            applicable_id=str(event_id or "default"),
            observation_type=ObservationType.ESTIMATED,
            confidence=0.5,
            is_fallback=True,
            details={"note": "Calendar unavailable and no matching database routine found"}
        )
        return {
            "event_id": str(event_id or "default_event"),
            "event_name": event_name or "Scheduled Departure",
            "start_time": target,
            "location": None,
            "destination": None,
            "importance": "medium",
            "estimated_prep_duration": 30.0,
            "estimated_travel_duration": 20.0,
            "source": "heuristic_default",
            "is_fallback": True,
            "metadata": meta
        }

    # -------------------------------------------------------------------------
    # Connection Management & Synchronized Events CRUD
    # -------------------------------------------------------------------------

    def list_connections(self, user_id: int, db: Session) -> List[Dict[str, Any]]:
        """List user's calendar connections without exposing encrypted tokens."""
        connections = db.query(CalendarConnection).filter(
            CalendarConnection.user_id == user_id
        ).all()

        results = []
        for c in connections:
            synced_count = db.query(CalendarEventRecord).filter(
                CalendarEventRecord.connection_id == c.id,
                CalendarEventRecord.status != "cancelled"
            ).count()

            departure_count = db.query(CalendarEventRecord).filter(
                CalendarEventRecord.connection_id == c.id,
                CalendarEventRecord.requires_departure_planning == True,
                CalendarEventRecord.status != "cancelled"
            ).count()

            results.append({
                "id": c.id,
                "provider": c.provider,
                "provider_account_reference": c.provider_account_reference,
                "connection_status": c.connection_status,
                "connected_at": c.connected_at,
                "last_synced_at": c.last_synced_at,
                "synced_event_count": synced_count,
                "departure_event_count": departure_count
            })
        return results

    def list_events(
        self,
        user_id: int,
        departure_only: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Query synchronized calendar events for the user."""
        if db is None:
            return []

        query = db.query(CalendarEventRecord).filter(
            CalendarEventRecord.user_id == user_id,
            CalendarEventRecord.status != "cancelled"
        )

        if departure_only:
            query = query.filter(CalendarEventRecord.requires_departure_planning == True)

        if start_date:
            query = query.filter(CalendarEventRecord.start_time >= start_date)
        if end_date:
            query = query.filter(CalendarEventRecord.start_time <= end_date)

        records = query.order_by(CalendarEventRecord.start_time.asc()).limit(limit).all()

        return [
            {
                "id": r.id,
                "provider_event_id": r.provider_event_id,
                "connection_id": r.connection_id,
                "provider": r.connection.provider if r.connection else "calendar",
                "title": r.title,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "location": r.location,
                "calendar_timezone": r.calendar_timezone,
                "is_all_day": r.is_all_day,
                "status": r.status,
                "travel_requirement": r.travel_requirement,
                "requires_departure_planning": r.requires_departure_planning,
                "missing_location_warning": r.missing_location_warning,
                "online_meeting_url": r.online_meeting_url,
                "associated_routine_id": r.associated_routine_id,
                "prep_buffer_override_min": r.prep_buffer_override_min,
                "travel_mode_override": r.travel_mode_override,
                "synced_at": r.synced_at
            }
            for r in records
        ]

    def sync_connection(
        self,
        connection: CalendarConnection,
        db: Session,
        full_resync: bool = False
    ) -> Dict[str, Any]:
        """Synchronize events for any calendar provider (Google or Outlook)."""
        if connection.provider == "outlook":
            return self.outlook_integration.sync_calendar_events(connection, db, full_resync=full_resync)
        else:
            return self.google_integration.sync_calendar_events(connection, db, full_resync=full_resync)

    def disconnect_connection(
        self,
        connection: CalendarConnection,
        db: Session
    ) -> bool:
        """Disconnect any calendar provider (Google or Outlook) and clean up events."""
        if connection.provider == "outlook":
            return self.outlook_integration.disconnect_and_revoke(connection, db)
        else:
            return self.google_integration.disconnect_and_revoke(connection, db)

    def update_event_classification(
        self,
        event_id: int,
        user_id: int,
        travel_requirement: Optional[str] = None,
        associated_routine_id: Optional[int] = None,
        prep_buffer_override_min: Optional[float] = None,
        travel_mode_override: Optional[str] = None,
        location: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Allow user to classify event or link to an existing routine template."""
        if db is None:
            raise ValueError("Database session required.")

        rec = db.query(CalendarEventRecord).filter(
            CalendarEventRecord.id == event_id,
            CalendarEventRecord.user_id == user_id
        ).first()

        if not rec:
            raise ValueError(f"Calendar event '{event_id}' not found.")

        if travel_requirement is not None:
            valid_reqs = ["physical", "online", "personal", "excluded", "unknown"]
            if travel_requirement not in valid_reqs:
                raise ValueError(f"Invalid travel_requirement '{travel_requirement}'. Must be one of {valid_reqs}")
            rec.travel_requirement = travel_requirement
            rec.requires_departure_planning = (travel_requirement == "physical")

        if location is not None:
            rec.location = location.strip() if location else None
            rec.missing_location_warning = bool(not rec.location and rec.travel_requirement == "physical")

        if associated_routine_id is not None:
            if associated_routine_id == 0:
                rec.associated_routine_id = None
            else:
                routine = db.query(Event).filter(
                    Event.id == associated_routine_id,
                    Event.user_id == user_id
                ).first()
                if not routine:
                    raise ValueError(f"Routine '{associated_routine_id}' not found for user.")
                rec.associated_routine_id = routine.id

        if prep_buffer_override_min is not None:
            rec.prep_buffer_override_min = prep_buffer_override_min

        if travel_mode_override is not None:
            rec.travel_mode_override = travel_mode_override

        rec.updated_at = datetime.now(timezone.utc).isoformat()
        db.commit()
        db.refresh(rec)

        return {
            "id": rec.id,
            "title": rec.title,
            "travel_requirement": rec.travel_requirement,
            "requires_departure_planning": rec.requires_departure_planning,
            "missing_location_warning": rec.missing_location_warning,
            "location": rec.location,
            "associated_routine_id": rec.associated_routine_id,
            "prep_buffer_override_min": rec.prep_buffer_override_min,
            "travel_mode_override": rec.travel_mode_override
        }

    @staticmethod
    def _combine_date_and_time(date_dt: datetime, time_str: str) -> datetime:
        try:
            parts = [int(p) for p in time_str.split(":")]
            hour = parts[0]
            minute = parts[1] if len(parts) > 1 else 0
            second = parts[2] if len(parts) > 2 else 0
            return date_dt.replace(hour=hour, minute=minute, second=second, microsecond=0)
        except Exception:
            return date_dt
