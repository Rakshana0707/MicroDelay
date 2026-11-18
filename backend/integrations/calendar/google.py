"""Google Calendar Integration with OAuth 2.0 PKCE, Incremental Sync, and Event Filtering.

Implements official Google OAuth 2.0 flow, token encryption at rest, token refresh,
token revocation, recurring event expansion, timezone handling, and incremental sync.
"""

import os
import json
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import httpx
from sqlalchemy.orm import Session

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import googleapiclient.discovery
from googleapiclient.errors import HttpError

try:
    from database import CalendarConnection, CalendarEventRecord, Event
    from integrations.base import (
        ObservationType, IntegrationStatus, IntegrationHealth,
        IntegrationAuthError, IntegrationTimeoutError, IntegrationError
    )
    from integrations.credentials import CredentialManager
    from integrations.calendar.base import CalendarIntegrationBase, CalendarEvent
    from integrations.calendar.crypto import OAuthTokenCrypto, OAuthStateStore, generate_pkce_pair
    from integrations.calendar.filter import EventFilterService
except ImportError:
    from backend.database import CalendarConnection, CalendarEventRecord, Event
    from backend.integrations.base import (
        ObservationType, IntegrationStatus, IntegrationHealth,
        IntegrationAuthError, IntegrationTimeoutError, IntegrationError
    )
    from backend.integrations.credentials import CredentialManager
    from backend.integrations.calendar.base import CalendarIntegrationBase, CalendarEvent
    from backend.integrations.calendar.crypto import OAuthTokenCrypto, OAuthStateStore, generate_pkce_pair
    from backend.integrations.calendar.filter import EventFilterService

logger = logging.getLogger("microdelay.calendar.google")

# Minimum required read-only scopes
GOOGLE_CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"


class GoogleCalendarIntegration(CalendarIntegrationBase):
    """Production-conscious Google Calendar integration with OAuth 2.0 and synchronization."""

    def __init__(
        self,
        enabled: bool = True,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: int = 300
    ):
        super().__init__(
            provider_name="google_calendar",
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds
        )
        self.credential_mgr = CredentialManager.get_instance()
        self.crypto = OAuthTokenCrypto.get_instance()
        self.state_store = OAuthStateStore.get_instance()

    # -------------------------------------------------------------------------
    # Credentials & Client Configuration
    # -------------------------------------------------------------------------

    def get_client_config(self) -> Dict[str, Any]:
        """Retrieve Google OAuth Client ID and Secret from secure manager or environment."""
        client_id = self.credential_mgr.get_credential("google_calendar", "CLIENT_ID") or "dummy-google-client-id.apps.googleusercontent.com"
        client_secret = self.credential_mgr.get_credential("google_calendar", "CLIENT_SECRET") or "dummy-google-client-secret"
        
        return {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs"
            }
        }

    # -------------------------------------------------------------------------
    # OAuth 2.0 Authorization & PKCE Flow
    # -------------------------------------------------------------------------

    def generate_authorization_url(
        self,
        user_id: int,
        redirect_uri: str
    ) -> Tuple[str, str]:
        """Generate Google OAuth 2.0 authorization URL with PKCE and CSRF state."""
        verifier, challenge = generate_pkce_pair()
        state = self.state_store.generate_state(
            user_id=user_id,
            redirect_uri=redirect_uri,
            code_verifier=verifier
        )

        client_config = self.get_client_config()
        flow = Flow.from_client_config(
            client_config=client_config,
            scopes=GOOGLE_CALENDAR_SCOPES,
            redirect_uri=redirect_uri
        )

        # Generate authorization URL with offline access to receive a refresh token
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
            code_challenge=challenge,
            code_challenge_method="S256"
        )

        logger.info("Generated Google OAuth URL for user_id=%s, state=%s", user_id, state[:8] + "...")
        return auth_url, state

    def handle_oauth_callback(
        self,
        user_id: int,
        code: str,
        state: str,
        redirect_uri: Optional[str] = None,
        db: Optional[Session] = None
    ) -> CalendarConnection:
        """Exchange authorization code for tokens, validate PKCE/state, and persist connection."""
        is_valid, verifier, stored_redirect_uri = self.state_store.validate_and_consume(state, expected_user_id=user_id)
        if not is_valid:
            logger.warning("Invalid or expired OAuth state token during callback for user_id=%s", user_id)
            raise IntegrationAuthError("Invalid or expired OAuth state token. Possible CSRF attempt.", service=self.provider_name)

        effective_redirect = redirect_uri or stored_redirect_uri or "http://localhost:5173/calendar/callback"

        client_config = self.get_client_config()
        flow = Flow.from_client_config(
            client_config=client_config,
            scopes=GOOGLE_CALENDAR_SCOPES,
            redirect_uri=effective_redirect,
            state=state
        )

        if verifier:
            flow.code_verifier = verifier

        try:
            flow.fetch_token(code=code)
        except Exception as e:
            logger.error("Failed to exchange OAuth code for tokens: %s", e)
            raise IntegrationAuthError(f"OAuth token exchange failed: {str(e)}", service=self.provider_name)

        creds = flow.credentials
        account_ref = self._extract_account_email(creds)

        # Encrypt token payload
        token_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes) if creds.scopes else GOOGLE_CALENDAR_SCOPES,
            "expiry": creds.expiry.isoformat() if creds.expiry else None
        }
        encrypted_tokens = self.crypto.encrypt_tokens(token_data)

        if db is not None:
            # Upsert user's Google calendar connection
            conn = db.query(CalendarConnection).filter(
                CalendarConnection.user_id == user_id,
                CalendarConnection.provider == "google"
            ).first()

            now_iso = datetime.now(timezone.utc).isoformat()
            if not conn:
                conn = CalendarConnection(
                    user_id=user_id,
                    provider="google",
                    provider_account_reference=account_ref,
                    encrypted_tokens=encrypted_tokens,
                    connection_status="connected",
                    connected_at=now_iso,
                    granted_scopes=json.dumps(token_data["scopes"])
                )
                db.add(conn)
            else:
                conn.provider_account_reference = account_ref
                conn.encrypted_tokens = encrypted_tokens
                conn.connection_status = "connected"
                conn.connected_at = now_iso
                conn.granted_scopes = json.dumps(token_data["scopes"])

            db.commit()
            db.refresh(conn)
            logger.info("Successfully established Google Calendar connection id=%s for user_id=%s", conn.id, user_id)
            return conn

        # Fallback in-memory representation
        return CalendarConnection(
            id=1,
            user_id=user_id,
            provider="google",
            provider_account_reference=account_ref,
            encrypted_tokens=encrypted_tokens,
            connection_status="connected",
            connected_at=datetime.now(timezone.utc).isoformat()
        )

    def _extract_account_email(self, creds: Credentials) -> str:
        """Fetch Google email address using userinfo endpoint or ID token."""
        try:
            if hasattr(creds, "id_token") and creds.id_token and isinstance(creds.id_token, dict):
                email = creds.id_token.get("email")
                if email:
                    return str(email)

            # Query Google UserInfo
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {creds.token}"}
                )
                if resp.status_code == 200:
                    info = resp.json()
                    return str(info.get("email") or "connected_google_account")
        except Exception as e:
            logger.warning("Could not fetch Google user email: %s", e)

        return "connected_google_user"

    # -------------------------------------------------------------------------
    # Token Decryption & Auto-Refresh
    # -------------------------------------------------------------------------

    def get_valid_credentials(
        self,
        connection: CalendarConnection,
        db: Optional[Session] = None
    ) -> Credentials:
        """Decrypt credentials and automatically refresh access token if expired."""
        token_data = self.crypto.decrypt_tokens(connection.encrypted_tokens)
        expiry_dt = None
        if token_data.get("expiry"):
            try:
                expiry_dt = datetime.fromisoformat(token_data["expiry"])
            except Exception:
                pass

        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri") or "https://oauth2.googleapis.com/token",
            client_id=token_data.get("client_id") or self.get_client_config()["web"]["client_id"],
            client_secret=token_data.get("client_secret") or self.get_client_config()["web"]["client_secret"],
            scopes=token_data.get("scopes") or GOOGLE_CALENDAR_SCOPES,
            expiry=expiry_dt
        )

        # Check expiry and refresh if needed
        now = datetime.now(timezone.utc)
        is_expired = False
        if creds.expiry:
            target = creds.expiry if creds.expiry.tzinfo else creds.expiry.replace(tzinfo=timezone.utc)
            is_expired = now >= (target - timedelta(seconds=60))

        if is_expired and creds.refresh_token:
            logger.info("Google Calendar access token expired; initiating refresh for connection_id=%s", connection.id)
            try:
                creds.refresh(Request())
                token_data["token"] = creds.token
                token_data["expiry"] = creds.expiry.isoformat() if creds.expiry else None
                connection.encrypted_tokens = self.crypto.encrypt_tokens(token_data)
                connection.connection_status = "connected"
                if db is not None:
                    db.commit()
                logger.info("Successfully refreshed Google Calendar access token for connection_id=%s", connection.id)
            except Exception as e:
                logger.error("Token refresh failed for connection_id=%s: %s", connection.id, e)
                connection.connection_status = "expired"
                if db is not None:
                    db.commit()
                raise IntegrationAuthError("Google Calendar token expired and refresh failed. Please reconnect.", service=self.provider_name)

        return creds

    # -------------------------------------------------------------------------
    # Token Revocation & Disconnect
    # -------------------------------------------------------------------------

    def disconnect_and_revoke(
        self,
        connection: CalendarConnection,
        db: Session
    ) -> bool:
        """Revoke token with Google endpoint and mark connection revoked."""
        token_data = {}
        try:
            token_data = self.crypto.decrypt_tokens(connection.encrypted_tokens)
        except Exception:
            pass

        token_to_revoke = token_data.get("refresh_token") or token_data.get("token")
        if token_to_revoke:
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(
                        GOOGLE_REVOKE_URL,
                        params={"token": token_to_revoke},
                        headers={"Content-Type": "application/x-www-form-urlencoded"}
                    )
                    logger.info("Google OAuth token revocation status: %s", resp.status_code)
            except Exception as e:
                logger.warning("Could not reach Google token revocation endpoint: %s", e)

        # Mark connection revoked and delete synchronized events
        connection.connection_status = "disconnected"
        connection.sync_token = None
        
        # Clean up synchronized calendar events
        deleted_count = db.query(CalendarEventRecord).filter(
            CalendarEventRecord.connection_id == connection.id
        ).delete()

        db.commit()
        logger.info(
            "Disconnected Google Calendar connection id=%s; removed %d synced events.",
            connection.id, deleted_count
        )
        return True

    # -------------------------------------------------------------------------
    # Event Synchronization (Initial & Incremental)
    # -------------------------------------------------------------------------

    def sync_calendar_events(
        self,
        connection: CalendarConnection,
        db: Session,
        full_resync: bool = False
    ) -> Dict[str, Any]:
        """Perform initial or incremental synchronization using Google Calendar sync tokens."""
        start_time = time.perf_counter()
        creds = self.get_valid_credentials(connection, db)

        # Build Google Calendar client
        service = googleapiclient.discovery.build(
            "calendar", "v3", credentials=creds, cache_discovery=False
        )

        use_incremental = bool(connection.sync_token and not full_resync)
        page_token = None
        next_sync_token = None
        events_processed = 0
        added_count = 0
        updated_count = 0
        cancelled_count = 0
        now_utc = datetime.now(timezone.utc)

        try:
            while True:
                req_kwargs: Dict[str, Any] = {
                    "calendarId": "primary",
                    "maxResults": 250,
                    "pageToken": page_token
                }

                if use_incremental:
                    req_kwargs["syncToken"] = connection.sync_token
                else:
                    # Sync window: from 3 days ago up to 30 days ahead
                    time_min = (now_utc - timedelta(days=3)).isoformat()
                    time_max = (now_utc + timedelta(days=30)).isoformat()
                    req_kwargs["timeMin"] = time_min
                    req_kwargs["timeMax"] = time_max
                    req_kwargs["singleEvents"] = True  # Automatically expands recurring events!
                    req_kwargs["orderBy"] = "startTime"

                try:
                    events_result = service.events().list(**req_kwargs).execute()
                except HttpError as err:
                    if err.resp.status == 410:
                        # 410 Gone: sync token is expired or invalid. Fallback to clean full sync!
                        logger.warning("Google Calendar syncToken expired (HTTP 410). Resetting and running full resync.")
                        connection.sync_token = None
                        db.commit()
                        return self.sync_calendar_events(connection, db, full_resync=True)
                    raise

                items = events_result.get("items", [])
                for item in items:
                    events_processed += 1
                    status = item.get("status", "confirmed")
                    provider_event_id = item["id"]

                    # Check for cancelled event
                    if status == "cancelled":
                        rec = db.query(CalendarEventRecord).filter(
                            CalendarEventRecord.connection_id == connection.id,
                            CalendarEventRecord.provider_event_id == provider_event_id
                        ).first()
                        if rec:
                            rec.status = "cancelled"
                            rec.requires_departure_planning = False
                            rec.updated_at = now_utc.isoformat()
                            cancelled_count += 1
                        continue

                    # Parse timestamps
                    start_dict = item.get("start", {})
                    end_dict = item.get("end", {})
                    is_all_day = "date" in start_dict and "dateTime" not in start_dict

                    if is_all_day:
                        start_iso = f"{start_dict['date']}T00:00:00Z"
                        end_iso = f"{end_dict.get('date', start_dict['date'])}T23:59:59Z"
                        cal_tz = "UTC"
                    else:
                        start_iso = start_dict.get("dateTime", "")
                        end_iso = end_dict.get("dateTime", "")
                        cal_tz = start_dict.get("timeZone") or "UTC"

                    title = item.get("summary") or "Untitled Event"
                    raw_loc = item.get("location")
                    recurrence_rule = str(item.get("recurrence", [])) if item.get("recurrence") else None
                    recurring_event_id = item.get("recurringEventId")

                    # Filter and classify event travel requirement
                    classification = EventFilterService.classify_event(
                        title=title,
                        location=raw_loc,
                        status=status,
                        is_all_day=is_all_day
                    )

                    # Upsert CalendarEventRecord idempotently
                    existing_rec = db.query(CalendarEventRecord).filter(
                        CalendarEventRecord.connection_id == connection.id,
                        CalendarEventRecord.provider_event_id == provider_event_id
                    ).first()

                    if existing_rec:
                        # Update fields, preserving user's manual override if set
                        existing_rec.title = title
                        existing_rec.start_time = start_iso
                        existing_rec.end_time = end_iso
                        existing_rec.location = classification.sanitized_location
                        existing_rec.calendar_timezone = cal_tz
                        existing_rec.is_all_day = is_all_day
                        existing_rec.status = status
                        existing_rec.recurrence_rule = recurrence_rule
                        existing_rec.recurring_event_id = recurring_event_id
                        existing_rec.online_meeting_url = classification.online_meeting_url
                        # If user hasn't manually overridden classification, update it
                        if existing_rec.travel_requirement == "unknown" or existing_rec.travel_requirement == classification.travel_requirement:
                            existing_rec.travel_requirement = classification.travel_requirement
                            existing_rec.requires_departure_planning = classification.requires_departure_planning
                            existing_rec.missing_location_warning = classification.missing_location_warning
                        existing_rec.updated_at = now_utc.isoformat()
                        existing_rec.synced_at = now_utc.isoformat()
                        updated_count += 1
                    else:
                        new_rec = CalendarEventRecord(
                            user_id=connection.user_id,
                            connection_id=connection.id,
                            provider_event_id=provider_event_id,
                            title=title,
                            start_time=start_iso,
                            end_time=end_iso,
                            location=classification.sanitized_location,
                            calendar_timezone=cal_tz,
                            is_all_day=is_all_day,
                            status=status,
                            recurrence_rule=recurrence_rule,
                            recurring_event_id=recurring_event_id,
                            travel_requirement=classification.travel_requirement,
                            requires_departure_planning=classification.requires_departure_planning,
                            missing_location_warning=classification.missing_location_warning,
                            online_meeting_url=classification.online_meeting_url,
                            synced_at=now_utc.isoformat(),
                            updated_at=now_utc.isoformat()
                        )
                        db.add(new_rec)
                        added_count += 1

                page_token = events_result.get("nextPageToken")
                next_sync_token = events_result.get("nextSyncToken")
                if not page_token:
                    break

            # Save nextSyncToken for incremental sync
            if next_sync_token:
                connection.sync_token = next_sync_token
            connection.last_synced_at = now_utc.isoformat()
            db.commit()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "Completed Google Calendar sync for connection_id=%s: %d added, %d updated, %d cancelled (%.1f ms)",
                connection.id, added_count, updated_count, cancelled_count, latency_ms
            )

            return {
                "status": "success",
                "connection_id": connection.id,
                "sync_mode": "incremental" if use_incremental else "full",
                "total_processed": events_processed,
                "added": added_count,
                "updated": updated_count,
                "cancelled": cancelled_count,
                "last_synced_at": connection.last_synced_at,
                "latency_ms": round(latency_ms, 1)
            }

        except HttpError as err:
            logger.error("Google Calendar API error during sync: %s", err)
            raise IntegrationError(f"Google Calendar API sync error: {str(err)}", service=self.provider_name)

    # -------------------------------------------------------------------------
    # CalendarIntegrationBase Interface Implementation
    # -------------------------------------------------------------------------

    def get_upcoming_events(
        self,
        user_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        db: Optional[Session] = None
    ) -> List[CalendarEvent]:
        """Fetch synchronized calendar events from local cache/database."""
        if db is None:
            return []

        now_utc = datetime.now(timezone.utc)
        start_iso = (start_time or now_utc).isoformat()
        end_iso = (end_time or (now_utc + timedelta(days=7))).isoformat()

        records = (
            db.query(CalendarEventRecord)
            .filter(
                CalendarEventRecord.user_id == user_id,
                CalendarEventRecord.status == "confirmed",
                CalendarEventRecord.start_time >= start_iso,
                CalendarEventRecord.start_time <= end_iso
            )
            .order_by(CalendarEventRecord.start_time.asc())
            .limit(limit)
            .all()
        )

        events: List[CalendarEvent] = []
        for r in records:
            meta = self.create_metadata(
                applicable_id=str(r.id),
                observation_type=ObservationType.LIVE,
                confidence=0.98,
                details={
                    "travel_requirement": r.travel_requirement,
                    "requires_departure_planning": r.requires_departure_planning,
                    "provider_event_id": r.provider_event_id
                }
            )
            events.append(CalendarEvent(
                event_id=str(r.id),
                title=r.title,
                start_time=datetime.fromisoformat(r.start_time),
                end_time=datetime.fromisoformat(r.end_time),
                location=r.location,
                destination=r.location,
                importance="medium",
                is_all_day=r.is_all_day,
                metadata=meta
            ))

        return events

    def get_event_by_id(
        self,
        user_id: int,
        event_id: str,
        db: Optional[Session] = None
    ) -> Optional[CalendarEvent]:
        if db is None:
            return None
        rec = db.query(CalendarEventRecord).filter(
            CalendarEventRecord.user_id == user_id,
            CalendarEventRecord.id == int(event_id) if str(event_id).isdigit() else False
        ).first()
        if not rec:
            return None

        meta = self.create_metadata(
            applicable_id=str(rec.id),
            observation_type=ObservationType.LIVE,
            details={"travel_requirement": rec.travel_requirement}
        )
        return CalendarEvent(
            event_id=str(rec.id),
            title=rec.title,
            start_time=datetime.fromisoformat(rec.start_time),
            end_time=datetime.fromisoformat(rec.end_time),
            location=rec.location,
            destination=rec.location,
            metadata=meta
        )
