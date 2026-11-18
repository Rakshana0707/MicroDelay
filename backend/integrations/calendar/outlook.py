"""Microsoft Outlook / Office 365 Calendar integration adapter with OAuth 2.0 PKCE and Delta Sync.

Implements Microsoft Graph API v1.0 OAuth 2.0 authorization code flow with PKCE,
AES-128-CBC + HMAC-SHA256 token encryption at rest, automatic token refresh,
recurring event expansion, timezone normalization, delta synchronization, and event filtering.
"""

import os
import json
import time
import logging
import urllib.parse
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import httpx
from sqlalchemy.orm import Session

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

logger = logging.getLogger("microdelay.calendar.outlook")

# Minimum required read-only calendar scopes for Microsoft Graph
MICROSOFT_CALENDAR_SCOPES = [
    "Calendars.Read",
    "User.Read",
    "offline_access"
]

MICROSOFT_DEFAULT_TENANT = "common"
MICROSOFT_AUTH_BASE = "https://login.microsoftonline.com"
MICROSOFT_GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class OutlookCalendarIntegration(CalendarIntegrationBase):
    """Production-conscious Microsoft Outlook / Office 365 Calendar integration adapter."""

    def __init__(
        self,
        enabled: bool = True,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: int = 300
    ):
        super().__init__(
            provider_name="outlook_calendar",
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds
        )
        self.credential_mgr = CredentialManager.get_instance()
        self.crypto = OAuthTokenCrypto.get_instance()
        self.state_store = OAuthStateStore.get_instance()
        self._simulated_events: Dict[str, List[Dict[str, Any]]] = {}

    # -------------------------------------------------------------------------
    # Client Configuration & Endpoints
    # -------------------------------------------------------------------------

    def get_client_config(self) -> Dict[str, str]:
        """Retrieve Microsoft OAuth Client ID, Secret, and Tenant from secure manager or env."""
        client_id = (
            self.credential_mgr.get_credential("outlook_calendar", "CLIENT_ID")
            or os.getenv("MICROSOFT_CLIENT_ID")
            or "dummy-outlook-client-id"
        )
        client_secret = (
            self.credential_mgr.get_credential("outlook_calendar", "CLIENT_SECRET")
            or os.getenv("MICROSOFT_CLIENT_SECRET")
            or "dummy-outlook-client-secret"
        )
        tenant_id = (
            self.credential_mgr.get_credential("outlook_calendar", "TENANT_ID")
            or os.getenv("MICROSOFT_TENANT_ID")
            or MICROSOFT_DEFAULT_TENANT
        )
        return {
            "client_id": client_id,
            "client_secret": client_secret,
            "tenant_id": tenant_id,
            "auth_url": f"{MICROSOFT_AUTH_BASE}/{tenant_id}/oauth2/v2.0/authorize",
            "token_url": f"{MICROSOFT_AUTH_BASE}/{tenant_id}/oauth2/v2.0/token"
        }

    def is_authenticated(self, user_id: int) -> bool:
        """Check if Microsoft Graph token or credentials exist for user."""
        has_client = self.credential_mgr.has_credential("outlook_calendar", "CLIENT_ID") or bool(os.getenv("MICROSOFT_CLIENT_ID"))
        has_token = self.credential_mgr.get_token("outlook_calendar", str(user_id)) is not None
        return bool(has_token or has_client)

    def health_check(self) -> IntegrationHealth:
        """Health check for Microsoft Outlook Calendar integration."""
        if not self.enabled:
            return IntegrationHealth(
                service_name=self.service_name,
                provider_name=self.provider_name,
                status=IntegrationStatus.DISABLED,
                message="Microsoft Outlook Calendar integration disabled"
            )

        config = self.get_client_config()
        if not config["client_id"] or config["client_id"] == "dummy-outlook-client-id":
            return IntegrationHealth(
                service_name=self.service_name,
                provider_name=self.provider_name,
                status=IntegrationStatus.DEGRADED,
                message="Outlook Calendar missing MS Graph client credentials; fallback enabled"
            )

        return IntegrationHealth(
            service_name=self.service_name,
            provider_name=self.provider_name,
            status=IntegrationStatus.HEALTHY,
            message="Microsoft Outlook Calendar integration operational"
        )

    # -------------------------------------------------------------------------
    # OAuth 2.0 PKCE Flow
    # -------------------------------------------------------------------------

    def generate_authorization_url(
        self,
        user_id: int,
        redirect_uri: str
    ) -> Tuple[str, str]:
        """Generate Microsoft OAuth 2.0 authorization URL with PKCE and anti-CSRF state."""
        verifier, challenge = generate_pkce_pair()
        state = self.state_store.generate_state(
            user_id=user_id,
            redirect_uri=redirect_uri,
            code_verifier=verifier
        )

        config = self.get_client_config()
        params = {
            "client_id": config["client_id"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "response_mode": "query",
            "scope": " ".join(MICROSOFT_CALENDAR_SCOPES),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "prompt": "select_account"
        }

        query_str = urllib.parse.urlencode(params)
        auth_url = f"{config['auth_url']}?{query_str}"

        logger.info(
            "Generated Microsoft OAuth URL for user_id=%s, state=%s",
            user_id, state[:8] + "..."
        )
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
        is_valid, verifier, stored_redirect_uri = self.state_store.validate_and_consume(
            state, expected_user_id=user_id
        )
        if not is_valid:
            logger.warning("Invalid or expired OAuth state token during callback for user_id=%s", user_id)
            raise IntegrationAuthError(
                "Invalid or expired OAuth state token. Possible CSRF attempt.",
                service=self.provider_name
            )

        effective_redirect = redirect_uri or stored_redirect_uri or "http://localhost:5173/calendar/callback"
        config = self.get_client_config()

        token_req_data = {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": effective_redirect,
            "scope": " ".join(MICROSOFT_CALENDAR_SCOPES)
        }
        if verifier:
            token_req_data["code_verifier"] = verifier

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                token_resp = client.post(
                    config["token_url"],
                    data=token_req_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                if token_resp.status_code != 200:
                    logger.error(
                        "Microsoft token endpoint error (%d): %s",
                        token_resp.status_code, token_resp.text
                    )
                    raise IntegrationAuthError(
                        f"Microsoft OAuth token exchange failed: {token_resp.text}",
                        service=self.provider_name
                    )
                token_data_raw = token_resp.json()
        except IntegrationAuthError:
            raise
        except Exception as e:
            logger.error("Network error during Microsoft token exchange: %s", e)
            raise IntegrationAuthError(f"OAuth token exchange request failed: {str(e)}", service=self.provider_name)

        access_token = token_data_raw.get("access_token")
        refresh_token = token_data_raw.get("refresh_token")
        expires_in = int(token_data_raw.get("expires_in", 3600))
        now_utc = datetime.now(timezone.utc)
        expiry_dt = now_utc + timedelta(seconds=expires_in)

        # Retrieve user email or principal name from Microsoft Graph
        account_ref = self._fetch_user_profile(access_token)

        # Encrypt token payload
        token_payload = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": token_data_raw.get("token_type", "Bearer"),
            "scope": token_data_raw.get("scope", " ".join(MICROSOFT_CALENDAR_SCOPES)),
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "tenant_id": config["tenant_id"],
            "expiry": expiry_dt.isoformat()
        }
        encrypted_tokens = self.crypto.encrypt_tokens(token_payload)

        if db is not None:
            # Upsert user's Outlook calendar connection
            conn = db.query(CalendarConnection).filter(
                CalendarConnection.user_id == user_id,
                CalendarConnection.provider == "outlook"
            ).first()

            now_iso = now_utc.isoformat()
            if not conn:
                conn = CalendarConnection(
                    user_id=user_id,
                    provider="outlook",
                    provider_account_reference=account_ref,
                    encrypted_tokens=encrypted_tokens,
                    connection_status="connected",
                    connected_at=now_iso,
                    granted_scopes=json.dumps(MICROSOFT_CALENDAR_SCOPES)
                )
                db.add(conn)
            else:
                conn.provider_account_reference = account_ref
                conn.encrypted_tokens = encrypted_tokens
                conn.connection_status = "connected"
                conn.connected_at = now_iso
                conn.granted_scopes = json.dumps(MICROSOFT_CALENDAR_SCOPES)

            db.commit()
            db.refresh(conn)
            logger.info("Successfully established Outlook Calendar connection id=%s for user_id=%s", conn.id, user_id)
            return conn

        # In-memory return fallback
        return CalendarConnection(
            id=2,
            user_id=user_id,
            provider="outlook",
            provider_account_reference=account_ref,
            encrypted_tokens=encrypted_tokens,
            connection_status="connected",
            connected_at=datetime.now(timezone.utc).isoformat()
        )

    def _fetch_user_profile(self, access_token: str) -> str:
        """Fetch user email/userPrincipalName from Microsoft Graph /me endpoint."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{MICROSOFT_GRAPH_BASE}/me",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                if resp.status_code == 200:
                    info = resp.json()
                    return str(
                        info.get("mail")
                        or info.get("userPrincipalName")
                        or info.get("displayName")
                        or "connected_outlook_account"
                    )
        except Exception as e:
            logger.warning("Could not fetch Microsoft user profile: %s", e)
        return "connected_outlook_user"

    # -------------------------------------------------------------------------
    # Token Decryption & Auto-Refresh
    # -------------------------------------------------------------------------

    def get_valid_access_token(
        self,
        connection: CalendarConnection,
        db: Optional[Session] = None
    ) -> str:
        """Decrypt credentials and automatically refresh access token if expired."""
        token_data = self.crypto.decrypt_tokens(connection.encrypted_tokens)
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expiry_str = token_data.get("expiry")

        now = datetime.now(timezone.utc)
        is_expired = False
        if expiry_str:
            try:
                expiry_dt = datetime.fromisoformat(expiry_str)
                if not expiry_dt.tzinfo:
                    expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                is_expired = now >= (expiry_dt - timedelta(seconds=60))
            except Exception:
                is_expired = True
        else:
            is_expired = False

        if (is_expired or not access_token) and refresh_token:
            logger.info("Outlook Calendar token expired; initiating refresh for connection_id=%s", connection.id)
            config = self.get_client_config()
            client_id = token_data.get("client_id") or config["client_id"]
            client_secret = token_data.get("client_secret") or config["client_secret"]
            token_url = config["token_url"]

            refresh_payload = {
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": " ".join(MICROSOFT_CALENDAR_SCOPES)
            }

            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(
                        token_url,
                        data=refresh_payload,
                        headers={"Content-Type": "application/x-www-form-urlencoded"}
                    )
                    if resp.status_code != 200:
                        logger.error("Token refresh failed (%d): %s", resp.status_code, resp.text)
                        connection.connection_status = "expired"
                        if db is not None:
                            db.commit()
                        raise IntegrationAuthError(
                            "Outlook Calendar token expired and refresh failed. Please reconnect.",
                            service=self.provider_name
                        )

                    new_data = resp.json()
                    access_token = new_data["access_token"]
                    token_data["access_token"] = access_token
                    if "refresh_token" in new_data:
                        token_data["refresh_token"] = new_data["refresh_token"]
                    expires_in = int(new_data.get("expires_in", 3600))
                    token_data["expiry"] = (now + timedelta(seconds=expires_in)).isoformat()

                    connection.encrypted_tokens = self.crypto.encrypt_tokens(token_data)
                    connection.connection_status = "connected"
                    if db is not None:
                        db.commit()
                    logger.info("Successfully refreshed Outlook Calendar token for connection_id=%s", connection.id)
            except IntegrationAuthError:
                raise
            except Exception as e:
                logger.error("Exception during Outlook token refresh: %s", e)
                connection.connection_status = "expired"
                if db is not None:
                    db.commit()
                raise IntegrationAuthError(
                    f"Outlook Calendar refresh network error: {str(e)}",
                    service=self.provider_name
                )

        return access_token or ""

    # -------------------------------------------------------------------------
    # Disconnect & Token Cleanup
    # -------------------------------------------------------------------------

    def disconnect_and_revoke(
        self,
        connection: CalendarConnection,
        db: Session
    ) -> bool:
        """Mark Outlook connection disconnected and remove synchronized events."""
        connection.connection_status = "disconnected"
        connection.sync_token = None

        deleted_count = db.query(CalendarEventRecord).filter(
            CalendarEventRecord.connection_id == connection.id
        ).delete()

        db.commit()
        logger.info(
            "Disconnected Outlook Calendar connection id=%s; removed %d synced events.",
            connection.id, deleted_count
        )
        return True

    # -------------------------------------------------------------------------
    # Event Synchronization with Microsoft Graph Delta Queries
    # -------------------------------------------------------------------------

    def sync_calendar_events(
        self,
        connection: CalendarConnection,
        db: Session,
        full_resync: bool = False
    ) -> Dict[str, Any]:
        """Perform initial or incremental delta synchronization with Microsoft Graph calendarView/delta."""
        start_time = time.perf_counter()
        access_token = self.get_valid_access_token(connection, db)

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Prefer": 'outlook.timezone="UTC"',
            "Accept": "application/json"
        }

        use_incremental = bool(connection.sync_token and not full_resync)
        now_utc = datetime.now(timezone.utc)

        # Build initial request URL
        if use_incremental:
            # connection.sync_token contains either full deltaLink or deltaToken
            if connection.sync_token.startswith("http"):
                request_url = connection.sync_token
            else:
                request_url = f"{MICROSOFT_GRAPH_BASE}/me/calendarView/delta?$deltatoken={connection.sync_token}"
        else:
            time_min = (now_utc - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
            time_max = (now_utc + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            fields = "id,subject,start,end,location,isAllDay,isCancelled,recurrence,seriesMasterId,isOnlineMeeting,onlineMeeting,onlineMeetingUrl"
            request_url = (
                f"{MICROSOFT_GRAPH_BASE}/me/calendarView/delta"
                f"?startDateTime={time_min}&endDateTime={time_max}&$select={fields}"
            )

        events_processed = 0
        added_count = 0
        updated_count = 0
        cancelled_count = 0
        next_delta_link = None

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                while request_url:
                    resp = client.get(request_url, headers=headers)

                    # Handle expired delta token (HTTP 410 or resyncRequired error)
                    if resp.status_code == 410 or (
                        resp.status_code >= 400
                        and any(k in resp.text for k in ["resyncRequired", "syncStateNotFound", "InvalidDeltaToken"])
                    ):
                        logger.warning(
                            "Microsoft Graph delta token expired or invalid (status=%d). Performing full resync.",
                            resp.status_code
                        )
                        connection.sync_token = None
                        db.commit()
                        return self.sync_calendar_events(connection, db, full_resync=True)

                    if resp.status_code != 200:
                        logger.error("Microsoft Graph error %d: %s", resp.status_code, resp.text)
                        raise IntegrationError(
                            f"Microsoft Graph API request failed with status {resp.status_code}: {resp.text}",
                            service=self.provider_name
                        )

                    payload = resp.json()
                    items = payload.get("value", [])

                    for item in items:
                        events_processed += 1
                        event_id = item["id"]

                        # 1. Handle deleted events via Microsoft Graph @removed indicator
                        if "@removed" in item:
                            rec = db.query(CalendarEventRecord).filter(
                                CalendarEventRecord.connection_id == connection.id,
                                CalendarEventRecord.provider_event_id == event_id
                            ).first()
                            if rec:
                                rec.status = "cancelled"
                                rec.requires_departure_planning = False
                                rec.updated_at = now_utc.isoformat()
                                cancelled_count += 1
                            continue

                        # 2. Parse event properties
                        subject = item.get("subject") or "Untitled Event"
                        is_cancelled = bool(item.get("isCancelled", False))
                        status = "cancelled" if is_cancelled else "confirmed"

                        start_data = item.get("start", {})
                        end_data = item.get("end", {})
                        is_all_day = bool(item.get("isAllDay", False))

                        start_iso = start_data.get("dateTime", "")
                        end_iso = end_data.get("dateTime", "")
                        cal_tz = start_data.get("timeZone") or "UTC"

                        # Ensure valid ISO format with UTC indicator if missing
                        if start_iso and not start_iso.endswith("Z") and "+" not in start_iso:
                            start_iso = f"{start_iso}Z"
                        if end_iso and not end_iso.endswith("Z") and "+" not in end_iso:
                            end_iso = f"{end_iso}Z"

                        # Location parsing (Graph returns dict or string)
                        raw_location = item.get("location")
                        location_name = None
                        if isinstance(raw_location, dict):
                            location_name = raw_location.get("displayName")
                            if not location_name and raw_location.get("address"):
                                location_name = raw_location["address"].get("street")
                        elif isinstance(raw_location, str):
                            location_name = raw_location

                        # Online meeting indicators
                        is_online_meeting = bool(item.get("isOnlineMeeting", False))
                        online_url = item.get("onlineMeetingUrl")
                        if not online_url and isinstance(item.get("onlineMeeting"), dict):
                            online_url = item.get("onlineMeeting", {}).get("joinUrl")

                        # Recurrence information
                        recurring_event_id = item.get("seriesMasterId")
                        recurrence_rule = str(item.get("recurrence")) if item.get("recurrence") else None

                        # Combine clues for filtering
                        combined_text = f"{subject} {location_name or ''}"
                        if online_url:
                            combined_text += f" {online_url}"
                        if is_online_meeting and not online_url:
                            combined_text += " teams meeting"

                        # Classify event requirements
                        classification = EventFilterService.classify_event(
                            title=subject,
                            location=location_name,
                            status=status,
                            is_all_day=is_all_day,
                            description=online_url or ( "teams meeting" if is_online_meeting else None )
                        )

                        if is_online_meeting:
                            classification.travel_requirement = "online"
                            classification.requires_departure_planning = False
                            if online_url:
                                classification.online_meeting_url = online_url

                        # Idempotent upsert in CalendarEventRecord
                        existing_rec = db.query(CalendarEventRecord).filter(
                            CalendarEventRecord.connection_id == connection.id,
                            CalendarEventRecord.provider_event_id == event_id
                        ).first()

                        if existing_rec:
                            existing_rec.title = subject
                            existing_rec.start_time = start_iso
                            existing_rec.end_time = end_iso
                            existing_rec.location = classification.sanitized_location
                            existing_rec.calendar_timezone = cal_tz
                            existing_rec.is_all_day = is_all_day
                            existing_rec.status = status
                            existing_rec.recurrence_rule = recurrence_rule
                            existing_rec.recurring_event_id = recurring_event_id
                            existing_rec.online_meeting_url = classification.online_meeting_url
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
                                provider_event_id=event_id,
                                title=subject,
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

                    # Check pagination
                    request_url = payload.get("@odata.nextLink")
                    if not request_url:
                        next_delta_link = payload.get("@odata.deltaLink")
                        break

            # Persist deltaLink for next incremental sync
            if next_delta_link:
                connection.sync_token = next_delta_link

            connection.last_synced_at = now_utc.isoformat()
            db.commit()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "Completed Outlook Calendar sync for connection_id=%s: %d added, %d updated, %d cancelled (%.1f ms)",
                connection.id, added_count, updated_count, cancelled_count, latency_ms
            )

            return {
                "status": "success",
                "connection_id": connection.id,
                "provider": "outlook",
                "sync_mode": "incremental" if use_incremental else "full",
                "total_processed": events_processed,
                "added": added_count,
                "updated": updated_count,
                "cancelled": cancelled_count,
                "duration_ms": round(latency_ms, 2)
            }

        except (IntegrationAuthError, IntegrationError):
            raise
        except Exception as e:
            logger.error("Error during Outlook Calendar synchronization: %s", e)
            raise IntegrationError(f"Outlook synchronization error: {str(e)}", service=self.provider_name)

    # -------------------------------------------------------------------------
    # CalendarIntegrationBase Abstract Implementations
    # -------------------------------------------------------------------------

    def get_upcoming_events(
        self,
        user_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10
    ) -> List[CalendarEvent]:
        """Fetch upcoming events for user (supporting simulated events and fallback)."""
        start_req = time.perf_counter()
        if not self.enabled:
            return []

        latency = (time.perf_counter() - start_req) * 1000.0
        user_events = self._simulated_events.get(str(user_id), [])
        results: List[CalendarEvent] = []

        for raw in user_events[:limit]:
            meta = self.create_metadata(
                applicable_id=raw["event_id"],
                observation_type=ObservationType.LIVE,
                latency_ms=latency,
                details={"provider": "outlook_calendar"}
            )
            ev = CalendarEvent(
                event_id=raw["event_id"],
                title=raw["title"],
                start_time=raw["start_time"],
                end_time=raw["end_time"],
                location=raw.get("location"),
                destination=raw.get("destination", raw.get("location")),
                description=raw.get("description"),
                importance=raw.get("importance", "medium"),
                is_all_day=raw.get("is_all_day", False),
                metadata=meta
            )
            results.append(ev)

        return results

    def get_event_by_id(
        self,
        user_id: int,
        event_id: str
    ) -> Optional[CalendarEvent]:
        events = self.get_upcoming_events(user_id, limit=50)
        for ev in events:
            if ev.event_id == str(event_id):
                return ev
        return None

    def add_simulated_event(self, user_id: int, event_data: Dict[str, Any]) -> None:
        uid = str(user_id)
        if uid not in self._simulated_events:
            self._simulated_events[uid] = []
        self._simulated_events[uid].append(event_data)
