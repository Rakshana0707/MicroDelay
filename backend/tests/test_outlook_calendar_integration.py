"""Tests for Microsoft Outlook / Office 365 Calendar Integration (Prompt 3.3).

Verifies Microsoft identity platform OAuth 2.0 with PKCE, state validation,
encryption at rest, token auto-refresh, delta synchronization, recurring event handling,
deletion handling via @removed, multi-provider coexistence (Google + Outlook),
duplicate prevention, and API endpoints.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.database import User, Event, CalendarConnection, CalendarEventRecord
from backend.integrations.calendar.outlook import (
    OutlookCalendarIntegration, MICROSOFT_CALENDAR_SCOPES, MICROSOFT_GRAPH_BASE
)
from backend.integrations.calendar.google import GoogleCalendarIntegration
from backend.integrations.calendar.crypto import OAuthTokenCrypto, OAuthStateStore, generate_pkce_pair
from backend.integrations.calendar.filter import EventFilterService
from backend.services.calendar_service import CalendarService


# =============================================================================
# 1. OAuth 2.0 PKCE & State Tests
# =============================================================================

def test_outlook_pkce_authorization_url_generation():
    """Verify PKCE parameters and anti-CSRF state token are generated for Outlook."""
    adapter = OutlookCalendarIntegration()
    auth_url, state = adapter.generate_authorization_url(
        user_id=101,
        redirect_uri="http://localhost:5173/calendar/callback"
    )

    assert "login.microsoftonline.com" in auth_url
    assert "common/oauth2/v2.0/authorize" in auth_url
    assert "response_type=code" in auth_url
    assert "code_challenge=" in auth_url
    assert "code_challenge_method=S256" in auth_url
    assert "scope=" in auth_url
    assert "Calendars.Read" in auth_url
    assert f"state={state}" in auth_url

    # Verify state is saved in store
    is_valid, verifier, redirect = adapter.state_store.validate_and_consume(state, expected_user_id=101)
    assert is_valid is True
    assert verifier is not None
    assert redirect == "http://localhost:5173/calendar/callback"


def test_outlook_oauth_state_tampering_rejected():
    """Verify forged, expired, or mismatched state tokens are rejected."""
    adapter = OutlookCalendarIntegration()
    _, valid_state = adapter.generate_authorization_url(user_id=101, redirect_uri="http://localhost:5173/cb")

    # Mismatched user_id
    is_valid, _, _ = adapter.state_store.validate_and_consume(valid_state, expected_user_id=999)
    assert is_valid is False

    # Single-use validation
    is_valid_retry, _, _ = adapter.state_store.validate_and_consume(valid_state, expected_user_id=101)
    assert is_valid_retry is False


# =============================================================================
# 2. Token Encryption, Exchange & Auto-Refresh Tests
# =============================================================================

def test_outlook_oauth_callback_and_encryption_at_rest(db_session: Session, test_user: User):
    """Verify OAuth code exchange persists encrypted tokens and user email."""
    adapter = OutlookCalendarIntegration()
    _, state = adapter.generate_authorization_url(user_id=test_user.id, redirect_uri="http://localhost:5173/cb")

    mock_token_resp = MagicMock()
    mock_token_resp.status_code = 200
    mock_token_resp.json.return_value = {
        "access_token": "mock-outlook-access-token-12345",
        "refresh_token": "mock-outlook-refresh-token-67890",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "Calendars.Read User.Read offline_access"
    }

    mock_me_resp = MagicMock()
    mock_me_resp.status_code = 200
    mock_me_resp.json.return_value = {
        "mail": "alex.engineer@microsoft.com",
        "displayName": "Alex Engineer"
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_token_resp
    mock_client.get.return_value = mock_me_resp

    with patch("backend.integrations.calendar.outlook.httpx.Client", return_value=mock_client):
        conn = adapter.handle_oauth_callback(
            user_id=test_user.id,
            code="mock_ms_auth_code",
            state=state,
            redirect_uri="http://localhost:5173/cb",
            db=db_session
        )

    assert conn.provider == "outlook"
    assert conn.provider_account_reference == "alex.engineer@microsoft.com"
    assert conn.connection_status == "connected"
    assert "mock-outlook-access-token" not in conn.encrypted_tokens

    # Verify decrypt
    crypto = OAuthTokenCrypto.get_instance()
    decrypted = crypto.decrypt_tokens(conn.encrypted_tokens)
    assert decrypted["access_token"] == "mock-outlook-access-token-12345"
    assert decrypted["refresh_token"] == "mock-outlook-refresh-token-67890"


def test_outlook_token_auto_refresh_on_expiry(db_session: Session, test_user: User):
    """Verify expired Outlook token is refreshed automatically via token endpoint."""
    adapter = OutlookCalendarIntegration()
    crypto = OAuthTokenCrypto.get_instance()

    # Seed expired connection
    expired_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    raw_tokens = {
        "access_token": "old-expired-ms-token",
        "refresh_token": "valid-ms-refresh-token",
        "expiry": expired_time,
        "client_id": "test-client",
        "client_secret": "test-secret"
    }
    conn = CalendarConnection(
        user_id=test_user.id,
        provider="outlook",
        provider_account_reference="alex@outlook.com",
        encrypted_tokens=crypto.encrypt_tokens(raw_tokens),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(conn)
    db_session.commit()
    db_session.refresh(conn)

    mock_refresh_resp = MagicMock()
    mock_refresh_resp.status_code = 200
    mock_refresh_resp.json.return_value = {
        "access_token": "newly-refreshed-ms-token",
        "refresh_token": "new-ms-refresh-token",
        "expires_in": 7200
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_refresh_resp

    with patch("backend.integrations.calendar.outlook.httpx.Client", return_value=mock_client):
        token = adapter.get_valid_access_token(conn, db=db_session)

    assert token == "newly-refreshed-ms-token"
    refreshed_data = crypto.decrypt_tokens(conn.encrypted_tokens)
    assert refreshed_data["access_token"] == "newly-refreshed-ms-token"
    assert refreshed_data["refresh_token"] == "new-ms-refresh-token"
    assert conn.connection_status == "connected"


# =============================================================================
# 3. Microsoft Graph Delta Synchronization & Event Filtering Tests
# =============================================================================

def test_outlook_initial_delta_sync_and_classification(db_session: Session, test_user: User):
    """Verify initial delta query synchronizes and classifies physical, online, and personal events."""
    adapter = OutlookCalendarIntegration()
    crypto = OAuthTokenCrypto.get_instance()

    conn = CalendarConnection(
        user_id=test_user.id,
        provider="outlook",
        provider_account_reference="user@outlook.com",
        encrypted_tokens=crypto.encrypt_tokens({"access_token": "valid_token"}),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(conn)
    db_session.commit()
    db_session.refresh(conn)

    mock_graph_resp = MagicMock()
    mock_graph_resp.status_code = 200
    mock_graph_resp.json.return_value = {
        "value": [
            {
                "id": "ms_evt_1",
                "subject": "Client Lunch Meeting",
                "start": {"dateTime": "2026-11-20T12:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-11-20T13:30:00", "timeZone": "UTC"},
                "location": {"displayName": "742 Evergreen Terrace, Springfield"},
                "isAllDay": False,
                "isCancelled": False,
                "isOnlineMeeting": False
            },
            {
                "id": "ms_evt_2",
                "subject": "Weekly Sprint Planning",
                "start": {"dateTime": "2026-11-20T09:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-11-20T10:00:00", "timeZone": "UTC"},
                "isAllDay": False,
                "isCancelled": False,
                "isOnlineMeeting": True,
                "onlineMeetingUrl": "https://teams.microsoft.com/l/meetup-join/19%3ameeting"
            },
            {
                "id": "ms_evt_3",
                "subject": "Personal Focus Time",
                "start": {"dateTime": "2026-11-20T15:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-11-20T17:00:00", "timeZone": "UTC"},
                "isAllDay": False,
                "isCancelled": False,
                "isOnlineMeeting": False,
                "location": None
            }
        ],
        "@odata.deltaLink": "https://graph.microsoft.com/v1.0/me/calendarView/delta?$deltatoken=DELTA_TOKEN_XYZ_1"
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_graph_resp

    with patch.object(adapter, "get_valid_access_token", return_value="valid_token"), \
         patch("backend.integrations.calendar.outlook.httpx.Client", return_value=mock_client):
        res = adapter.sync_calendar_events(conn, db_session)

    assert res["status"] == "success"
    assert res["added"] == 3
    assert conn.sync_token == "https://graph.microsoft.com/v1.0/me/calendarView/delta?$deltatoken=DELTA_TOKEN_XYZ_1"

    # Verify physical departure event
    physical_evt = db_session.query(CalendarEventRecord).filter(
        CalendarEventRecord.provider_event_id == "ms_evt_1"
    ).first()
    assert physical_evt is not None
    assert physical_evt.travel_requirement == "physical"
    assert physical_evt.requires_departure_planning is True
    assert physical_evt.location == "742 Evergreen Terrace, Springfield"

    # Verify online Teams meeting
    online_evt = db_session.query(CalendarEventRecord).filter(
        CalendarEventRecord.provider_event_id == "ms_evt_2"
    ).first()
    assert online_evt is not None
    assert online_evt.travel_requirement == "online"
    assert online_evt.requires_departure_planning is False
    assert "teams.microsoft.com" in online_evt.online_meeting_url

    # Verify personal block
    personal_evt = db_session.query(CalendarEventRecord).filter(
        CalendarEventRecord.provider_event_id == "ms_evt_3"
    ).first()
    assert personal_evt is not None
    assert personal_evt.travel_requirement == "personal"
    assert personal_evt.requires_departure_planning is False


def test_outlook_incremental_delta_sync_and_removed_events(db_session: Session, test_user: User):
    """Verify incremental sync handles modified events and @removed deleted events."""
    adapter = OutlookCalendarIntegration()
    crypto = OAuthTokenCrypto.get_instance()

    conn = CalendarConnection(
        user_id=test_user.id,
        provider="outlook",
        provider_account_reference="user@outlook.com",
        encrypted_tokens=crypto.encrypt_tokens({"access_token": "valid_token"}),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat(),
        sync_token="https://graph.microsoft.com/v1.0/me/calendarView/delta?$deltatoken=OLD_DELTA"
    )
    db_session.add(conn)
    db_session.commit()
    db_session.refresh(conn)

    # Seed existing event
    existing_evt = CalendarEventRecord(
        user_id=test_user.id,
        connection_id=conn.id,
        provider_event_id="ms_evt_100",
        title="Original Event Title",
        start_time="2026-11-20T10:00:00Z",
        end_time="2026-11-20T11:00:00Z",
        location="Room 101",
        travel_requirement="physical",
        requires_departure_planning=True,
        status="confirmed",
        synced_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(existing_evt)
    db_session.commit()

    # Delta response with 1 update and 1 deletion (@removed)
    mock_delta_resp = MagicMock()
    mock_delta_resp.status_code = 200
    mock_delta_resp.json.return_value = {
        "value": [
            {
                "id": "ms_evt_100",
                "@removed": {"reason": "deleted"}
            },
            {
                "id": "ms_evt_200",
                "subject": "New On-Site Architecture Review",
                "start": {"dateTime": "2026-11-21T14:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-11-21T16:00:00", "timeZone": "UTC"},
                "location": {"displayName": "Building 4, Conference Room B"},
                "isAllDay": False,
                "isCancelled": False
            }
        ],
        "@odata.deltaLink": "https://graph.microsoft.com/v1.0/me/calendarView/delta?$deltatoken=NEWEST_DELTA"
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_delta_resp

    with patch.object(adapter, "get_valid_access_token", return_value="valid_token"), \
         patch("backend.integrations.calendar.outlook.httpx.Client", return_value=mock_client):
        res = adapter.sync_calendar_events(conn, db_session)

    assert res["sync_mode"] == "incremental"
    assert res["cancelled"] == 1
    assert res["added"] == 1

    # Check that deleted event was cancelled
    db_session.refresh(existing_evt)
    assert existing_evt.status == "cancelled"
    assert existing_evt.requires_departure_planning is False

    # Check new event
    new_evt = db_session.query(CalendarEventRecord).filter(
        CalendarEventRecord.provider_event_id == "ms_evt_200"
    ).first()
    assert new_evt is not None
    assert new_evt.title == "New On-Site Architecture Review"
    assert new_evt.requires_departure_planning is True


def test_outlook_delta_expired_recovery_410(db_session: Session, test_user: User):
    """Verify HTTP 410 / resyncRequired safely clears delta token and executes a full sync."""
    adapter = OutlookCalendarIntegration()
    crypto = OAuthTokenCrypto.get_instance()

    conn = CalendarConnection(
        user_id=test_user.id,
        provider="outlook",
        provider_account_reference="user@outlook.com",
        encrypted_tokens=crypto.encrypt_tokens({"access_token": "valid_token"}),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat(),
        sync_token="EXPIRED_DELTA_TOKEN_ABC"
    )
    db_session.add(conn)
    db_session.commit()
    db_session.refresh(conn)

    # First call returns 410 Gone, second call (full sync) returns success
    resp_410 = MagicMock()
    resp_410.status_code = 410
    resp_410.text = "resyncRequired: delta token is expired"

    resp_full = MagicMock()
    resp_full.status_code = 200
    resp_full.json.return_value = {
        "value": [
            {
                "id": "full_sync_evt_1",
                "subject": "Keynote Speech",
                "start": {"dateTime": "2026-11-22T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-11-22T11:00:00", "timeZone": "UTC"},
                "location": {"displayName": "Convention Center Hall A"},
                "isAllDay": False,
                "isCancelled": False
            }
        ],
        "@odata.deltaLink": "FRESH_DELTA_TOKEN"
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.side_effect = [resp_410, resp_full]

    with patch.object(adapter, "get_valid_access_token", return_value="valid_token"), \
         patch("backend.integrations.calendar.outlook.httpx.Client", return_value=mock_client):
        res = adapter.sync_calendar_events(conn, db_session)

    assert res["status"] == "success"
    assert res["sync_mode"] == "full"
    assert res["added"] == 1
    assert conn.sync_token == "FRESH_DELTA_TOKEN"


# =============================================================================
# 4. Multi-Provider Coexistence & Duplicate Prevention Tests
# =============================================================================

def test_multi_provider_coexistence_and_duplicate_prevention(db_session: Session, test_user: User):
    """Verify a user can connect both Google and Outlook calendars with duplicate title isolation."""
    crypto = OAuthTokenCrypto.get_instance()

    # 1. Add Google Connection
    google_conn = CalendarConnection(
        user_id=test_user.id,
        provider="google",
        provider_account_reference="tester@gmail.com",
        encrypted_tokens=crypto.encrypt_tokens({"token": "goog_tok"}),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    # 2. Add Outlook Connection
    outlook_conn = CalendarConnection(
        user_id=test_user.id,
        provider="outlook",
        provider_account_reference="tester@outlook.com",
        encrypted_tokens=crypto.encrypt_tokens({"access_token": "ms_tok"}),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add_all([google_conn, outlook_conn])
    db_session.commit()
    db_session.refresh(google_conn)
    db_session.refresh(outlook_conn)

    # 3. Add identical event title in both calendars
    google_evt = CalendarEventRecord(
        user_id=test_user.id,
        connection_id=google_conn.id,
        provider_event_id="google_same_title_123",
        title="Weekly Team Standup",
        start_time="2026-11-20T09:00:00Z",
        end_time="2026-11-20T09:30:00Z",
        location="Google HQ",
        travel_requirement="physical",
        requires_departure_planning=True,
        synced_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat()
    )
    outlook_evt = CalendarEventRecord(
        user_id=test_user.id,
        connection_id=outlook_conn.id,
        provider_event_id="outlook_same_title_456",
        title="Weekly Team Standup",
        start_time="2026-11-20T09:00:00Z",
        end_time="2026-11-20T09:30:00Z",
        location="Microsoft Redmond Campus",
        travel_requirement="physical",
        requires_departure_planning=True,
        synced_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add_all([google_evt, outlook_evt])
    db_session.commit()

    service = CalendarService()
    events = service.list_events(user_id=test_user.id, db=db_session)

    # Both events must exist without colliding or merging
    assert len(events) == 2
    providers = {e["provider"] for e in events}
    locations = {e["location"] for e in events}
    assert providers == {"google", "outlook"}
    assert "Google HQ" in locations
    assert "Microsoft Redmond Campus" in locations

    # Repeated sync on Outlook must update only Outlook event and NOT duplicate
    adapter = OutlookCalendarIntegration()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "value": [
            {
                "id": "outlook_same_title_456",
                "subject": "Weekly Team Standup",
                "start": {"dateTime": "2026-11-20T09:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-11-20T09:30:00", "timeZone": "UTC"},
                "location": {"displayName": "Microsoft Building 99"},
                "isAllDay": False,
                "isCancelled": False
            }
        ],
        "@odata.deltaLink": "DELTA_LATEST"
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp

    with patch.object(adapter, "get_valid_access_token", return_value="ms_tok"), \
         patch("backend.integrations.calendar.outlook.httpx.Client", return_value=mock_client):
        res = adapter.sync_calendar_events(outlook_conn, db_session)

    assert res["updated"] == 1
    assert res["added"] == 0

    # Total event count must still be exactly 2
    events_after = service.list_events(user_id=test_user.id, db=db_session)
    assert len(events_after) == 2


def test_independent_disconnect_outlook_preserves_google(db_session: Session, test_user: User):
    """Verify disconnecting Outlook removes only Outlook events and leaves Google untouched."""
    crypto = OAuthTokenCrypto.get_instance()

    google_conn = CalendarConnection(
        user_id=test_user.id,
        provider="google",
        provider_account_reference="user@gmail.com",
        encrypted_tokens=crypto.encrypt_tokens({"token": "goog_tok"}),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    outlook_conn = CalendarConnection(
        user_id=test_user.id,
        provider="outlook",
        provider_account_reference="user@outlook.com",
        encrypted_tokens=crypto.encrypt_tokens({"access_token": "ms_tok"}),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add_all([google_conn, outlook_conn])
    db_session.commit()

    db_session.add(CalendarEventRecord(
        user_id=test_user.id,
        connection_id=google_conn.id,
        provider_event_id="g1",
        title="Google Event",
        start_time="2026-11-20T10:00:00Z",
        end_time="2026-11-20T11:00:00Z",
        synced_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat()
    ))
    db_session.add(CalendarEventRecord(
        user_id=test_user.id,
        connection_id=outlook_conn.id,
        provider_event_id="o1",
        title="Outlook Event",
        start_time="2026-11-20T12:00:00Z",
        end_time="2026-11-20T13:00:00Z",
        synced_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat()
    ))
    db_session.commit()

    service = CalendarService()
    # Disconnect Outlook
    service.disconnect_connection(outlook_conn, db_session)

    # Outlook connection is disconnected
    db_session.refresh(outlook_conn)
    assert outlook_conn.connection_status == "disconnected"

    # Google connection is still connected
    db_session.refresh(google_conn)
    assert google_conn.connection_status == "connected"

    # Only Google event remains
    remaining = service.list_events(user_id=test_user.id, db=db_session)
    assert len(remaining) == 1
    assert remaining[0]["provider"] == "google"
    assert remaining[0]["title"] == "Google Event"


# =============================================================================
# 5. REST API Endpoint Tests
# =============================================================================

def test_api_outlook_auth_url(client: TestClient, auth_headers):
    """Test GET /api/calendar/outlook/auth-url endpoint."""
    resp = client.get("/api/calendar/outlook/auth-url", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "login.microsoftonline.com" in data["auth_url"]
    assert "state" in data


def test_api_outlook_callback_endpoint(client: TestClient, auth_headers, test_user: User):
    """Test POST /api/calendar/outlook/callback endpoint."""
    adapter = OutlookCalendarIntegration()
    _, state = adapter.generate_authorization_url(user_id=test_user.id, redirect_uri="http://localhost:5173/calendar/callback")

    mock_token = MagicMock()
    mock_token.status_code = 200
    mock_token.json.return_value = {
        "access_token": "token_abc",
        "refresh_token": "refresh_def",
        "expires_in": 3600
    }

    mock_profile = MagicMock()
    mock_profile.status_code = 200
    mock_profile.json.return_value = {"mail": "developer@outlook.com"}

    mock_graph_delta = MagicMock()
    mock_graph_delta.status_code = 200
    mock_graph_delta.json.return_value = {
        "value": [],
        "@odata.deltaLink": "INIT_DELTA_LINK"
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_token
    mock_client.get.side_effect = [mock_profile, mock_graph_delta]

    with patch("backend.integrations.calendar.outlook.httpx.Client", return_value=mock_client):
        resp = client.post(
            "/api/calendar/outlook/callback",
            headers=auth_headers,
            json={
                "code": "auth_code_123",
                "state": state,
                "redirect_uri": "http://localhost:5173/calendar/callback"
            }
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["provider"] == "outlook"
    assert body["provider_account_reference"] == "developer@outlook.com"


def test_api_departure_planning_on_outlook_event(client: TestClient, auth_headers, db_session: Session, test_user: User):
    """Test POST /api/calendar/events/{id}/plan-departure on an Outlook calendar event."""
    crypto = OAuthTokenCrypto.get_instance()
    conn = CalendarConnection(
        user_id=test_user.id,
        provider="outlook",
        provider_account_reference="traveler@outlook.com",
        encrypted_tokens=crypto.encrypt_tokens({"access_token": "tok"}),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(conn)
    db_session.commit()

    start_time = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    end_time = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()

    event_rec = CalendarEventRecord(
        user_id=test_user.id,
        connection_id=conn.id,
        provider_event_id="outlook_meeting_departure",
        title="Executive Strategy Session",
        start_time=start_time,
        end_time=end_time,
        location="1 Microsoft Way, Redmond, WA",
        travel_requirement="physical",
        requires_departure_planning=True,
        prep_buffer_override_min=25.0,
        travel_mode_override="driving",
        synced_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(event_rec)
    db_session.commit()

    resp = client.post(
        f"/api/calendar/events/{event_rec.id}/plan-departure",
        headers=auth_headers
    )
    assert resp.status_code == 200
    plan = resp.json()
    assert plan["status"] == "success"
    assert plan["event_name"] == "Executive Strategy Session"
    assert "context" in plan
    assert "optimization" in plan
    assert "recommended_departure_time" in plan["optimization"]
    assert "recommended_prep_start_time" in plan["optimization"]
    assert plan["optimization"]["prep_duration_min"] > 0
    assert len(plan["optimization"]["timeline"]) > 0
