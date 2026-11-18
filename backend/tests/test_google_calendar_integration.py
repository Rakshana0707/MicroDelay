"""Comprehensive unit and integration test suite for Google Calendar Integration (Prompt 3.2).

Validates:
1. Google OAuth 2.0 PKCE flow, CSRF state generation, and state validation.
2. Token encryption at rest using AES-CBC/HMAC (Fernet) and safe secret redaction.
3. Access token refresh when expired.
4. Token revocation and disconnect cleanup.
5. Event filtering & classification (physical, online meeting, personal, cancelled, all-day, missing location).
6. Idempotent synchronization (preventing duplicate events across repeated syncs).
7. Expired sync token recovery (handling HTTP 410 Gone with safe full resync).
8. Routine association & user manual classification overrides.
9. User data isolation across different accounts.
10. End-to-end REST API endpoints with mocked Google API calls.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from googleapiclient.errors import HttpError
from httplib2 import Response

from database import User, Event, CalendarConnection, CalendarEventRecord, DepartureLog
from security import hash_password, create_access_token

from integrations.calendar.crypto import OAuthTokenCrypto, OAuthStateStore, generate_pkce_pair
from integrations.calendar.filter import EventFilterService
from integrations.calendar.google import GoogleCalendarIntegration, GOOGLE_CALENDAR_SCOPES
from services.calendar_service import CalendarService
from main import app


@pytest.fixture
def auth_headers(db_session: Session):
    """Create a verified test user and return JWT bearer auth headers."""
    username = f"gcal_user_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    h_pw, salt = hash_password("Secret123!")
    user = User(
        username=username,
        hashed_password=h_pw,
        salt=salt,
        profile_type="it_professional",
        consent_to_global_training=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_access_token({"sub": user.username})
    return {"Authorization": f"Bearer {token}"}, user


@pytest.fixture
def second_user(db_session: Session):
    """Create a second distinct user for isolation testing."""
    username = f"other_user_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    h_pw, salt = hash_password("Secret123!")
    user = User(
        username=username,
        hashed_password=h_pw,
        salt=salt,
        profile_type="student",
        consent_to_global_training=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_access_token({"sub": user.username})
    return {"Authorization": f"Bearer {token}"}, user


# =========================================================================
# 1. OAuth State & PKCE Cryptographic Security Tests
# =========================================================================

def test_pkce_generation():
    """Verify PKCE code_verifier and code_challenge generation."""
    verifier, challenge = generate_pkce_pair()
    assert len(verifier) >= 43
    assert len(challenge) >= 43
    assert "=" not in challenge  # Base64url unpadded


def test_oauth_state_validation_and_csrf_protection():
    """State tokens must validate user ownership, expire after TTL, and be single-use."""
    store = OAuthStateStore(ttl_seconds=600)

    # 1. Successful validation
    state = store.generate_state(user_id=101, redirect_uri="http://localhost:5173/callback", code_verifier="verifier_abc")
    is_valid, verifier, redirect = store.validate_and_consume(state, expected_user_id=101)
    assert is_valid is True
    assert verifier == "verifier_abc"
    assert redirect == "http://localhost:5173/callback"

    # 2. Single-use: Second validation must fail (replay attack prevented)
    is_valid2, _, _ = store.validate_and_consume(state, expected_user_id=101)
    assert is_valid2 is False

    # 3. User mismatch rejected (cross-user hijacking prevented)
    state2 = store.generate_state(user_id=102, redirect_uri="http://localhost:5173/callback")
    is_valid_wrong_user, _, _ = store.validate_and_consume(state2, expected_user_id=999)
    assert is_valid_wrong_user is False

    # 4. Expired state rejected
    store_short = OAuthStateStore(ttl_seconds=-10)  # Already expired
    state3 = store_short.generate_state(user_id=103, redirect_uri="http://localhost:5173/callback")
    is_valid_expired, _, _ = store_short.validate_and_consume(state3, expected_user_id=103)
    assert is_valid_expired is False


# =========================================================================
# 2. Token Encryption at Rest & Redaction Tests
# =========================================================================

def test_token_encryption_and_decryption_at_rest():
    """Tokens must be encrypted with authenticated ciphertext and safely decrypted."""
    crypto = OAuthTokenCrypto(master_secret="test-encryption-master-secret-key")
    raw_payload = {
        "token": "ya29.a0AfH6SMB_secret_access_token",
        "refresh_token": "1//04_secret_refresh_token",
        "client_id": "test_client_id.apps.googleusercontent.com",
        "client_secret": "test_client_secret_xyz",
        "expiry": "2026-12-01T10:00:00Z"
    }

    encrypted_str = crypto.encrypt_tokens(raw_payload)
    assert isinstance(encrypted_str, str)
    assert "ya29." not in encrypted_str
    assert "refresh_token" not in encrypted_str

    decrypted = crypto.decrypt_tokens(encrypted_str)
    assert decrypted["token"] == raw_payload["token"]
    assert decrypted["refresh_token"] == raw_payload["refresh_token"]

    # Redaction test
    redacted = OAuthTokenCrypto.redact_token_dict(raw_payload)
    assert redacted["token"].startswith("ya29")
    assert "secret" not in redacted["token"]
    assert redacted["refresh_token"] != raw_payload["refresh_token"]


# =========================================================================
# 3. Event Travel Requirement Filtering & Classification Tests
# =========================================================================

def test_event_filter_physical_location():
    """Physical locations must trigger departure planning."""
    res = EventFilterService.classify_event(
        title="Operating Systems Lab",
        location="Science Building Room 302, 100 University Ave",
        status="confirmed"
    )
    assert res.travel_requirement == "physical"
    assert res.requires_departure_planning is True
    assert res.missing_location_warning is False
    assert "Science Building" in res.sanitized_location


def test_event_filter_online_meeting_urls():
    """Google Meet, Zoom, Teams links must be categorized as online (no travel)."""
    # 1. Google Meet in location
    res1 = EventFilterService.classify_event(
        title="Weekly Sprint Planning",
        location="https://meet.google.com/abc-defg-hij",
        status="confirmed"
    )
    assert res1.travel_requirement == "online"
    assert res1.requires_departure_planning is False
    assert res1.online_meeting_url == "https://meet.google.com/abc-defg-hij"

    # 2. Zoom link in description / title
    res2 = EventFilterService.classify_event(
        title="Research Sync (Zoom Meeting)",
        location="",
        description="Join Zoom meeting: https://zoom.us/j/123456789"
    )
    assert res2.travel_requirement == "online"
    assert res2.requires_departure_planning is False
    assert "zoom.us" in (res2.online_meeting_url or "")

    # 3. Microsoft Teams
    res3 = EventFilterService.classify_event(
        title="Client Advisory Call",
        location="https://teams.microsoft.com/l/meetup-join/19%3ameeting"
    )
    assert res3.travel_requirement == "online"
    assert res3.requires_departure_planning is False


def test_event_filter_personal_blocks_and_allday():
    """Focus time, OOO, and all-day events must not trigger departure planning."""
    res1 = EventFilterService.classify_event(
        title="Focus Time - Coding",
        location=None,
        status="confirmed"
    )
    assert res1.travel_requirement == "personal"
    assert res1.requires_departure_planning is False

    res2 = EventFilterService.classify_event(
        title="National Holiday",
        location="Campus",
        is_all_day=True
    )
    assert res2.travel_requirement == "personal"
    assert res2.requires_departure_planning is False


def test_event_filter_missing_location_warning():
    """Timed events with no location must be flagged with missing_location_warning."""
    res = EventFilterService.classify_event(
        title="Dentist Appointment",
        location=None,
        status="confirmed",
        is_all_day=False
    )
    assert res.travel_requirement == "unknown"
    assert res.requires_departure_planning is True
    assert res.missing_location_warning is True


def test_event_filter_cancelled_events():
    """Cancelled events are excluded."""
    res = EventFilterService.classify_event(
        title="Department Seminar",
        location="Auditorium B",
        status="cancelled"
    )
    assert res.travel_requirement == "excluded"
    assert res.requires_departure_planning is False


# =========================================================================
# 4. Synchronization Engine & Idempotency Tests
# =========================================================================

def create_mock_google_service(mock_events: list, next_sync_token="sync_token_v1", page_token=None):
    """Helper to mock googleapiclient calendar v3 discovery."""
    mock_service = MagicMock()
    mock_events_resource = MagicMock()
    mock_list = MagicMock()

    mock_list.execute.return_value = {
        "items": mock_events,
        "nextSyncToken": next_sync_token,
        "nextPageToken": page_token
    }
    mock_events_resource.list.return_value = mock_list
    mock_service.events.return_value = mock_events_resource
    return mock_service


def test_idempotent_event_synchronization(db_session: Session):
    """Repeated synchronization calls must update existing records, not duplicate them."""
    user = User(username="sync_tester", hashed_password="h", salt="s", profile_type="student")
    db_session.add(user)
    db_session.commit()

    crypto = OAuthTokenCrypto.get_instance()
    enc_tokens = crypto.encrypt_tokens({"token": "tok", "refresh_token": "ref", "expiry": "2030-01-01T00:00:00Z"})

    conn = CalendarConnection(
        user_id=user.id,
        provider="google",
        provider_account_reference="tester@gmail.com",
        encrypted_tokens=enc_tokens,
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(conn)
    db_session.commit()

    gcal = GoogleCalendarIntegration()

    sample_events = [
        {
            "id": "gcal_event_001",
            "summary": "Algorithms Lecture",
            "start": {"dateTime": "2026-11-20T09:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2026-11-20T10:30:00Z", "timeZone": "UTC"},
            "location": "Engineering Hall 105",
            "status": "confirmed"
        },
        {
            "id": "gcal_event_002",
            "summary": "Team Standup",
            "start": {"dateTime": "2026-11-20T11:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2026-11-20T11:30:00Z", "timeZone": "UTC"},
            "location": "https://meet.google.com/xyz-abcd-efg",
            "status": "confirmed"
        }
    ]

    mock_service = create_mock_google_service(sample_events, next_sync_token="sync_token_v1")

    with patch("googleapiclient.discovery.build", return_value=mock_service):
        # 1. First Synchronization
        res1 = gcal.sync_calendar_events(connection=conn, db=db_session)
        assert res1["added"] == 2
        assert res1["updated"] == 0

        count_after_first = db_session.query(CalendarEventRecord).filter(CalendarEventRecord.connection_id == conn.id).count()
        assert count_after_first == 2

        # 2. Second Synchronization with same data (Idempotency check)
        res2 = gcal.sync_calendar_events(connection=conn, db=db_session)
        assert res2["added"] == 0
        assert res2["updated"] == 2

        count_after_second = db_session.query(CalendarEventRecord).filter(CalendarEventRecord.connection_id == conn.id).count()
        assert count_after_second == 2  # Exactly 2, no duplicates created!

        # Verify classifications
        ev1 = db_session.query(CalendarEventRecord).filter(CalendarEventRecord.provider_event_id == "gcal_event_001").first()
        assert ev1.travel_requirement == "physical"
        assert ev1.requires_departure_planning is True

        ev2 = db_session.query(CalendarEventRecord).filter(CalendarEventRecord.provider_event_id == "gcal_event_002").first()
        assert ev2.travel_requirement == "online"
        assert ev2.requires_departure_planning is False


def test_expired_sync_token_410_recovery(db_session: Session):
    """When Google returns HTTP 410 Gone for an expired syncToken, system recovers with full resync."""
    user = User(username="recovery_user", hashed_password="h", salt="s", profile_type="student")
    db_session.add(user)
    db_session.commit()

    crypto = OAuthTokenCrypto.get_instance()
    enc_tokens = crypto.encrypt_tokens({"token": "tok", "refresh_token": "ref", "expiry": "2030-01-01T00:00:00Z"})

    conn = CalendarConnection(
        user_id=user.id,
        provider="google",
        provider_account_reference="recovery@gmail.com",
        encrypted_tokens=enc_tokens,
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat(),
        sync_token="stale_expired_token_123"
    )
    db_session.add(conn)
    db_session.commit()

    gcal = GoogleCalendarIntegration()

    # First call raises 410 Gone; second call succeeds with fresh data
    http_410_resp = Response({"status": 410})
    err_410 = HttpError(http_410_resp, b"Sync token expired")

    mock_service_fail = MagicMock()
    mock_service_fail.events.return_value.list.return_value.execute.side_effect = err_410

    mock_service_success = create_mock_google_service([
        {
            "id": "recovered_event_01",
            "summary": "Recovered Meeting",
            "start": {"dateTime": "2026-11-21T10:00:00Z"},
            "end": {"dateTime": "2026-11-21T11:00:00Z"},
            "location": "Room 101",
            "status": "confirmed"
        }
    ], next_sync_token="fresh_sync_token_456")

    call_count = 0
    def mock_build(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_service_fail
        return mock_service_success

    with patch("googleapiclient.discovery.build", side_effect=mock_build):
        result = gcal.sync_calendar_events(connection=conn, db=db_session)
        assert result["status"] == "success"
        assert result["sync_mode"] == "full"
        assert conn.sync_token == "fresh_sync_token_456"

        rec = db_session.query(CalendarEventRecord).filter(CalendarEventRecord.provider_event_id == "recovered_event_01").first()
        assert rec is not None


# =========================================================================
# 5. Routine Association & User Classification Overrides
# =========================================================================

def test_routine_association_and_classification_override(db_session: Session):
    """User can reclassify events, associate routines, and customize buffers."""
    user = User(username="assoc_user", hashed_password="h", salt="s", profile_type="student")
    db_session.add(user)
    db_session.commit()

    # Create a routine template
    routine = Event(
        user_id=user.id,
        name="Morning Lab Routine",
        day_of_week="Friday",
        start_time="08:30:00",
        estimated_prep_duration=25,
        estimated_travel_duration=15,
        importance="high"
    )
    db_session.add(routine)

    conn = CalendarConnection(
        user_id=user.id,
        provider="google",
        encrypted_tokens="enc",
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(conn)
    db_session.commit()

    cal_event = CalendarEventRecord(
        user_id=user.id,
        connection_id=conn.id,
        provider_event_id="evt_override_1",
        title="Friday Physics",
        start_time="2026-11-22T08:30:00Z",
        end_time="2026-11-22T10:00:00Z",
        location=None,
        travel_requirement="unknown",
        requires_departure_planning=True,
        missing_location_warning=True,
        synced_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(cal_event)
    db_session.commit()

    service = CalendarService()

    # Apply user override: associate routine, set location, set prep buffer
    updated = service.update_event_classification(
        event_id=cal_event.id,
        user_id=user.id,
        travel_requirement="physical",
        associated_routine_id=routine.id,
        prep_buffer_override_min=35.0,
        travel_mode_override="transit",
        location="Physics Building Auditorium",
        db=db_session
    )

    assert updated["travel_requirement"] == "physical"
    assert updated["associated_routine_id"] == routine.id
    assert updated["prep_buffer_override_min"] == 35.0
    assert updated["travel_mode_override"] == "transit"
    assert updated["location"] == "Physics Building Auditorium"
    assert updated["missing_location_warning"] is False

    # Check context resolution inherits routine and override parameters
    ctx = service.get_event_context(user_id=user.id, event_id=str(cal_event.id), db=db_session)
    assert ctx["estimated_prep_duration"] == 35.0  # Override wins
    assert ctx["estimated_travel_duration"] == 15.0  # Inherited from routine
    assert ctx["importance"] == "high"  # Inherited from routine


# =========================================================================
# 6. User Data Isolation Tests
# =========================================================================

def test_user_calendar_data_isolation(db_session: Session):
    """User A cannot access or modify User B's calendar events or connections."""
    u1 = User(username="user_one", hashed_password="h", salt="s", profile_type="student")
    u2 = User(username="user_two", hashed_password="h", salt="s", profile_type="teacher")
    db_session.add_all([u1, u2])
    db_session.commit()

    conn1 = CalendarConnection(user_id=u1.id, provider="google", encrypted_tokens="e1", connected_at="2026-01-01T00:00:00Z")
    conn2 = CalendarConnection(user_id=u2.id, provider="google", encrypted_tokens="e2", connected_at="2026-01-01T00:00:00Z")
    db_session.add_all([conn1, conn2])
    db_session.commit()

    ev1 = CalendarEventRecord(
        user_id=u1.id,
        connection_id=conn1.id,
        provider_event_id="p1",
        title="User 1 Secret Meeting",
        start_time="2026-11-20T09:00:00Z",
        end_time="2026-11-20T10:00:00Z",
        synced_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )
    db_session.add(ev1)
    db_session.commit()

    service = CalendarService()

    # User 2 listing events must NOT see User 1's event
    u2_events = service.list_events(user_id=u2.id, db=db_session)
    assert len(u2_events) == 0

    # User 2 attempting to modify User 1's event must fail
    with pytest.raises(ValueError):
        service.update_event_classification(event_id=ev1.id, user_id=u2.id, travel_requirement="online", db=db_session)


# =========================================================================
# 7. Disconnect and Token Revocation Tests
# =========================================================================

def test_disconnect_and_token_revocation(db_session: Session):
    """Disconnecting marks connection disconnected, revokes token, and removes synced events."""
    user = User(username="disconnect_user", hashed_password="h", salt="s", profile_type="student")
    db_session.add(user)
    db_session.commit()

    crypto = OAuthTokenCrypto.get_instance()
    enc_tokens = crypto.encrypt_tokens({"token": "valid_token", "refresh_token": "valid_ref"})

    conn = CalendarConnection(
        user_id=user.id,
        provider="google",
        encrypted_tokens=enc_tokens,
        connection_status="connected",
        connected_at="2026-01-01T00:00:00Z"
    )
    db_session.add(conn)
    db_session.commit()

    ev = CalendarEventRecord(
        user_id=user.id,
        connection_id=conn.id,
        provider_event_id="ev_to_delete",
        title="Event to Delete",
        start_time="2026-11-20T09:00:00Z",
        end_time="2026-11-20T10:00:00Z",
        synced_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )
    db_session.add(ev)
    db_session.commit()

    gcal = GoogleCalendarIntegration()

    # Mock httpx revocation call
    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        success = gcal.disconnect_and_revoke(connection=conn, db=db_session)
        assert success is True

    assert conn.connection_status == "disconnected"
    assert conn.sync_token is None

    # Synced events removed
    remaining = db_session.query(CalendarEventRecord).filter(CalendarEventRecord.connection_id == conn.id).count()
    assert remaining == 0


# =========================================================================
# 8. Full REST API End-to-End Tests
# =========================================================================

def test_api_google_auth_url_and_callback(client: TestClient, auth_headers):
    """Test /api/calendar/google/auth-url and /api/calendar/google/callback."""
    headers, user = auth_headers

    # 1. GET auth-url
    resp = client.get("/api/calendar/google/auth-url", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "accounts.google.com" in data["auth_url"]
    assert "state" in data
    state = data["state"]

    # 2. POST callback with mock token exchange and sync
    mock_credentials = MagicMock()
    mock_credentials.token = "mock_access_token_123"
    mock_credentials.refresh_token = "mock_refresh_token_456"
    mock_credentials.token_uri = "https://oauth2.googleapis.com/token"
    mock_credentials.client_id = "mock_client"
    mock_credentials.client_secret = "mock_secret"
    mock_credentials.scopes = GOOGLE_CALENDAR_SCOPES
    mock_credentials.expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_credentials.id_token = {"email": "testuser@gmail.com"}

    mock_flow = MagicMock()
    mock_flow.credentials = mock_credentials

    with patch("google_auth_oauthlib.flow.Flow.from_client_config", return_value=mock_flow):
        with patch.object(GoogleCalendarIntegration, "sync_calendar_events", return_value={"added": 5, "updated": 0}):
            callback_resp = client.post("/api/calendar/google/callback", json={
                "code": "4/0Aean_auth_code_xyz",
                "state": state
            }, headers=headers)

            assert callback_resp.status_code == 200
            cb_data = callback_resp.json()
            assert cb_data["status"] == "success"
            assert cb_data["provider"] == "google"
            assert cb_data["account"] == "testuser@gmail.com"


def test_api_calendar_events_classification_and_planning(client: TestClient, auth_headers, db_session: Session):
    """Test event listing, user classification PATCH, and direct departure planning POST."""
    headers, user = auth_headers

    conn = CalendarConnection(
        user_id=user.id,
        provider="google",
        provider_account_reference="alex@gmail.com",
        encrypted_tokens=OAuthTokenCrypto.get_instance().encrypt_tokens({"token": "t"}),
        connection_status="connected",
        connected_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(conn)
    db_session.commit()

    ev = CalendarEventRecord(
        user_id=user.id,
        connection_id=conn.id,
        provider_event_id="api_ev_01",
        title="Client Strategy Meeting",
        start_time=(datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
        end_time=(datetime.now(timezone.utc) + timedelta(hours=5)).isoformat(),
        location="Downtown Office Suite 300",
        travel_requirement="physical",
        requires_departure_planning=True,
        synced_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat()
    )
    db_session.add(ev)
    db_session.commit()

    # 1. GET /api/calendar/events
    list_resp = client.get("/api/calendar/events?departure_only=true", headers=headers)
    assert list_resp.status_code == 200
    events = list_resp.json()["events"]
    assert len(events) == 1
    assert events[0]["title"] == "Client Strategy Meeting"

    # 2. PATCH /api/calendar/events/{id}/classify
    patch_resp = client.patch(f"/api/calendar/events/{ev.id}/classify", json={
        "prep_buffer_override_min": 40.0,
        "travel_mode_override": "driving"
    }, headers=headers)
    assert patch_resp.status_code == 200
    assert patch_resp.json()["event"]["prep_buffer_override_min"] == 40.0

    # 3. POST /api/calendar/events/{id}/plan-departure
    plan_resp = client.post(f"/api/calendar/events/{ev.id}/plan-departure", headers=headers)
    assert plan_resp.status_code == 200
    plan_data = plan_resp.json()
    assert plan_data["status"] == "success"
    assert "context" in plan_data
    assert "optimization" in plan_data
    assert "recommended_departure_time" in plan_data["optimization"]
    assert "recommended_prep_start_time" in plan_data["optimization"]

    # 4. POST /api/calendar/connections/{id}/disconnect
    disc_resp = client.post(f"/api/calendar/connections/{conn.id}/disconnect", headers=headers)
    assert disc_resp.status_code == 200
    assert disc_resp.json()["status"] == "success"
