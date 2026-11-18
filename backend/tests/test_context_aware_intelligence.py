"""Comprehensive unit and integration test suite for MicroDelay Phase 3 Architecture.

Validates:
1. Missing integrations & full graceful degradation
2. Partial context aggregation
3. Invalid external data & schema boundary validation
4. Expired context detection & TTL lifecycle
5. API timeout handling & resilient recovery
6. Authentication & credential security
7. End-to-end Context Aggregation with Phase 2 Behavioral Intelligence
8. Dynamic Departure Optimization & Timeline calculation
9. Smart Proactive Notification scheduling & milestone evaluation
10. Journey Tracking state machine & database feedback loop
11. Backward compatibility with Phase 1 & Phase 2 APIs
"""

import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from database import User, Event, DepartureLog, get_db
from security import hash_password, create_access_token

from integrations.base import (
    ObservationType, ExternalObservationMetadata, IntegrationStatus,
    IntegrationTimeoutError, IntegrationAuthError, IntegrationDataError
)
from integrations.credentials import CredentialManager, OAuthToken
from integrations.registry import IntegrationRegistry
from integrations.calendar.base import CalendarEvent
from integrations.calendar.google import GoogleCalendarIntegration
from integrations.calendar.outlook import OutlookCalendarIntegration
from integrations.calendar.mock import MockCalendarIntegration
from integrations.travel.base import TravelMode, TravelEstimate
from integrations.travel.mock import MockTravelIntegration
from integrations.traffic.base import TrafficCongestionLevel, TrafficObservation
from integrations.traffic.mock import MockTrafficIntegration
from integrations.weather.base import WeatherSeverity, WeatherObservation
from integrations.weather.mock import MockWeatherIntegration

from services.context_service import ContextService, DepartureContext
from services.calendar_service import CalendarService
from services.travel_service import TravelService
from services.weather_service import WeatherService
from services.departure_optimizer import DepartureOptimizer
from services.notification_service import NotificationService, NotificationMilestone
from services.journey_service import JourneyService, JourneyPhase, InvalidJourneyTransitionError

from main import app


@pytest.fixture
def auth_headers(db_session: Session):
    """Create a verified test user and return JWT bearer auth headers."""
    username = f"phase3_user_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
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
# 1. External Data Contract and Observation Metadata Tests
# =========================================================================

def test_observation_metadata_contract():
    """Verify that every external observation adheres to the data contract."""
    now = datetime.now(timezone.utc)
    valid_until = (now + timedelta(minutes=10)).isoformat()

    meta = ExternalObservationMetadata(
        source="google_calendar:primary",
        applicable_id="event_123",
        valid_from=now.isoformat(),
        valid_until=valid_until,
        observation_type=ObservationType.LIVE,
        confidence=0.98,
        is_fallback=False,
        details={"provider_version": "v3"}
    )

    data = meta.to_dict()
    assert data["source"] == "google_calendar:primary"
    assert data["applicable_id"] == "event_123"
    assert data["observation_type"] == "live"
    assert data["confidence"] == 0.98
    assert data["is_fallback"] is False
    assert meta.is_valid(now) is True

    # Test expired validity check
    future_expired = now + timedelta(minutes=15)
    assert meta.is_valid(future_expired) is False


def test_observation_metadata_reconstruction():
    """Test serializing and deserializing metadata."""
    raw = {
        "source": "mock_traffic",
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "applicable_id": "route_99",
        "valid_from": datetime.now(timezone.utc).isoformat(),
        "valid_until": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        "observation_type": "historical",
        "confidence": 0.82,
        "is_fallback": True,
        "latency_ms": 14.5,
        "details": {"jam": 3.2}
    }
    meta = ExternalObservationMetadata.from_dict(raw)
    assert meta.observation_type == ObservationType.HISTORICAL
    assert meta.is_fallback is True
    assert meta.confidence == 0.82


# =========================================================================
# 2. Secure Credential Management Tests
# =========================================================================

def test_credential_manager_security_and_masking(monkeypatch):
    """Test that credentials are never exposed, environment variables fall back, and masking works."""
    cm = CredentialManager()

    # Secret masking
    assert cm.mask_secret("short") == "***"
    assert cm.mask_secret("sk-12345678") == "sk...78"
    assert cm.mask_secret("AIzaSyDummySecretKey12345") == "AIza...2345"

    # Store and retrieve
    cm.set_credential("google_calendar", "CLIENT_SECRET", "super_secret_oauth_secret_123")
    assert cm.get_credential("google_calendar", "CLIENT_SECRET") == "super_secret_oauth_secret_123"

    inventory = cm.get_masked_inventory()
    assert "google_calendar" in inventory
    assert inventory["google_calendar"]["CLIENT_SECRET"] == "supe..._123"

    # Environment variable fallback
    monkeypatch.setenv("MICRODELAY_WEATHER_API_KEY", "env_secret_key_8899")
    assert cm.get_credential("weather", "API_KEY") == "env_secret_key_8899"

    # Delete
    assert cm.delete_credential("google_calendar", "CLIENT_SECRET") is True
    assert cm.has_credential("google_calendar", "CLIENT_SECRET") is False


def test_oauth_token_expiry_handling():
    """Test OAuthToken expiry detection and safety buffer."""
    now = datetime.now(timezone.utc)
    expired_token = OAuthToken(
        access_token="expired_token_abc",
        expires_at=now - timedelta(seconds=10)
    )
    assert expired_token.is_expired() is True

    valid_token = OAuthToken(
        access_token="valid_token_xyz",
        expires_at=now + timedelta(minutes=30)
    )
    assert valid_token.is_expired() is False

    soon_token = OAuthToken(
        access_token="soon_token_123",
        expires_at=now + timedelta(seconds=30)
    )
    # With 60-second default buffer, 30-second token is considered expired
    assert soon_token.is_expired(buffer_seconds=60) is True


# =========================================================================
# 3. Independent Service Boundaries & Registry Tests
# =========================================================================

def test_integration_registry_isolation():
    """Verify each integration can be registered, toggled, and inspected independently."""
    reg = IntegrationRegistry()
    cal = MockCalendarIntegration(enabled=True)
    trav = MockTravelIntegration(enabled=True)
    traf = MockTrafficIntegration(enabled=True)
    weat = MockWeatherIntegration(enabled=True)

    reg.register("calendar", cal, is_default=True)
    reg.register("travel", trav, is_default=True)
    reg.register("traffic", traf, is_default=True)
    reg.register("weather", weat, is_default=True)

    assert reg.is_category_available("calendar") is True
    assert reg.is_category_available("weather") is True

    # Disable only weather
    reg.disable_integration("weather")
    assert reg.is_category_available("weather") is False
    assert reg.is_category_available("calendar") is True

    health = reg.get_all_health()
    assert health["weather"]["mock_weather"]["status"] == "disabled"
    assert health["calendar"]["mock_calendar"]["status"] == "healthy"


# =========================================================================
# 4. Graceful Degradation: Missing & Partial Integrations
# =========================================================================

def test_graceful_degradation_when_all_external_integrations_disabled(db_session: Session):
    """When all external APIs are disabled, system falls back cleanly to routine templates & heuristics."""
    reg = IntegrationRegistry()
    reg.register("calendar", MockCalendarIntegration(enabled=False))
    reg.register("travel", MockTravelIntegration(enabled=False))
    reg.register("traffic", MockTrafficIntegration(enabled=False))
    reg.register("weather", MockWeatherIntegration(enabled=False))

    user = User(
        username="offline_user",
        hashed_password="h",
        salt="s",
        profile_type="student",
        consent_to_global_training=True
    )
    db_session.add(user)
    db_session.commit()

    # Add a database routine template
    event = Event(
        user_id=user.id,
        name="Offline Physics Lab",
        day_of_week="Monday",
        start_time="09:00:00",
        estimated_prep_duration=25,
        estimated_travel_duration=18,
        importance="high"
    )
    db_session.add(event)
    db_session.commit()

    ctx_service = ContextService(registry=reg)
    dep_ctx = ctx_service.aggregate_context(
        user_id=user.id,
        event_name="Offline Physics Lab",
        destination="Science Hall",
        db=db_session
    )

    # Context should succeed without errors
    assert dep_ctx.user_id == user.id
    assert dep_ctx.event_name == "Offline Physics Lab"
    assert dep_ctx.degradation_status["calendar"] == "database_routine_template"
    assert dep_ctx.degradation_status["travel"] == "travel_api_unavailable"
    assert dep_ctx.degradation_status["weather"] == "weather_disabled"
    assert dep_ctx.weather_conditions is None
    assert dep_ctx.estimated_prep_duration == 25.0
    assert dep_ctx.estimated_travel_duration == 18.0


def test_partial_context_aggregation(db_session: Session):
    """Test partial availability (e.g. travel available, but weather and traffic fail)."""
    reg = IntegrationRegistry()
    reg.register("calendar", MockCalendarIntegration(enabled=True))
    travel = MockTravelIntegration(enabled=True)
    travel.set_route_duration("Home", "Campus", 22.0)
    reg.register("travel", travel)
    # Traffic times out, weather disabled
    reg.register("traffic", MockTrafficIntegration(simulate_timeout=True))
    reg.register("weather", MockWeatherIntegration(enabled=False))

    user = User(
        username="partial_user",
        hashed_password="h",
        salt="s",
        profile_type="it_professional",
        consent_to_global_training=True
    )
    db_session.add(user)
    db_session.commit()

    ctx_service = ContextService(registry=reg)
    dep_ctx = ctx_service.aggregate_context(
        user_id=user.id,
        current_location="Home",
        destination="Campus",
        db=db_session
    )

    assert dep_ctx.destination == "Campus"
    assert dep_ctx.live_travel_duration == 22.0
    assert dep_ctx.degradation_status["weather"] == "weather_disabled"
    # Traffic degradation handled gracefully
    assert dep_ctx.traffic_conditions["congestion_level"] == "unknown"


# =========================================================================
# 5. Invalid External Data & API Timeouts
# =========================================================================

def test_invalid_external_travel_data_handling(db_session: Session):
    """When external travel provider returns invalid negative duration, system falls back safely."""
    reg = IntegrationRegistry()
    reg.register("travel", MockTravelIntegration(simulate_invalid_data=True))

    user = User(username="invalid_data_user", hashed_password="h", salt="s", profile_type="teacher")
    db_session.add(user)
    db_session.commit()

    ctx_service = ContextService(registry=reg)
    dep_ctx = ctx_service.aggregate_context(
        user_id=user.id,
        current_location="A",
        destination="B",
        db=db_session
    )

    # Validated fallback should take over
    assert dep_ctx.estimated_travel_duration > 0
    assert dep_ctx.degradation_status["travel"] == "travel_api_unavailable"
    val = dep_ctx.validate()
    assert val.is_valid is True


def test_api_timeout_handling(db_session: Session):
    """External API timeouts must not crash MicroDelay."""
    reg = IntegrationRegistry()
    reg.register("calendar", MockCalendarIntegration(simulate_timeout=True))
    reg.register("travel", MockTravelIntegration(simulate_timeout=True))
    reg.register("traffic", MockTrafficIntegration(simulate_timeout=True))
    reg.register("weather", MockWeatherIntegration(simulate_timeout=True))

    user = User(username="timeout_user", hashed_password="h", salt="s", profile_type="student")
    db_session.add(user)
    db_session.commit()

    ctx_service = ContextService(registry=reg)
    dep_ctx = ctx_service.aggregate_context(
        user_id=user.id,
        destination="Hospital",
        db=db_session
    )

    assert dep_ctx.user_id == user.id
    assert dep_ctx.degradation_status["calendar"] == "heuristic_default"
    assert dep_ctx.degradation_status["travel"] == "travel_api_unavailable"
    assert dep_ctx.degradation_status["weather"] == "weather_failed_fallback"


def test_authentication_failure_handling(db_session: Session):
    """Google and Outlook OAuth token expiration / auth failures degrade gracefully."""
    reg = IntegrationRegistry()
    reg.register("calendar", MockCalendarIntegration(simulate_auth_error=True))

    user = User(username="auth_fail_user", hashed_password="h", salt="s", profile_type="professor")
    db_session.add(user)
    db_session.commit()

    ctx_service = ContextService(registry=reg)
    dep_ctx = ctx_service.aggregate_context(
        user_id=user.id,
        event_name="Faculty Review",
        db=db_session
    )

    assert dep_ctx.user_id == user.id
    assert dep_ctx.degradation_status["calendar"] == "heuristic_default"


# =========================================================================
# 6. Missing Destination Handling
# =========================================================================

def test_missing_destination_handling(db_session: Session):
    """Missing destination is detected, flagged for user input, and does not crash."""
    reg = IntegrationRegistry()
    reg.register("travel", MockTravelIntegration(enabled=True))

    user = User(username="no_dest_user", hashed_password="h", salt="s", profile_type="student")
    db_session.add(user)
    db_session.commit()

    ctx_service = ContextService(registry=reg)
    dep_ctx = ctx_service.aggregate_context(
        user_id=user.id,
        destination=None,  # No destination
        db=db_session
    )

    assert dep_ctx.is_destination_required is True
    assert "destination" in dep_ctx.missing_fields
    val = dep_ctx.validate()
    assert any("Destination is required" in w for w in val.warnings)


# =========================================================================
# 7. Context Expiration & TTL Lifecycle
# =========================================================================

def test_context_expiration_lifecycle():
    """Verify DepartureContext TTL expiration check."""
    now = datetime.now(timezone.utc)
    ctx = DepartureContext(
        user_id=1,
        event_name="Team Standup",
        context_timestamp=(now - timedelta(seconds=400)).isoformat(),
        ttl_seconds=300
    )
    assert ctx.is_expired(now) is True

    fresh_ctx = DepartureContext(
        user_id=1,
        event_name="Team Standup",
        context_timestamp=now.isoformat(),
        ttl_seconds=300
    )
    assert fresh_ctx.is_expired(now) is False


# =========================================================================
# 8. Dynamic Departure Optimizer Tests
# =========================================================================

def test_departure_optimizer_timeline():
    """Test timeline computation: prep start, departure, buffer, arrival."""
    now = datetime.now(timezone.utc)
    start_time = now + timedelta(hours=3)

    ctx = DepartureContext(
        user_id=1,
        event_name="Algorithm Exam",
        event_start_time=start_time.isoformat(),
        destination="Main Hall",
        event_importance="critical",
        estimated_prep_duration=30.0,
        personalized_prep_duration=36.0,
        estimated_travel_duration=20.0,
        live_travel_duration=25.0,  # 25 min travel
        safety_buffer=15.0,         # 15 min buffer
        weather_conditions={"recommended_buffer_min": 5.0}
    )

    optimizer = DepartureOptimizer()
    result = optimizer.optimize_departure(ctx)

    assert result.event_start_time == start_time.isoformat()
    assert result.prep_duration_min == 36.0
    assert result.travel_duration_min == 25.0
    assert result.safety_buffer_min == 15.0

    # Total lead time should equal prep (36) + buffer (15) + travel (25) + critical early padding (2) = 78 min
    assert result.total_lead_time_min == 78.0
    assert len(result.timeline) == 4
    phases = [seg.phase for seg in result.timeline]
    assert phases == ["preparation", "safety_buffer", "transit", "arrival"]


# =========================================================================
# 9. Smart Proactive Notification Service Tests
# =========================================================================

def test_notification_scheduling_and_evaluation():
    """Test scheduling and retrieving due departure notifications."""
    now = datetime.now(timezone.utc)
    future_event = now + timedelta(hours=2)

    ctx = DepartureContext(
        user_id=10,
        event_name="Thesis Defense",
        event_start_time=future_event.isoformat(),
        destination="Auditorium",
        estimated_prep_duration=30.0,
        personalized_prep_duration=30.0,
        estimated_travel_duration=20.0,
        live_travel_duration=20.0,
        safety_buffer=10.0
    )

    optimizer = DepartureOptimizer()
    opt = optimizer.optimize_departure(ctx)

    notif_service = NotificationService()
    scheduled = notif_service.schedule_departure_notifications(
        user_id=10,
        optimization=opt,
        event_name="Thesis Defense"
    )

    assert len(scheduled) > 0
    milestones = [n.milestone for n in scheduled]
    assert NotificationMilestone.PREP_START in milestones or NotificationMilestone.DEPART_NOW in milestones

    # Test dispatch
    notif_id = scheduled[0].notification_id
    assert notif_service.dispatch(notif_id) is True
    assert scheduled[0].is_dispatched is True


# =========================================================================
# 10. Journey Tracking State Machine Tests
# =========================================================================

def test_journey_tracking_lifecycle(db_session: Session):
    """Test journey state machine transitions and persistence into `departure_logs`."""
    user = User(username="journey_runner", hashed_password="h", salt="s", profile_type="student")
    db_session.add(user)
    db_session.commit()

    ctx = DepartureContext(
        user_id=user.id,
        event_name="Morning Lecture",
        estimated_prep_duration=30.0,
        estimated_travel_duration=20.0
    )

    service = JourneyService()
    journey = service.create_journey(
        user_id=user.id,
        context=ctx,
        recommended_prep_start="08:00:00",
        recommended_departure="08:30:00",
        target_arrival="08:50:00"
    )

    assert journey.phase == JourneyPhase.SCHEDULED

    # Valid transition to PREP_STARTED
    t1 = datetime.now(timezone.utc)
    service.transition_phase(journey.journey_id, JourneyPhase.PREP_STARTED, timestamp=t1)
    assert journey.phase == JourneyPhase.PREP_STARTED
    assert journey.actual_prep_start is not None

    # Valid transition to DEPARTED
    t2 = t1 + timedelta(minutes=32)
    service.transition_phase(journey.journey_id, JourneyPhase.DEPARTED, timestamp=t2)
    assert journey.phase == JourneyPhase.DEPARTED
    assert journey.actual_prep_duration_min == 32.0

    # Valid transition to ARRIVED
    t3 = t2 + timedelta(minutes=22)
    service.transition_phase(journey.journey_id, JourneyPhase.ARRIVED, timestamp=t3)
    assert journey.phase == JourneyPhase.ARRIVED
    assert journey.actual_travel_duration_min == 22.0

    # Complete and verify persistence to departure_logs
    service.transition_phase(journey.journey_id, JourneyPhase.COMPLETED, db=db_session)
    assert journey.phase == JourneyPhase.COMPLETED

    log_entry = db_session.query(DepartureLog).filter(DepartureLog.user_id == user.id).first()
    assert log_entry is not None
    assert log_entry.event_name == "Morning Lecture"
    assert log_entry.actual_prep_duration_min == 32.0
    assert log_entry.actual_travel_duration_min == 22.0


def test_invalid_journey_transition_rejection():
    """Verify illegal transitions raise InvalidJourneyTransitionError."""
    service = JourneyService()
    ctx = DepartureContext(user_id=1, event_name="Test")
    journey = service.create_journey(1, ctx, "08:00", "08:30", "08:50")

    # Cannot transition directly from SCHEDULED to ARRIVED
    with pytest.raises(InvalidJourneyTransitionError):
        service.transition_phase(journey.journey_id, JourneyPhase.ARRIVED)


# =========================================================================
# 11. REST API Integration Endpoints Tests
# =========================================================================

def test_api_integrations_status_and_toggle(client: TestClient, auth_headers):
    """Test /api/integrations/status and /api/integrations/toggle endpoints."""
    headers, user = auth_headers

    # GET status
    resp = client.get("/api/integrations/status", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "calendar" in data["integrations"]
    assert "travel" in data["integrations"]

    # Toggle weather off
    toggle_resp = client.post("/api/integrations/toggle", json={
        "category": "weather",
        "provider_name": "mock_weather",
        "enabled": False
    }, headers=headers)
    assert toggle_resp.status_code == 200
    assert toggle_resp.json()["enabled"] is False

    # Toggle back on
    toggle_resp2 = client.post("/api/integrations/toggle", json={
        "category": "weather",
        "provider_name": "mock_weather",
        "enabled": True
    }, headers=headers)
    assert toggle_resp2.status_code == 200
    assert toggle_resp2.json()["enabled"] is True


def test_api_departure_context_and_optimize(client: TestClient, auth_headers):
    """Test /api/context/departure and /api/departure/optimize REST endpoints."""
    headers, user = auth_headers

    # Context Aggregation
    ctx_resp = client.post("/api/context/departure", json={
        "event_name": "AI Systems Meeting",
        "destination": "Tech Park Suite 400",
        "current_location": "Residence",
        "bag_prepared": True
    }, headers=headers)
    assert ctx_resp.status_code == 200
    ctx_data = ctx_resp.json()
    assert ctx_data["user_id"] == user.id
    assert ctx_data["destination"] == "Tech Park Suite 400"
    assert "source_metadata" in ctx_data
    assert "behavioral" in ctx_data["source_metadata"]

    # Optimize Departure
    opt_resp = client.post("/api/departure/optimize", json={
        "event_name": "AI Systems Meeting",
        "destination": "Tech Park Suite 400",
        "bag_prepared": True
    }, headers=headers)
    assert opt_resp.status_code == 200
    opt_data = opt_resp.json()
    assert "recommended_departure_time" in opt_data
    assert "recommended_prep_start_time" in opt_data
    assert "timeline" in opt_data
    assert len(opt_data["timeline"]) == 4


def test_api_journey_tracking_endpoints(client: TestClient, auth_headers):
    """Test /api/journey/start and /api/journey/{journey_id}/transition REST endpoints."""
    headers, user = auth_headers

    # Start journey
    start_resp = client.post("/api/journey/start", json={
        "event_name": "Lab Demo",
        "destination": "Lab 3"
    }, headers=headers)
    assert start_resp.status_code == 200
    j_data = start_resp.json()
    journey_id = j_data["journey_id"]
    assert j_data["phase"] == "scheduled"

    # Transition to prep_started
    tr_resp = client.post(f"/api/journey/{journey_id}/transition", json={
        "target_phase": "prep_started"
    }, headers=headers)
    assert tr_resp.status_code == 200
    assert tr_resp.json()["phase"] == "prep_started"

    # Get status
    get_resp = client.get(f"/api/journey/{journey_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["journey_id"] == journey_id


# =========================================================================
# 12. Backward Compatibility with Phase 1 & Phase 2 APIs
# =========================================================================

def test_phase1_and_phase2_endpoints_backward_compatibility(client: TestClient, auth_headers):
    """Ensure existing prediction and intelligence APIs continue to work unmodified."""
    headers, user = auth_headers

    # Phase 1 predict endpoint
    resp1 = client.get("/api/predict?event_name=Seminar&day_of_week=Tuesday&estimated_prep=30&estimated_travel=20&importance=medium&bag_prepared=false", headers=headers)
    assert resp1.status_code == 200
    assert "predicted_delay_min" in resp1.json()

    # Phase 2 intelligence predict endpoint
    resp2 = client.get("/api/intelligence/predict?event_name=Seminar&day_of_week=Tuesday&estimated_prep=30&estimated_travel=20&importance=medium&bag_prepared=false", headers=headers)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "predicted_delay_minutes" in data2 or "predicted_delay_min" in data2
    assert "risk_category" in data2
    assert "confidence_status" in data2
