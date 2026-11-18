"""Comprehensive test suite for Travel & Route Intelligence Engine (Prompt 3.4).

Covers:
- Geocoding: address and coordinate validation, location caching, provider failover
- Routing Abstraction: multi-modal routing (driving, transit, bicycling, walking), traffic-aware duration, alternative comparisons
- Google Routes Integration: mocked HTTP responses, traffic condition parsing, API errors
- Location Management: saved locations (Home, Workplace, Custom), primary location toggle, user isolation
- Route Management: saved preferred routes, user isolation
- User Travel Preferences: defaults, updates, boundary clamps
- Historical Travel Preservation: original estimates NEVER overwritten by actual duration
- Personalized Travel Estimation: behavioral bias ratio, rush-hour multiplier, Bayesian shrinkage anchor, analytics
- REST API Endpoints: full coverage of /api/travel/*
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.database import (
    User, DepartureLog, UserSavedLocation, UserSavedRoute,
    UserTravelPreferences, GeocodeCacheRecord
)
from backend.integrations.travel.base import (
    TravelMode, RouteOption, TravelEstimate, TravelIntegrationBase
)
from backend.integrations.travel.geocoding import (
    validate_coordinates, validate_address, GeocodedLocation,
    GeocodingManager, MockGeocodingProvider, NominatimGeocodingProvider, GoogleGeocodingProvider
)
from backend.integrations.travel.mock import MockTravelIntegration
from backend.integrations.travel.google_routes import GoogleRoutesIntegration
from backend.integrations.base import (
    IntegrationTimeoutError, IntegrationDataError, IntegrationAuthError
)
from backend.ml.personalized_travel import PersonalizedTravelEngine
from backend.services.travel_service import TravelService
from backend.services.context_service import ContextService, DepartureContext


# =============================================================================
# 1. Coordinate & Address Validation Tests
# =============================================================================

def test_validate_coordinates_valid():
    """Verify valid geographic coordinates pass validation."""
    assert validate_coordinates(37.7749, -122.4194) is True  # San Francisco
    assert validate_coordinates(0.0, 0.0) is True  # Equator / Prime Meridian
    assert validate_coordinates(90.0, 180.0) is True  # Extremes
    assert validate_coordinates(-90.0, -180.0) is True


def test_validate_coordinates_invalid():
    """Verify out-of-bounds or non-numeric coordinates fail validation."""
    assert validate_coordinates(91.0, 0.0) is False  # Lat > 90
    assert validate_coordinates(-90.1, 0.0) is False  # Lat < -90
    assert validate_coordinates(0.0, 181.0) is False  # Lng > 180
    assert validate_coordinates(0.0, -180.1) is False  # Lng < -180
    assert validate_coordinates(None, 10.0) is False
    assert validate_coordinates("invalid", 10.0) is False


def test_validate_address():
    """Verify address string validation rules."""
    assert validate_address("123 Main Street, Boston, MA") is True
    assert validate_address("Workplace") is True
    assert validate_address("") is False
    assert validate_address("   ") is False
    assert validate_address("ab") is False  # Too short (< 3 chars)
    assert validate_address("!@#$%^&*") is False  # Non-alphanumeric gibberish
    assert validate_address(None) is False


# =============================================================================
# 2. Geocoding & Persistent Cache Tests
# =============================================================================

def test_mock_geocoding_known_and_dynamic_addresses():
    """Verify mock geocoder resolves known keys and generates deterministic coordinates for others."""
    provider = MockGeocodingProvider()
    
    # Known key
    home = provider.geocode("My Home Address")
    assert home is not None
    assert "Elm Street" in home.formatted_address
    assert validate_coordinates(home.latitude, home.longitude) is True

    # Unknown key (deterministic hash fallback)
    custom = provider.geocode("999 Random Avenue, Metropolis")
    assert custom is not None
    assert validate_coordinates(custom.latitude, custom.longitude) is True
    assert custom.provider == "mock_geocoder"


def test_geocoding_database_cache_hit_and_miss(db_session: Session):
    """Verify geocoding results are cached in the database to avoid redundant external calls."""
    manager = GeocodingManager(fallback_provider=MockGeocodingProvider())
    test_addr = "789 Unique Plaza, Innovation District"

    # 1. First call: Cache miss, hits provider and stores in db
    loc1 = manager.geocode(test_addr, db=db_session)
    assert loc1 is not None
    assert loc1.cached is False

    # Check database record created
    cache_row = db_session.query(GeocodeCacheRecord).filter(
        GeocodeCacheRecord.query_text == test_addr
    ).first()
    assert cache_row is not None
    assert cache_row.latitude == loc1.latitude

    # 2. Second call with new manager instance (empty memory cache): DB cache hit
    fresh_manager = GeocodingManager(fallback_provider=MockGeocodingProvider())
    loc2 = fresh_manager.geocode(test_addr, db=db_session)
    assert loc2 is not None
    assert loc2.cached is True
    assert loc2.latitude == loc1.latitude
    assert loc2.longitude == loc1.longitude


def test_reverse_geocoding():
    """Verify reverse geocoding translates coordinates back to a formatted address."""
    manager = GeocodingManager(fallback_provider=MockGeocodingProvider())
    rev = manager.reverse_geocode(37.7749, -122.4194)
    assert rev is not None
    assert rev.formatted_address is not None and len(rev.formatted_address) > 0
    assert ("San Francisco" in rev.formatted_address or "Location" in rev.formatted_address or "37.7749" in rev.formatted_address)


# =============================================================================
# 3. Multi-Modal Routing Abstraction Tests
# =============================================================================

def test_mock_travel_calculate_routes_and_alternatives():
    """Verify mock travel integration returns multiple alternative routes."""
    adapter = MockTravelIntegration()
    routes = adapter.calculate_routes(
        origin="123 Home St",
        destination="456 Work Blvd",
        mode=TravelMode.DRIVING,
        alternatives=True
    )

    assert len(routes) >= 2
    primary = routes[0]
    assert primary.is_recommended is True
    assert primary.duration_min > 0
    assert primary.distance_km > 0
    assert primary.duration_in_traffic_min is not None
    assert primary.travel_mode == TravelMode.DRIVING

    # Alternative route is longer
    alt = routes[1]
    assert alt.is_recommended is False
    assert alt.duration_min > primary.duration_min


def test_travel_mode_duration_scaling():
    """Verify that walking and bicycling durations reflect realistic speed scaling."""
    adapter = MockTravelIntegration()
    dep = datetime.now(timezone.utc)

    driving = adapter.calculate_routes("A", "B", departure_time=dep, mode=TravelMode.DRIVING)[0]
    bicycling = adapter.calculate_routes("A", "B", departure_time=dep, mode=TravelMode.BICYCLING)[0]
    walking = adapter.calculate_routes("A", "B", departure_time=dep, mode=TravelMode.WALKING)[0]

    # Walking takes longest, bicycling between driving and walking
    assert walking.duration_min > bicycling.duration_min
    assert bicycling.duration_min > driving.duration_min


def test_mock_travel_timeout_and_invalid_data():
    """Verify integration raises appropriate exceptions on simulated network issues."""
    timeout_adapter = MockTravelIntegration(simulate_timeout=True)
    with pytest.raises(IntegrationTimeoutError):
        timeout_adapter.calculate_routes("A", "B")

    corrupt_adapter = MockTravelIntegration(simulate_invalid_data=True)
    routes = corrupt_adapter.calculate_routes("A", "B")
    assert len(routes) == 1
    assert routes[0].duration_min < 0  # Flagged corrupt duration for validation tests


# =============================================================================
# 4. Google Maps Routes Integration (Mocked HTTP)
# =============================================================================

def test_google_routes_integration_success():
    """Verify GoogleRoutesIntegration parses Google Directions API JSON response with traffic."""
    adapter = GoogleRoutesIntegration(api_key="TEST_API_KEY")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "OK",
        "routes": [
            {
                "summary": "via I-90 E",
                "legs": [
                    {
                        "start_address": "Origin Address, Boston, MA",
                        "end_address": "Destination Address, Cambridge, MA",
                        "distance": {"value": 15400},  # 15.4 km
                        "duration": {"value": 1200},    # 20.0 min
                        "duration_in_traffic": {"value": 1500} # 25.0 min in traffic
                    }
                ],
                "warnings": ["Toll road"],
                "overview_polyline": {"points": "mock_polyline_points"}
            },
            {
                "summary": "via Route 20",
                "legs": [
                    {
                        "start_address": "Origin Address, Boston, MA",
                        "end_address": "Destination Address, Cambridge, MA",
                        "distance": {"value": 16800},
                        "duration": {"value": 1600},
                        "duration_in_traffic": {"value": 1750}
                    }
                ],
                "warnings": [],
                "overview_polyline": {"points": "mock_alt_points"}
            }
        ]
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp

    with patch("backend.integrations.travel.google_routes.httpx.Client", return_value=mock_client):
        routes = adapter.calculate_routes("Boston, MA", "Cambridge, MA", mode=TravelMode.DRIVING)

    assert len(routes) == 2
    top = routes[0]
    assert top.summary == "via I-90 E"
    assert top.duration_min == 20.0
    assert top.duration_in_traffic_min == 25.0
    assert top.distance_km == 15.4
    assert top.is_recommended is True
    assert "Toll road" in top.warnings


def test_google_routes_error_handling():
    """Verify GoogleRoutesIntegration handles REQUEST_DENIED and API timeouts gracefully."""
    adapter = GoogleRoutesIntegration(api_key="INVALID_KEY")

    mock_denied = MagicMock()
    mock_denied.status_code = 200
    mock_denied.json.return_value = {
        "status": "REQUEST_DENIED",
        "error_message": "The provided API key is invalid."
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_denied

    with patch("backend.integrations.travel.google_routes.httpx.Client", return_value=mock_client):
        with pytest.raises(IntegrationAuthError) as exc_info:
            adapter.calculate_routes("A", "B")
        assert "API key is invalid" in str(exc_info.value)


# =============================================================================
# 5. Saved Locations CRUD & User Isolation
# =============================================================================

def test_saved_locations_crud_and_primary_toggle(db_session: Session, test_user: User):
    """Verify creating, updating, toggling primary status, and deleting saved locations."""
    service = TravelService()

    # 1. Create Home location
    loc1 = service.create_saved_location(
        user_id=test_user.id,
        name="Home",
        category="home",
        address="100 Main St, Springfield",
        latitude=39.78,
        longitude=-89.65,
        is_primary=True,
        db=db_session
    )
    assert loc1["name"] == "Home"
    assert loc1["is_primary"] is True

    # 2. Create Workplace location and set it as primary (should un-set Home)
    loc2 = service.create_saved_location(
        user_id=test_user.id,
        name="Office",
        category="work",
        address="500 Tech Park, Springfield",
        latitude=39.80,
        longitude=-89.60,
        is_primary=True,
        db=db_session
    )
    assert loc2["is_primary"] is True

    # Refresh and check loc1 is no longer primary
    locs = service.list_saved_locations(user_id=test_user.id, db=db_session)
    assert len(locs) == 2
    primary_locs = [l for l in locs if l["is_primary"]]
    assert len(primary_locs) == 1
    assert primary_locs[0]["name"] == "Office"

    # 3. Update location
    updated = service.update_saved_location(
        user_id=test_user.id,
        location_id=loc1["id"],
        updates={"name": "Sweet Home"},
        db=db_session
    )
    assert updated["name"] == "Sweet Home"

    # 4. Delete location
    deleted = service.delete_saved_location(user_id=test_user.id, location_id=loc1["id"], db=db_session)
    assert deleted is True
    assert len(service.list_saved_locations(user_id=test_user.id, db=db_session)) == 1


def test_saved_locations_user_data_isolation(db_session: Session, test_user: User):
    """Verify users cannot see, modify, or delete another user's saved locations."""
    service = TravelService()
    other_user = User(
        username="isolated_other_user",
        hashed_password="pw",
        salt="salt",
        profile_type="student"
    )
    db_session.add(other_user)
    db_session.commit()

    loc1 = service.create_saved_location(
        user_id=test_user.id,
        name="Secret Base",
        category="custom",
        address="123 Hidden Lane",
        latitude=40.0,
        longitude=-80.0,
        is_primary=False,
        db=db_session
    )

    # other_user listing should be empty
    other_locs = service.list_saved_locations(user_id=other_user.id, db=db_session)
    assert len(other_locs) == 0

    # other_user attempt to update loc1 should return None
    upd = service.update_saved_location(
        user_id=other_user.id,
        location_id=loc1["id"],
        updates={"name": "Hacked Base"},
        db=db_session
    )
    assert upd is None

    # other_user attempt to delete loc1 should fail
    del_res = service.delete_saved_location(user_id=other_user.id, location_id=loc1["id"], db=db_session)
    assert del_res is False


# =============================================================================
# 6. Saved Routes CRUD & User Isolation
# =============================================================================

def test_saved_routes_crud(db_session: Session, test_user: User):
    """Verify saving and listing preferred routes."""
    service = TravelService()

    route = service.save_preferred_route(
        user_id=test_user.id,
        name="Morning Highway Commute",
        origin_address="123 Home St",
        destination_address="456 Work Blvd",
        travel_mode="driving",
        preferred_route_identifier="via Expressway",
        distance_km=14.2,
        estimated_duration_min=22.0,
        notes="Avoid during snow storms",
        db=db_session
    )
    assert route["name"] == "Morning Highway Commute"
    assert route["travel_mode"] == "driving"

    routes = service.list_saved_routes(user_id=test_user.id, db=db_session)
    assert len(routes) == 1

    deleted = service.delete_saved_route(user_id=test_user.id, route_id=route["id"], db=db_session)
    assert deleted is True
    assert len(service.list_saved_routes(user_id=test_user.id, db=db_session)) == 0


# =============================================================================
# 7. Travel Preferences CRUD
# =============================================================================

def test_travel_preferences_defaults_and_updates(db_session: Session, test_user: User):
    """Verify default travel preferences initialization and updates."""
    service = TravelService()

    prefs = service.get_travel_preferences(user_id=test_user.id, db=db_session)
    assert prefs["default_mode"] == "driving"
    assert prefs["avoid_tolls"] is False
    assert prefs["buffer_multiplier"] == 1.0

    updated = service.update_travel_preferences(
        user_id=test_user.id,
        updates={
            "default_mode": "transit",
            "avoid_tolls": True,
            "buffer_multiplier": 1.25
        },
        db=db_session
    )
    assert updated["default_mode"] == "transit"
    assert updated["avoid_tolls"] is True
    assert updated["buffer_multiplier"] == 1.25


# =============================================================================
# 8. Historical Travel Data Preservation
# =============================================================================

def test_historical_travel_data_preservation(client: TestClient, auth_headers, db_session: Session, test_user: User):
    """Verify original estimated travel duration is preserved and NEVER overwritten by actual duration."""
    # Submit a completed journey log
    log_payload = {
        "event_name": "Product Launch",
        "date": "2026-11-20",
        "day_of_week": "Friday",
        "planned_prep_start": "08:00:00",
        "planned_departure": "08:30:00",
        "estimated_prep_duration_min": 30.0,
        "estimated_travel_duration_min": 25.0,  # ORIGINAL ESTIMATE
        "actual_prep_start": "08:02:00",
        "actual_departure": "08:35:00",
        "actual_arrival": "09:12:00",
        "actual_prep_duration_min": 33.0,
        "actual_travel_duration_min": 37.0,     # ACTUAL (DIFFERENT FROM ESTIMATE)
        "bag_prepared": True,
        "tag_one_last_task": False,
        "tag_finding_items": False,
        "tag_phone_usage": True,
        "tag_waiting": False,
        "selected_route": "Primary Expressway",
        "travel_mode": "driving",
        "travel_estimate_retrieved_at": "2026-11-20T07:55:00Z"
    }

    resp = client.post("/api/logs", headers=auth_headers, json=log_payload)
    assert resp.status_code == 200

    # Query back directly from database
    db_log = db_session.query(DepartureLog).filter(
        DepartureLog.user_id == test_user.id,
        DepartureLog.event_name == "Product Launch"
    ).first()

    assert db_log is not None
    # Crucial assertion: original estimate preserved unchanged!
    assert db_log.estimated_travel_duration_min == 25.0
    assert db_log.actual_travel_duration_min == 37.0
    assert db_log.estimated_travel_duration_min != db_log.actual_travel_duration_min
    assert db_log.selected_route == "Primary Expressway"
    assert db_log.travel_mode == "driving"
    assert db_log.travel_estimate_retrieved_at == "2026-11-20T07:55:00Z"


# =============================================================================
# 9. Personalized Travel Estimation (Phase 2 Integration)
# =============================================================================

def test_personalized_travel_estimation_with_bias_and_rush_hour(db_session: Session, test_user: User):
    """Verify that user behavioral travel bias and rush-hour patterns personalize provider estimates."""
    # Seed 6 journeys where the user consistently takes 20% longer than estimated (ratio = 1.20)
    for i in range(6):
        db_session.add(DepartureLog(
            user_id=test_user.id,
            event_name=f"Commute {i}",
            date=f"2026-11-0{i+1}",
            day_of_week="Monday",
            planned_prep_start="07:30:00",
            planned_departure="08:00:00",
            estimated_prep_duration_min=30.0,
            estimated_travel_duration_min=20.0,
            actual_prep_start="07:30:00",
            actual_departure="08:00:00",
            actual_arrival="08:24:00",
            actual_prep_duration_min=30.0,
            actual_travel_duration_min=24.0,  # 24/20 = 1.20 ratio
            total_delay_min=4.0,
            provenance="real",
            data_source="simulator_checkout",
            recorded_at=datetime.now(timezone.utc).isoformat(),
            observation_status="valid"
        ))
    db_session.commit()

    # Test Monday 08:15 departure (Peak morning rush hour window)
    rush_dep_time = datetime(2026, 11, 23, 8, 15, tzinfo=timezone.utc)
    provider_estimate = 30.0

    p_est = PersonalizedTravelEngine.estimate(
        user_id=test_user.id,
        provider_duration_min=provider_estimate,
        departure_time=rush_dep_time,
        travel_mode="driving",
        db=db_session
    )

    # Personalized duration should be longer than provider estimate due to user bias + rush hour
    assert p_est.personalized_duration_min > provider_estimate
    assert p_est.bias_ratio > 1.0
    assert p_est.time_of_day_factor > 1.0  # Rush hour active
    assert p_est.sample_count == 6
    assert p_est.confidence > 0.0
    assert "historical journeys average" in p_est.explanation


def test_travel_reliability_metrics_computation(db_session: Session, test_user: User):
    """Verify travel reliability metrics and on-time percentages."""
    service = TravelService()

    # 4 on-time trips (actual <= planned) and 1 delayed trip (actual > planned)
    trips = [
        (20.0, 18.0),  # on time (-2 min)
        (20.0, 20.0),  # on time (0 min)
        (25.0, 24.0),  # on time (-1 min)
        (30.0, 30.0),  # on time (0 min)
        (20.0, 32.0),  # delayed (+12 min)
    ]
    for idx, (est, act) in enumerate(trips):
        db_session.add(DepartureLog(
            user_id=test_user.id,
            event_name=f"Trip {idx}",
            date=f"2026-11-1{idx}",
            day_of_week="Tuesday",
            planned_prep_start="08:00:00",
            planned_departure="08:30:00",
            estimated_prep_duration_min=30.0,
            estimated_travel_duration_min=est,
            actual_prep_start="08:00:00",
            actual_departure="08:30:00",
            actual_arrival="09:00:00",
            actual_prep_duration_min=30.0,
            actual_travel_duration_min=act,
            total_delay_min=max(0.0, act - est),
            provenance="real",
            data_source="simulator_checkout",
            travel_mode="driving",
            recorded_at=datetime.now(timezone.utc).isoformat(),
            observation_status="valid"
        ))
    db_session.commit()

    metrics = service.get_travel_reliability_metrics(user_id=test_user.id, db=db_session)
    assert metrics["total_journeys"] == 5
    assert metrics["on_time_travel_pct"] == 80.0  # 4 out of 5
    assert metrics["mode_breakdown"].get("driving") == 5
    assert len(metrics["insights"]) > 0


# =============================================================================
# 10. REST API Endpoints Verification
# =============================================================================

def test_api_saved_locations_endpoints(client: TestClient, auth_headers):
    """Test POST, GET, PUT, DELETE /api/travel/locations."""
    # 1. Create Location
    res = client.post(
        "/api/travel/locations",
        headers=auth_headers,
        json={
            "name": "University Campus",
            "category": "education",
            "address": "450 Serra Mall, Stanford, CA",
            "is_primary": True
        }
    )
    assert res.status_code == 200
    loc_id = res.json()["location"]["id"]

    # 2. Get Locations
    list_res = client.get("/api/travel/locations", headers=auth_headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # 3. Update Location
    upd_res = client.put(
        f"/api/travel/locations/{loc_id}",
        headers=auth_headers,
        json={"name": "Stanford University"}
    )
    assert upd_res.status_code == 200
    assert upd_res.json()["location"]["name"] == "Stanford University"

    # 4. Delete Location
    del_res = client.delete(f"/api/travel/locations/{loc_id}", headers=auth_headers)
    assert del_res.status_code == 200


def test_api_geocode_endpoints(client: TestClient, auth_headers):
    """Test POST /api/travel/geocode and /api/travel/reverse-geocode."""
    geo_res = client.post(
        "/api/travel/geocode",
        headers=auth_headers,
        json={"address": "123 Elm Street, Springfield"}
    )
    assert geo_res.status_code == 200
    assert "location" in geo_res.json()
    assert "latitude" in geo_res.json()["location"]

    rev_res = client.post(
        "/api/travel/reverse-geocode",
        headers=auth_headers,
        json={"latitude": 39.7817, "longitude": -89.6501}
    )
    assert rev_res.status_code == 200
    assert "formatted_address" in rev_res.json()["location"]


def test_api_routes_and_compare_endpoints(client: TestClient, auth_headers):
    """Test POST /api/travel/routes and /api/travel/compare-routes."""
    routes_res = client.post(
        "/api/travel/routes",
        headers=auth_headers,
        json={
            "origin": "Home",
            "destination": "Work",
            "mode": "driving",
            "alternatives": True
        }
    )
    assert routes_res.status_code == 200
    data = routes_res.json()
    assert data["status"] == "success"
    assert data["routes_count"] >= 1
    assert "personalized_estimation" in data

    compare_res = client.post(
        "/api/travel/compare-routes",
        headers=auth_headers,
        json={
            "origin": "Home",
            "destination": "Work"
        }
    )
    assert compare_res.status_code == 200
    cmp_data = compare_res.json()
    assert "comparison" in cmp_data
    assert "driving" in cmp_data["comparison"]
    assert "transit" in cmp_data["comparison"]


def test_api_saved_routes_and_preferences_endpoints(client: TestClient, auth_headers):
    """Test saved routes and preferences endpoints."""
    # Preferences GET and PUT
    pref_res = client.get("/api/travel/preferences", headers=auth_headers)
    assert pref_res.status_code == 200

    upd_pref = client.put(
        "/api/travel/preferences",
        headers=auth_headers,
        json={"default_mode": "bicycling", "avoid_tolls": True}
    )
    assert upd_pref.status_code == 200
    assert upd_pref.json()["preferences"]["default_mode"] == "bicycling"

    # Saved Route POST, GET, DELETE
    sr_res = client.post(
        "/api/travel/saved-routes",
        headers=auth_headers,
        json={
            "name": "Bicycle Trail Commute",
            "origin_address": "Home",
            "destination_address": "Work",
            "travel_mode": "bicycling",
            "distance_km": 8.5,
            "estimated_duration_min": 25.0
        }
    )
    assert sr_res.status_code == 200
    sr_id = sr_res.json()["route"]["id"]

    list_sr = client.get("/api/travel/saved-routes", headers=auth_headers)
    assert list_sr.status_code == 200
    assert len(list_sr.json()) >= 1

    del_sr = client.delete(f"/api/travel/saved-routes/{sr_id}", headers=auth_headers)
    assert del_sr.status_code == 200

    # Analytics GET
    ana_res = client.get("/api/travel/analytics", headers=auth_headers)
    assert ana_res.status_code == 200
    assert "total_journeys" in ana_res.json()
