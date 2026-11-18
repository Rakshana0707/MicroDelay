"""Travel and Route Intelligence Service (Prompt 3.4).

Coordinating route estimation, multi-modal comparison, geocoding validation,
saved locations/routes, user preferences, and personalized behavioral travel modeling.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func

try:
    from backend.integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationError,
        IntegrationTimeoutError, IntegrationDataError
    )
    from backend.integrations.registry import IntegrationRegistry
    from backend.integrations.travel.base import (
        TravelIntegrationBase, TravelEstimate, TravelMode, RouteOption
    )
    from backend.integrations.traffic.base import (
        TrafficIntegrationBase, TrafficObservation, TrafficCongestionLevel
    )
    from backend.integrations.travel.geocoding import (
        GeocodingManager, validate_coordinates, validate_address
    )
    from backend.ml.personalized_travel import PersonalizedTravelEngine
    from backend.database import (
        DepartureLog, Event, UserSavedLocation, UserSavedRoute,
        UserTravelPreferences
    )
except ImportError:
    from integrations.base import (
        ObservationType, ExternalObservationMetadata, IntegrationError,
        IntegrationTimeoutError, IntegrationDataError
    )
    from integrations.registry import IntegrationRegistry
    from integrations.travel.base import (
        TravelIntegrationBase, TravelEstimate, TravelMode, RouteOption
    )
    from integrations.traffic.base import (
        TrafficIntegrationBase, TrafficObservation, TrafficCongestionLevel
    )
    from integrations.travel.geocoding import (
        GeocodingManager, validate_coordinates, validate_address
    )
    from ml.personalized_travel import PersonalizedTravelEngine
    from database import (
        DepartureLog, Event, UserSavedLocation, UserSavedRoute,
        UserTravelPreferences
    )

logger = logging.getLogger("microdelay.services.travel")


class TravelService:
    """Manages travel-time estimation, live traffic adjustments, and graceful degradation."""

    def __init__(
        self,
        registry: Optional[IntegrationRegistry] = None,
        geocoding_manager: Optional[GeocodingManager] = None
    ):
        self.registry = registry or IntegrationRegistry.get_instance()
        self.geocoding_manager = geocoding_manager or GeocodingManager()

    # =========================================================================
    # 1. GEOCODING & ADDRESS VALIDATION
    # =========================================================================

    def geocode_address(self, address: str, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """Geocode address to coordinates with cache and validation."""
        if not validate_address(address):
            return None
        res = self.geocoding_manager.geocode(address, db=db)
        return res.to_dict() if res else None

    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """Reverse geocode coordinates to formatted address."""
        if not validate_coordinates(latitude, longitude):
            return None
        res = self.geocoding_manager.reverse_geocode(latitude, longitude)
        return res.to_dict() if res else None

    # =========================================================================
    # 2. USER SAVED LOCATIONS (Home, Workplace, Frequent, Custom)
    # =========================================================================

    def list_saved_locations(self, user_id: int, db: Session) -> List[Dict[str, Any]]:
        """List all saved locations for a user."""
        locations = db.query(UserSavedLocation).filter(
            UserSavedLocation.user_id == user_id
        ).order_by(UserSavedLocation.is_primary.desc(), UserSavedLocation.name.asc()).all()

        return [
            {
                "id": loc.id,
                "user_id": loc.user_id,
                "name": loc.name,
                "category": loc.category,
                "address": loc.address,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "is_primary": loc.is_primary,
                "created_at": loc.created_at,
                "updated_at": loc.updated_at
            }
            for loc in locations
        ]

    def create_saved_location(
        self,
        user_id: int,
        name: str,
        category: str,
        address: str,
        latitude: Optional[float],
        longitude: Optional[float],
        is_primary: bool,
        db: Session
    ) -> Dict[str, Any]:
        """Create and geocode (if needed) a new saved location."""
        if not validate_address(address):
            raise ValueError(f"Invalid address provided: '{address}'")

        # Auto-geocode coordinates if missing
        if not validate_coordinates(latitude, longitude):
            geo = self.geocode_address(address, db=db)
            if geo:
                latitude = geo["latitude"]
                longitude = geo["longitude"]

        now_str = datetime.now(timezone.utc).isoformat()

        # If primary, reset other locations of this user
        if is_primary:
            db.query(UserSavedLocation).filter(
                UserSavedLocation.user_id == user_id
            ).update({"is_primary": False})

        loc = UserSavedLocation(
            user_id=user_id,
            name=name.strip(),
            category=category.strip().lower(),
            address=address.strip(),
            latitude=latitude,
            longitude=longitude,
            is_primary=is_primary,
            created_at=now_str,
            updated_at=now_str
        )
        db.add(loc)
        db.commit()
        db.refresh(loc)

        return {
            "id": loc.id,
            "user_id": loc.user_id,
            "name": loc.name,
            "category": loc.category,
            "address": loc.address,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "is_primary": loc.is_primary,
            "created_at": loc.created_at,
            "updated_at": loc.updated_at
        }

    def update_saved_location(
        self,
        user_id: int,
        location_id: int,
        updates: Dict[str, Any],
        db: Session
    ) -> Optional[Dict[str, Any]]:
        """Update an existing saved location."""
        loc = db.query(UserSavedLocation).filter(
            UserSavedLocation.id == location_id,
            UserSavedLocation.user_id == user_id
        ).first()
        if not loc:
            return None

        if "name" in updates and updates["name"]:
            loc.name = str(updates["name"]).strip()
        if "category" in updates and updates["category"]:
            loc.category = str(updates["category"]).strip().lower()
        if "address" in updates and updates["address"]:
            new_addr = str(updates["address"]).strip()
            if validate_address(new_addr):
                loc.address = new_addr
                # Geocode new address if coordinates not supplied
                if "latitude" not in updates or "longitude" not in updates:
                    geo = self.geocode_address(new_addr, db=db)
                    if geo:
                        loc.latitude = geo["latitude"]
                        loc.longitude = geo["longitude"]

        if "latitude" in updates and "longitude" in updates:
            if validate_coordinates(updates["latitude"], updates["longitude"]):
                loc.latitude = float(updates["latitude"])
                loc.longitude = float(updates["longitude"])

        if "is_primary" in updates and updates["is_primary"]:
            db.query(UserSavedLocation).filter(
                UserSavedLocation.user_id == user_id,
                UserSavedLocation.id != location_id
            ).update({"is_primary": False})
            loc.is_primary = True
        elif "is_primary" in updates:
            loc.is_primary = bool(updates["is_primary"])

        loc.updated_at = datetime.now(timezone.utc).isoformat()
        db.commit()
        db.refresh(loc)

        return {
            "id": loc.id,
            "user_id": loc.user_id,
            "name": loc.name,
            "category": loc.category,
            "address": loc.address,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "is_primary": loc.is_primary,
            "created_at": loc.created_at,
            "updated_at": loc.updated_at
        }

    def delete_saved_location(self, user_id: int, location_id: int, db: Session) -> bool:
        """Delete a saved location ensuring user isolation."""
        loc = db.query(UserSavedLocation).filter(
            UserSavedLocation.id == location_id,
            UserSavedLocation.user_id == user_id
        ).first()
        if not loc:
            return False
        db.delete(loc)
        db.commit()
        return True

    # =========================================================================
    # 3. ROUTE CALCULATION & ALTERNATIVE COMPARISON
    # =========================================================================

    def calculate_routes(
        self,
        user_id: int,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        mode: TravelMode = TravelMode.DRIVING,
        alternatives: bool = True,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Calculate primary and alternative routes between origin and destination."""
        if not origin or not destination:
            raise ValueError("Origin and destination must both be specified.")

        travel_integration: Optional[TravelIntegrationBase] = self.registry.get("travel")
        if not travel_integration or not travel_integration.is_enabled():
            raise IntegrationError("Routing service integration is not configured or disabled.")

        dep_dt = departure_time or datetime.now(timezone.utc)
        routes = travel_integration.calculate_routes(
            origin=origin,
            destination=destination,
            departure_time=dep_dt,
            mode=mode,
            alternatives=alternatives,
            applicable_id=str(user_id)
        )

        # Compute personalized estimate for the primary route if db provided
        personalized_info = None
        if db is not None and routes:
            primary = routes[0]
            eff_dur = primary.duration_in_traffic_min or primary.duration_min
            p_est = PersonalizedTravelEngine.estimate(
                user_id=user_id,
                provider_duration_min=eff_dur,
                departure_time=dep_dt,
                travel_mode=mode.value if isinstance(mode, TravelMode) else str(mode),
                db=db
            )
            personalized_info = p_est.to_dict()

        return {
            "origin": origin,
            "destination": destination,
            "travel_mode": mode.value if isinstance(mode, TravelMode) else str(mode),
            "departure_time": dep_dt.isoformat(),
            "routes_count": len(routes),
            "routes": [r.to_dict() for r in routes],
            "personalized_estimation": personalized_info
        }

    def compare_modes(
        self,
        user_id: int,
        origin: str,
        destination: str,
        departure_time: Optional[datetime] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Compare route duration, distance, and traffic across all available travel modes."""
        modes = [TravelMode.DRIVING, TravelMode.TRANSIT, TravelMode.BICYCLING, TravelMode.WALKING]
        results = {}

        for m in modes:
            try:
                res = self.calculate_routes(
                    user_id=user_id,
                    origin=origin,
                    destination=destination,
                    departure_time=departure_time,
                    mode=m,
                    alternatives=False,
                    db=db
                )
                if res.get("routes"):
                    top = res["routes"][0]
                    results[m.value] = {
                        "available": True,
                        "duration_min": top["duration_min"],
                        "duration_in_traffic_min": top.get("duration_in_traffic_min"),
                        "distance_km": top["distance_km"],
                        "summary": top["summary"]
                    }
                else:
                    results[m.value] = {"available": False, "reason": "No route found"}
            except Exception as e:
                logger.debug(f"Mode comparison for {m.value} failed: {e}")
                results[m.value] = {"available": False, "reason": str(e)}

        return {
            "origin": origin,
            "destination": destination,
            "comparison": results
        }

    # =========================================================================
    # 4. USER SAVED ROUTES (Preferred corridors)
    # =========================================================================

    def list_saved_routes(self, user_id: int, db: Session) -> List[Dict[str, Any]]:
        """List user's saved preferred routes."""
        routes = db.query(UserSavedRoute).filter(
            UserSavedRoute.user_id == user_id
        ).order_by(UserSavedRoute.created_at.desc()).all()

        return [
            {
                "id": r.id,
                "user_id": r.user_id,
                "name": r.name,
                "origin_address": r.origin_address,
                "origin_lat": r.origin_lat,
                "origin_lng": r.origin_lng,
                "destination_address": r.destination_address,
                "destination_lat": r.destination_lat,
                "destination_lng": r.destination_lng,
                "travel_mode": r.travel_mode,
                "preferred_route_identifier": r.preferred_route_identifier,
                "distance_km": r.distance_km,
                "estimated_duration_min": r.estimated_duration_min,
                "notes": r.notes,
                "created_at": r.created_at,
                "updated_at": r.updated_at
            }
            for r in routes
        ]

    def save_preferred_route(
        self,
        user_id: int,
        name: str,
        origin_address: str,
        destination_address: str,
        travel_mode: str,
        preferred_route_identifier: Optional[str],
        distance_km: Optional[float],
        estimated_duration_min: Optional[float],
        notes: Optional[str],
        db: Session,
        origin_lat: Optional[float] = None,
        origin_lng: Optional[float] = None,
        destination_lat: Optional[float] = None,
        destination_lng: Optional[float] = None
    ) -> Dict[str, Any]:
        """Save a preferred route choice."""
        now_str = datetime.now(timezone.utc).isoformat()
        route = UserSavedRoute(
            user_id=user_id,
            name=name.strip(),
            origin_address=origin_address.strip(),
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            destination_address=destination_address.strip(),
            destination_lat=destination_lat,
            destination_lng=destination_lng,
            travel_mode=travel_mode,
            preferred_route_identifier=preferred_route_identifier,
            distance_km=distance_km,
            estimated_duration_min=estimated_duration_min,
            notes=notes,
            created_at=now_str,
            updated_at=now_str
        )
        db.add(route)
        db.commit()
        db.refresh(route)

        return {
            "id": route.id,
            "user_id": route.user_id,
            "name": route.name,
            "origin_address": route.origin_address,
            "destination_address": route.destination_address,
            "travel_mode": route.travel_mode,
            "preferred_route_identifier": route.preferred_route_identifier,
            "distance_km": route.distance_km,
            "estimated_duration_min": route.estimated_duration_min,
            "notes": route.notes,
            "created_at": route.created_at,
            "updated_at": route.updated_at
        }

    def delete_saved_route(self, user_id: int, route_id: int, db: Session) -> bool:
        """Delete a saved route ensuring user isolation."""
        r = db.query(UserSavedRoute).filter(
            UserSavedRoute.id == route_id,
            UserSavedRoute.user_id == user_id
        ).first()
        if not r:
            return False
        db.delete(r)
        db.commit()
        return True

    # =========================================================================
    # 5. USER TRAVEL PREFERENCES
    # =========================================================================

    def get_travel_preferences(self, user_id: int, db: Session) -> Dict[str, Any]:
        """Get or initialize travel preferences for user."""
        pref = db.query(UserTravelPreferences).filter(
            UserTravelPreferences.user_id == user_id
        ).first()
        if not pref:
            now_str = datetime.now(timezone.utc).isoformat()
            pref = UserTravelPreferences(
                user_id=user_id,
                default_mode="driving",
                avoid_tolls=False,
                avoid_highways=False,
                buffer_multiplier=1.0,
                updated_at=now_str
            )
            db.add(pref)
            db.commit()
            db.refresh(pref)

        return {
            "user_id": pref.user_id,
            "default_mode": pref.default_mode,
            "avoid_tolls": pref.avoid_tolls,
            "avoid_highways": pref.avoid_highways,
            "buffer_multiplier": pref.buffer_multiplier,
            "updated_at": pref.updated_at
        }

    def update_travel_preferences(
        self,
        user_id: int,
        updates: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Update user routing preferences."""
        pref = db.query(UserTravelPreferences).filter(
            UserTravelPreferences.user_id == user_id
        ).first()
        now_str = datetime.now(timezone.utc).isoformat()

        if not pref:
            pref = UserTravelPreferences(
                user_id=user_id,
                default_mode=updates.get("default_mode", "driving"),
                avoid_tolls=bool(updates.get("avoid_tolls", False)),
                avoid_highways=bool(updates.get("avoid_highways", False)),
                buffer_multiplier=float(updates.get("buffer_multiplier", 1.0)),
                updated_at=now_str
            )
            db.add(pref)
        else:
            if "default_mode" in updates:
                pref.default_mode = str(updates["default_mode"]).lower()
            if "avoid_tolls" in updates:
                pref.avoid_tolls = bool(updates["avoid_tolls"])
            if "avoid_highways" in updates:
                pref.avoid_highways = bool(updates["avoid_highways"])
            if "buffer_multiplier" in updates:
                pref.buffer_multiplier = max(0.5, min(3.0, float(updates["buffer_multiplier"])))
            pref.updated_at = now_str

        db.commit()
        db.refresh(pref)

        return {
            "user_id": pref.user_id,
            "default_mode": pref.default_mode,
            "avoid_tolls": pref.avoid_tolls,
            "avoid_highways": pref.avoid_highways,
            "buffer_multiplier": pref.buffer_multiplier,
            "updated_at": pref.updated_at
        }

    # =========================================================================
    # 6. HISTORICAL TRAVEL PRESERVATION & RELIABILITY ANALYTICS
    # =========================================================================

    def get_travel_reliability_metrics(self, user_id: int, db: Session) -> Dict[str, Any]:
        """Compute personal travel reliability metrics from historical departure logs."""
        logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id,
            DepartureLog.actual_travel_duration_min > 0,
            DepartureLog.estimated_travel_duration_min > 0
        ).order_by(DepartureLog.date.desc()).all()

        n = len(logs)
        if n == 0:
            return {
                "total_journeys": 0,
                "mean_estimated_min": 0.0,
                "mean_actual_min": 0.0,
                "mean_travel_bias_min": 0.0,
                "travel_bias_ratio": 1.0,
                "mae_min": 0.0,
                "on_time_travel_pct": 100.0,
                "mode_breakdown": {},
                "insights": ["Gathering journey data. Complete your upcoming departures to build travel intelligence."]
            }

        planned = [float(l.estimated_travel_duration_min) for l in logs]
        actual = [float(l.actual_travel_duration_min) for l in logs]
        errors = [act - pl for act, pl in zip(actual, planned)]

        mean_est = float(np.mean(planned))
        mean_act = float(np.mean(actual))
        mean_bias = float(np.mean(errors))
        bias_ratio = float(mean_act / max(0.01, mean_est))
        mae = float(np.mean(np.abs(errors)))
        on_time = sum(1 for e in errors if e <= 0.0)
        on_time_pct = round((on_time / n) * 100.0, 1)

        # Mode breakdown
        mode_counts: Dict[str, int] = {}
        for l in logs:
            m = l.travel_mode or "driving"
            mode_counts[m] = mode_counts.get(m, 0) + 1

        insights = []
        if mean_bias > 3.0:
            insights.append(f"Journeys typically take {mean_bias:.1f} minutes longer than planned.")
        elif mean_bias < -2.0:
            insights.append(f"Journeys consistently complete {abs(mean_bias):.1f} minutes faster than planned.")
        else:
            insights.append("Travel duration estimates are well-calibrated to actual travel times.")

        if on_time_pct < 65.0:
            insights.append(f"Only {on_time_pct}% of trips arrive within planned travel duration. Increasing safety buffer recommended.")
        elif on_time_pct >= 85.0:
            insights.append(f"High travel reliability: {on_time_pct}% of trips arrive within planned travel duration.")

        return {
            "total_journeys": n,
            "mean_estimated_min": round(mean_est, 1),
            "mean_actual_min": round(mean_act, 1),
            "mean_travel_bias_min": round(mean_bias, 1),
            "travel_bias_ratio": round(bias_ratio, 3),
            "mae_min": round(mae, 1),
            "on_time_travel_pct": on_time_pct,
            "mode_breakdown": mode_counts,
            "insights": insights
        }

    # =========================================================================
    # 7. CONTEXT INTEGRATION (Context-Aware Departure Planning)
    # =========================================================================

    def get_travel_context(
        self,
        user_id: int,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        departure_time: Optional[datetime] = None,
        event_id: Optional[str] = None,
        mode: TravelMode = TravelMode.DRIVING,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Compute estimated, live, and personalized travel duration with graceful degradation."""
        now = datetime.now(timezone.utc)
        dep_time = departure_time or now

        # If origin missing, resolve from primary saved location
        clean_origin = origin.strip() if origin else None
        if not clean_origin and db is not None:
            primary_loc = db.query(UserSavedLocation).filter(
                UserSavedLocation.user_id == user_id,
                UserSavedLocation.is_primary == True
            ).first()
            if primary_loc:
                clean_origin = primary_loc.address
        if not clean_origin:
            clean_origin = "Current Location"

        clean_dest = destination.strip() if destination else None

        # Check for missing destination
        if not clean_dest:
            logger.info("Destination missing for user_id=%s; requesting user input and falling back to routine estimate", user_id)
            fallback_duration = self._get_historical_or_routine_travel(user_id, event_id, db)
            meta = ExternalObservationMetadata(
                source="missing_destination_fallback",
                applicable_id=str(event_id or "unknown"),
                observation_type=ObservationType.ESTIMATED,
                confidence=0.5,
                is_fallback=True,
                details={"reason": "Destination not provided; user input recommended"}
            )
            return {
                "origin": clean_origin,
                "destination": None,
                "is_destination_required": True,
                "estimated_travel_duration": fallback_duration,
                "live_travel_duration": None,
                "personalized_travel_duration": fallback_duration,
                "traffic_conditions": None,
                "source": "missing_destination_fallback",
                "is_fallback": True,
                "degradation_reason": "destination_missing",
                "travel_metadata": meta,
                "traffic_metadata": None,
                "route_options": []
            }

        # 1. Attempt Live Route & Travel Estimation
        travel_integration: Optional[TravelIntegrationBase] = self.registry.get("travel")
        live_travel_est: Optional[TravelEstimate] = None
        travel_meta: Optional[ExternalObservationMetadata] = None

        if travel_integration and travel_integration.is_enabled():
            try:
                live_travel_est = travel_integration.estimate_travel(
                    origin=clean_origin,
                    destination=clean_dest,
                    departure_time=dep_time,
                    mode=mode,
                    applicable_id=str(event_id or "")
                )
                if live_travel_est.duration_min <= 0.0 or live_travel_est.duration_min > 720.0:
                    raise IntegrationDataError(
                        f"Unrealistic travel duration received: {live_travel_est.duration_min} min",
                        service=travel_integration.provider_name
                    )
                travel_meta = live_travel_est.metadata
            except Exception as e:
                logger.warning("Travel integration failed for user_id=%s: %s. Using historical/routine fallback.", user_id, e)
                live_travel_est = None

        # 2. Attempt Live Traffic Estimation
        traffic_integration: Optional[TrafficIntegrationBase] = self.registry.get("traffic")
        traffic_obs: Optional[TrafficObservation] = None
        traffic_meta: Optional[ExternalObservationMetadata] = None

        if traffic_integration and traffic_integration.is_enabled():
            try:
                traffic_obs = traffic_integration.get_traffic_conditions(
                    origin=clean_origin,
                    destination=clean_dest,
                    departure_time=dep_time,
                    applicable_id=str(event_id or "")
                )
                traffic_meta = traffic_obs.metadata
            except Exception as e:
                logger.warning("Traffic integration unavailable for user_id=%s: %s. Omitting traffic adjustments.", user_id, e)
                traffic_obs = None

        # 3. Resolve Durations and Apply Personalized Adjustment
        if live_travel_est is not None:
            base_duration = live_travel_est.duration_min
            if traffic_obs is not None:
                live_duration = base_duration * max(1.0, traffic_obs.delay_multiplier) + traffic_obs.delay_minutes
            else:
                live_duration = live_travel_est.duration_in_traffic_min or base_duration
                traffic_obs = TrafficObservation(
                    origin=clean_origin,
                    destination=clean_dest,
                    congestion_level=TrafficCongestionLevel.UNKNOWN,
                    delay_multiplier=1.0,
                    delay_minutes=0.0,
                    metadata=ExternalObservationMetadata(
                        source="traffic_unavailable_fallback",
                        observation_type=ObservationType.ESTIMATED,
                        is_fallback=True
                    )
                )

            # Apply personalized ML model
            personalized_duration = live_duration
            personalized_details = None
            if db is not None:
                p_est = PersonalizedTravelEngine.estimate(
                    user_id=user_id,
                    provider_duration_min=live_duration,
                    departure_time=dep_time,
                    travel_mode=mode.value if isinstance(mode, TravelMode) else str(mode),
                    db=db
                )
                personalized_duration = p_est.personalized_duration_min
                personalized_details = p_est.to_dict()

            return {
                "origin": clean_origin,
                "destination": clean_dest,
                "is_destination_required": False,
                "estimated_travel_duration": round(base_duration, 1),
                "live_travel_duration": round(live_duration, 1),
                "personalized_travel_duration": round(personalized_duration, 1),
                "personalized_details": personalized_details,
                "distance_km": live_travel_est.distance_km,
                "traffic_conditions": traffic_obs.to_dict(),
                "selected_route": live_travel_est.selected_route,
                "route_options": [opt.to_dict() for opt in live_travel_est.route_options],
                "source": f"live_travel:{travel_integration.provider_name}",
                "is_fallback": False,
                "travel_metadata": travel_meta,
                "traffic_metadata": traffic_meta
            }

        # 4. Fallback Path: Historical Personal Travel or Event Routine
        fallback_travel_min = self._get_historical_or_routine_travel(user_id, event_id, db)
        meta = ExternalObservationMetadata(
            source="historical_personal_travel",
            applicable_id=str(event_id or "unknown"),
            observation_type=ObservationType.HISTORICAL,
            confidence=0.8,
            is_fallback=True,
            details={"note": "External travel API unavailable; using personal historical travel duration"}
        )

        return {
            "origin": clean_origin,
            "destination": clean_dest,
            "is_destination_required": False,
            "estimated_travel_duration": fallback_travel_min,
            "live_travel_duration": fallback_travel_min,
            "personalized_travel_duration": fallback_travel_min,
            "personalized_details": None,
            "distance_km": None,
            "traffic_conditions": None,
            "selected_route": None,
            "route_options": [],
            "source": "historical_personal_travel",
            "is_fallback": True,
            "degradation_reason": "travel_api_unavailable",
            "travel_metadata": meta,
            "traffic_metadata": None
        }

    def _get_historical_or_routine_travel(
        self,
        user_id: int,
        event_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> float:
        """Fetch historical actual travel mean from `departure_logs` or routine template."""
        if db is not None:
            try:
                # 1. User event-specific historical mean
                if event_id and str(event_id).isdigit():
                    avg_travel = db.query(
                        func.avg(DepartureLog.actual_travel_duration_min)
                    ).filter(
                        DepartureLog.user_id == user_id,
                        DepartureLog.event_id == int(event_id),
                        DepartureLog.actual_travel_duration_min > 0
                    ).scalar()
                    if avg_travel is not None and avg_travel > 0:
                        return float(round(avg_travel, 1))

                    # 2. Check Event routine template
                    ev = db.query(Event).filter(Event.id == int(event_id)).first()
                    if ev and ev.estimated_travel_duration > 0:
                        return float(ev.estimated_travel_duration)

                # 3. Overall user historical mean travel
                user_avg = db.query(
                    func.avg(DepartureLog.actual_travel_duration_min)
                ).filter(
                    DepartureLog.user_id == user_id,
                    DepartureLog.actual_travel_duration_min > 0
                ).scalar()
                if user_avg is not None and user_avg > 0:
                    return float(round(user_avg, 1))

            except Exception as e:
                logger.warning("Failed to calculate historical travel for user_id=%s: %s", user_id, e)

        return 20.0  # Heuristic fallback
