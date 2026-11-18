"""Geocoding service and validation architecture (Prompt 3.4).

Provides coordinate and address validation, location search, persistent caching,
and provider integrations (Nominatim / Google / Mock fallback).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timezone, timedelta
import hashlib
import logging
import re

import httpx
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def validate_coordinates(latitude: Optional[float], longitude: Optional[float]) -> bool:
    """Validate latitude (-90 to 90) and longitude (-180 to 180)."""
    if latitude is None or longitude is None:
        return False
    try:
        lat = float(latitude)
        lng = float(longitude)
        if not (-90.0 <= lat <= 90.0):
            return False
        if not (-180.0 <= lng <= 180.0):
            return False
        return True
    except (TypeError, ValueError):
        return False


def validate_address(address: Optional[str]) -> bool:
    """Validate that an address is non-empty, reasonably formatted, and contains printable chars."""
    if not address or not isinstance(address, str):
        return False
    trimmed = address.strip()
    if len(trimmed) < 3 or len(trimmed) > 500:
        return False
    # Check for basic invalid character spam
    if re.match(r"^[^a-zA-Z0-9\s,.-]+$", trimmed):
        return False
    return True


class GeocodedLocation:
    """Structured result of a forward or reverse geocode lookup."""

    def __init__(
        self,
        formatted_address: str,
        latitude: float,
        longitude: float,
        provider: str = "geocoder",
        raw_metadata: Optional[Dict[str, Any]] = None,
        cached: bool = False
    ):
        self.formatted_address = formatted_address
        self.latitude = latitude
        self.longitude = longitude
        self.provider = provider
        self.raw_metadata = raw_metadata or {}
        self.cached = cached

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formatted_address": self.formatted_address,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "provider": self.provider,
            "cached": self.cached
        }


class BaseGeocodingProvider(ABC):
    """Abstract interface for geocoding providers."""

    @abstractmethod
    def geocode(self, address: str) -> Optional[GeocodedLocation]:
        """Convert address string to latitude/longitude coordinates."""
        pass

    @abstractmethod
    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[GeocodedLocation]:
        """Convert latitude/longitude coordinates to formatted address."""
        pass


class NominatimGeocodingProvider(BaseGeocodingProvider):
    """OpenStreetMap Nominatim Geocoding Provider (free, no API key required, with User-Agent)."""

    BASE_URL = "https://nominatim.openstreetmap.org"

    def __init__(self, user_agent: str = "MicroDelay-App/2.0", timeout: float = 5.0):
        self.user_agent = user_agent
        self.timeout = timeout

    def geocode(self, address: str) -> Optional[GeocodedLocation]:
        if not validate_address(address):
            return None
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(
                    f"{self.BASE_URL}/search",
                    params={"q": address, "format": "json", "limit": 1},
                    headers={"User-Agent": self.user_agent}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        top = data[0]
                        return GeocodedLocation(
                            formatted_address=top.get("display_name", address),
                            latitude=float(top["lat"]),
                            longitude=float(top["lon"]),
                            provider="nominatim",
                            raw_metadata=top
                        )
        except Exception as e:
            logger.warning(f"Nominatim geocoding error for '{address}': {e}")
        return None

    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[GeocodedLocation]:
        if not validate_coordinates(latitude, longitude):
            return None
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(
                    f"{self.BASE_URL}/reverse",
                    params={"lat": latitude, "lon": longitude, "format": "json"},
                    headers={"User-Agent": self.user_agent}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return GeocodedLocation(
                        formatted_address=data.get("display_name", f"{latitude:.4f}, {longitude:.4f}"),
                        latitude=latitude,
                        longitude=longitude,
                        provider="nominatim",
                        raw_metadata=data
                    )
        except Exception as e:
            logger.warning(f"Nominatim reverse geocode error for ({latitude}, {longitude}): {e}")
        return None


class GoogleGeocodingProvider(BaseGeocodingProvider):
    """Official Google Geocoding API provider."""

    BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    def __init__(self, api_key: str, timeout: float = 5.0):
        self.api_key = api_key
        self.timeout = timeout

    def geocode(self, address: str) -> Optional[GeocodedLocation]:
        if not validate_address(address) or not self.api_key:
            return None
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(
                    self.BASE_URL,
                    params={"address": address, "key": self.api_key}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "OK" and data.get("results"):
                        top = data["results"][0]
                        loc = top["geometry"]["location"]
                        return GeocodedLocation(
                            formatted_address=top.get("formatted_address", address),
                            latitude=float(loc["lat"]),
                            longitude=float(loc["lng"]),
                            provider="google",
                            raw_metadata=top
                        )
        except Exception as e:
            logger.warning(f"Google geocoding error for '{address}': {e}")
        return None

    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[GeocodedLocation]:
        if not validate_coordinates(latitude, longitude) or not self.api_key:
            return None
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(
                    self.BASE_URL,
                    params={"latlng": f"{latitude},{longitude}", "key": self.api_key}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "OK" and data.get("results"):
                        top = data["results"][0]
                        return GeocodedLocation(
                            formatted_address=top.get("formatted_address", f"{latitude:.4f}, {longitude:.4f}"),
                            latitude=latitude,
                            longitude=longitude,
                            provider="google",
                            raw_metadata=top
                        )
        except Exception as e:
            logger.warning(f"Google reverse geocode error for ({latitude}, {longitude}): {e}")
        return None


class MockGeocodingProvider(BaseGeocodingProvider):
    """Deterministic mock geocoding provider for testing and offline environments."""

    KNOWN_LOCATIONS = {
        "home": ("123 Elm Street, Springfield, IL 62701", 39.7817, -89.6501),
        "office": ("456 Tech Boulevard, Silicon Valley, CA 94025", 37.4848, -122.1484),
        "work": ("456 Tech Boulevard, Silicon Valley, CA 94025", 37.4848, -122.1484),
        "workplace": ("456 Tech Boulevard, Silicon Valley, CA 94025", 37.4848, -122.1484),
        "university": ("Stanford University, 450 Serra Mall, Stanford, CA 94305", 37.4275, -122.1697),
        "college": ("Stanford University, 450 Serra Mall, Stanford, CA 94305", 37.4275, -122.1697),
        "gym": ("789 Fitness Way, Downtown Metro, NY 10001", 40.7505, -73.9934),
        "airport": ("San Francisco International Airport (SFO), CA 94128", 37.6213, -122.3790),
    }

    def geocode(self, address: str) -> Optional[GeocodedLocation]:
        if not validate_address(address):
            return None
        norm = address.strip().lower()
        for key, (formatted, lat, lng) in self.KNOWN_LOCATIONS.items():
            if key in norm:
                return GeocodedLocation(
                    formatted_address=formatted,
                    latitude=lat,
                    longitude=lng,
                    provider="mock_geocoder"
                )
        # Deterministic coordinate generation based on string hash for testing
        h = int(hashlib.md5(norm.encode()).hexdigest()[:8], 16)
        lat = 30.0 + ((h % 2000) / 100.0)  # 30.0 to 50.0 N
        lng = -120.0 + (((h // 2000) % 4000) / 100.0)  # -120.0 to -80.0 W
        return GeocodedLocation(
            formatted_address=address.strip().title(),
            latitude=round(lat, 6),
            longitude=round(lng, 6),
            provider="mock_geocoder"
        )

    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[GeocodedLocation]:
        if not validate_coordinates(latitude, longitude):
            return None
        return GeocodedLocation(
            formatted_address=f"Location at ({latitude:.4f}, {longitude:.4f})",
            latitude=latitude,
            longitude=longitude,
            provider="mock_geocoder"
        )


class GeocodingManager:
    """Manages geocoding operations with database caching and multi-provider failover."""

    def __init__(
        self,
        primary_provider: Optional[BaseGeocodingProvider] = None,
        fallback_provider: Optional[BaseGeocodingProvider] = None,
        cache_ttl_days: int = 30
    ):
        self.primary_provider = primary_provider or NominatimGeocodingProvider()
        self.fallback_provider = fallback_provider or MockGeocodingProvider()
        self.cache_ttl_days = cache_ttl_days
        self._memory_cache: Dict[str, GeocodedLocation] = {}

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    def geocode(self, address: str, db: Optional[Session] = None) -> Optional[GeocodedLocation]:
        """Geocode an address with database and memory caching, fallback, and validation."""
        if not validate_address(address):
            return None

        q_hash = self._hash_query(address)

        # 1. Check in-memory cache
        if q_hash in self._memory_cache:
            loc = self._memory_cache[q_hash]
            loc.cached = True
            return loc

        # 2. Check DB cache if session provided
        if db is not None:
            try:
                from backend.database import GeocodeCacheRecord
                now_str = datetime.now(timezone.utc).isoformat()
                cached_rec = db.query(GeocodeCacheRecord).filter(
                    GeocodeCacheRecord.query_hash == q_hash
                ).first()
                if cached_rec:
                    # Check expiration if set
                    if not cached_rec.expires_at or cached_rec.expires_at > now_str:
                        loc = GeocodedLocation(
                            formatted_address=cached_rec.formatted_address,
                            latitude=cached_rec.latitude,
                            longitude=cached_rec.longitude,
                            provider=cached_rec.provider,
                            cached=True
                        )
                        self._memory_cache[q_hash] = loc
                        return loc
            except Exception as e:
                logger.warning(f"Geocode cache read error: {e}")

        # 3. Call primary provider
        result = None
        try:
            result = self.primary_provider.geocode(address)
        except Exception as e:
            logger.warning(f"Primary geocoding failed: {e}")

        # 4. Fallback if primary returned nothing
        if not result and self.fallback_provider:
            try:
                result = self.fallback_provider.geocode(address)
            except Exception as e:
                logger.warning(f"Fallback geocoding failed: {e}")

        # 5. Cache result if successful
        if result and validate_coordinates(result.latitude, result.longitude):
            self._memory_cache[q_hash] = result
            if db is not None:
                try:
                    from backend.database import GeocodeCacheRecord
                    now = datetime.now(timezone.utc)
                    expires = (now + timedelta(days=self.cache_ttl_days)).isoformat()
                    existing = db.query(GeocodeCacheRecord).filter(
                        GeocodeCacheRecord.query_hash == q_hash
                    ).first()
                    if not existing:
                        rec = GeocodeCacheRecord(
                            query_hash=q_hash,
                            query_text=address.strip(),
                            formatted_address=result.formatted_address,
                            latitude=result.latitude,
                            longitude=result.longitude,
                            provider=result.provider,
                            cached_at=now.isoformat(),
                            expires_at=expires
                        )
                        db.add(rec)
                        db.commit()
                except Exception as e:
                    db.rollback()
                    logger.warning(f"Geocode cache write error: {e}")

        return result

    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[GeocodedLocation]:
        """Convert coordinates to address."""
        if not validate_coordinates(latitude, longitude):
            return None

        result = None
        try:
            result = self.primary_provider.reverse_geocode(latitude, longitude)
        except Exception as e:
            logger.warning(f"Primary reverse geocoding failed: {e}")

        if not result and self.fallback_provider:
            try:
                result = self.fallback_provider.reverse_geocode(latitude, longitude)
            except Exception as e:
                logger.warning(f"Fallback reverse geocoding failed: {e}")

        return result
