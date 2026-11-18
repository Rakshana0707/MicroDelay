"""Event filtering and travel requirement classification engine for calendar events.

Differentiates physical events requiring morning departure from online/virtual meetings,
all-day events, personal blocks, and cancelled entries.
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("microdelay.calendar.filter")

# Virtual meeting URL patterns
VIRTUAL_URL_PATTERNS = [
    re.compile(r"https?://[^\s]*zoom\.us/[^\s]*", re.IGNORECASE),
    re.compile(r"https?://[^\s]*meet\.google\.com/[a-z0-9\-]+", re.IGNORECASE),
    re.compile(r"https?://[^\s]*teams\.microsoft\.com/[^\s]*", re.IGNORECASE),
    re.compile(r"https?://[^\s]*[a-z0-9\-]+\.webex\.com/[^\s]*", re.IGNORECASE),
    re.compile(r"https?://[^\s]*whereby\.com/[^\s]*", re.IGNORECASE),
    re.compile(r"https?://[^\s]*gotomeeting\.com/[^\s]*", re.IGNORECASE),
    re.compile(r"https?://[^\s]*chime\.aws/[^\s]*", re.IGNORECASE),
]

# Online meeting text indicators
VIRTUAL_KEYWORD_PATTERNS = [
    re.compile(r"\bzoom\b", re.IGNORECASE),
    re.compile(r"\bgoogle\s+meet\b", re.IGNORECASE),
    re.compile(r"\bteams\s+meeting\b", re.IGNORECASE),
    re.compile(r"\bwebinar\b", re.IGNORECASE),
    re.compile(r"\bvirtual\b", re.IGNORECASE),
    re.compile(r"\bremote\s+(sync|meeting|session)\b", re.IGNORECASE),
    re.compile(r"\bteleconference\b", re.IGNORECASE),
    re.compile(r"\bdial-in\b", re.IGNORECASE),
]

# Personal / non-travel event keywords
PERSONAL_KEYWORD_PATTERNS = [
    re.compile(r"\bfocus\s+time\b", re.IGNORECASE),
    re.compile(r"\b(ooo|out\s+of\s+office)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+book\b", re.IGNORECASE),
    re.compile(r"\b(hold|blocked|tentative)\b", re.IGNORECASE),
    re.compile(r"\blunch(\s+break)?\b", re.IGNORECASE),
    re.compile(r"\breminder\b", re.IGNORECASE),
    re.compile(r"\bheads-down\b", re.IGNORECASE),
]


@dataclass
class EventClassificationResult:
    """Outcome of event travel requirement classification."""
    travel_requirement: str  # "physical", "online", "personal", "excluded", "unknown"
    requires_departure_planning: bool
    missing_location_warning: bool
    online_meeting_url: Optional[str] = None
    sanitized_location: Optional[str] = None
    classification_reason: str = ""


class EventFilterService:
    """Analyzes and filters calendar events to detect genuine physical departures."""

    @classmethod
    def extract_virtual_meeting_url(cls, text: Optional[str]) -> Optional[str]:
        """Search text for video conference links."""
        if not text:
            return None
        for pattern in VIRTUAL_URL_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(0).strip(").,;")
        return None

    @classmethod
    def has_virtual_keywords(cls, text: Optional[str]) -> bool:
        """Check for online meeting keywords."""
        if not text:
            return False
        for pattern in VIRTUAL_KEYWORD_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @classmethod
    def has_personal_keywords(cls, text: Optional[str]) -> bool:
        """Check for personal / focus / block keywords."""
        if not text:
            return False
        for pattern in PERSONAL_KEYWORD_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @classmethod
    def sanitize_location(cls, location: Optional[str]) -> Optional[str]:
        """Clean location string, removing trailing URLs, conference pins, or phone numbers."""
        if not location:
            return None
        clean = location.strip()
        # Remove any meeting links in location field
        clean = re.sub(r"https?://\S+", "", clean).strip()
        # Remove dial-in numbers or pins
        clean = re.sub(r"(\+?\d[\d\s\-]{8,}\d)", "", clean).strip()
        # Clean stray delimiters
        clean = re.sub(r"^[\s,;|\-]+|[\s,;|\-]+$", "", clean).strip()
        return clean if clean else None

    @classmethod
    def classify_event(
        cls,
        title: str,
        location: Optional[str] = None,
        status: str = "confirmed",
        is_all_day: bool = False,
        description: Optional[str] = None
    ) -> EventClassificationResult:
        """Classify calendar event into travel requirements without storing private descriptions."""
        clean_title = (title or "").strip()
        clean_loc = cls.sanitize_location(location)
        combined_text = f"{clean_title} {location or ''} {description or ''}"

        # 1. Cancelled events
        if status.lower() in ["cancelled", "canceled"]:
            return EventClassificationResult(
                travel_requirement="excluded",
                requires_departure_planning=False,
                missing_location_warning=False,
                sanitized_location=clean_loc,
                classification_reason="Event is cancelled"
            )

        # 2. All-day events (e.g. Holidays, Birthdays, OOO days)
        if is_all_day:
            return EventClassificationResult(
                travel_requirement="personal",
                requires_departure_planning=False,
                missing_location_warning=False,
                sanitized_location=clean_loc,
                classification_reason="All-day event; does not require morning departure timing"
            )

        # 3. Online meeting detection
        meeting_url = cls.extract_virtual_meeting_url(combined_text)
        if meeting_url or cls.has_virtual_keywords(combined_text):
            return EventClassificationResult(
                travel_requirement="online",
                requires_departure_planning=False,
                missing_location_warning=False,
                online_meeting_url=meeting_url,
                sanitized_location=clean_loc,
                classification_reason="Online video conference or virtual meeting detected"
            )

        # 4. Personal non-travel blocks
        if cls.has_personal_keywords(clean_title) and not clean_loc:
            return EventClassificationResult(
                travel_requirement="personal",
                requires_departure_planning=False,
                missing_location_warning=False,
                sanitized_location=clean_loc,
                classification_reason="Personal block, focus time, or out-of-office entry"
            )

        # 5. Physical location present
        if clean_loc:
            return EventClassificationResult(
                travel_requirement="physical",
                requires_departure_planning=True,
                missing_location_warning=False,
                sanitized_location=clean_loc,
                classification_reason="Physical location detected requiring travel"
            )

        # 6. Timed event with missing location
        return EventClassificationResult(
            travel_requirement="unknown",
            requires_departure_planning=True,
            missing_location_warning=True,
            sanitized_location=None,
            classification_reason="Timed event has no physical or virtual location specified"
        )
