"""Unified prediction context for MicroDelay Adaptive Intelligence.

Contains only information available at the time of the prediction request.
Guarantees strict absence of future lookahead information.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


VALID_DAYS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
}

VALID_IMPORTANCE = {
    "low", "medium", "high", "normal", "critical"
}


@dataclass
class ContextValidationResult:
    """Result of context validation checks."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    sanitized_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionContext:
    """Unified contextual state for departure delay and risk prediction."""
    user_id: int
    event_name: str
    day_of_week: str
    event_importance: str
    estimated_prep_duration: float
    estimated_travel_duration: float
    bag_prepared: bool
    event_id: Optional[int] = None
    prediction_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    historical_behavior_features: Dict[str, Any] = field(default_factory=dict)
    active_model_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ContextValidationResult:
        """Validate input values against behavioral sanity bounds."""
        errors: List[str] = []
        sanitized: Dict[str, Any] = {}

        if not isinstance(self.user_id, int) or self.user_id <= 0:
            errors.append(f"Invalid user_id '{self.user_id}': must be a positive integer.")

        clean_event = str(self.event_name).strip() if self.event_name else ""
        if not clean_event:
            errors.append("event_name must be a non-empty string.")
        sanitized["event_name"] = clean_event

        clean_day = str(self.day_of_week).strip().capitalize() if self.day_of_week else ""
        if clean_day not in VALID_DAYS:
            errors.append(f"Invalid day_of_week '{self.day_of_week}': must be one of {sorted(VALID_DAYS)}.")
        sanitized["day_of_week"] = clean_day

        clean_importance = str(self.event_importance).strip().lower() if self.event_importance else "medium"
        if clean_importance not in VALID_IMPORTANCE:
            clean_importance = "medium"
        sanitized["event_importance"] = clean_importance

        try:
            prep_dur = float(self.estimated_prep_duration)
            if prep_dur < 0.0 or prep_dur > 360.0:
                errors.append(f"estimated_prep_duration {prep_dur} min is outside sane bounds [0, 360].")
            sanitized["estimated_prep_duration"] = prep_dur
        except (ValueError, TypeError):
            errors.append(f"estimated_prep_duration '{self.estimated_prep_duration}' is not a valid number.")

        try:
            travel_dur = float(self.estimated_travel_duration)
            if travel_dur < 0.0 or travel_dur > 360.0:
                errors.append(f"estimated_travel_duration {travel_dur} min is outside sane bounds [0, 360].")
            sanitized["estimated_travel_duration"] = travel_dur
        except (ValueError, TypeError):
            errors.append(f"estimated_travel_duration '{self.estimated_travel_duration}' is not a valid number.")

        sanitized["bag_prepared"] = bool(self.bag_prepared)

        return ContextValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            sanitized_values=sanitized
        )

    def to_inference_dict(self) -> Dict[str, Any]:
        """Convert context to standard input feature dict expected by ML models and preprocessors."""
        return {
            "user_id": self.user_id,
            "event_name": self.event_name,
            "day_of_week": self.day_of_week,
            "importance": self.event_importance,
            "bag_prepared": 1 if self.bag_prepared else 0,
            "estimated_prep_duration_min": float(self.estimated_prep_duration),
            "estimated_travel_duration_min": float(self.estimated_travel_duration),
            **self.historical_behavior_features
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to a dictionary."""
        return {
            "user_id": self.user_id,
            "event_id": self.event_id,
            "event_name": self.event_name,
            "day_of_week": self.day_of_week,
            "event_importance": self.event_importance,
            "estimated_prep_duration": self.estimated_prep_duration,
            "estimated_travel_duration": self.estimated_travel_duration,
            "bag_prepared": self.bag_prepared,
            "prediction_timestamp": (
                self.prediction_timestamp.isoformat()
                if isinstance(self.prediction_timestamp, datetime)
                else str(self.prediction_timestamp)
            ),
            "historical_behavior_features": self.historical_behavior_features,
            "active_model_version": self.active_model_version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionContext":
        """Construct a PredictionContext from a dictionary safely."""
        ts = data.get("prediction_timestamp")
        if isinstance(ts, str):
            try:
                parsed_ts = datetime.fromisoformat(ts)
            except Exception:
                parsed_ts = datetime.now(timezone.utc)
        elif isinstance(ts, datetime):
            parsed_ts = ts
        else:
            parsed_ts = datetime.now(timezone.utc)

        return cls(
            user_id=int(data.get("user_id", 0)),
            event_id=data.get("event_id"),
            event_name=str(data.get("event_name", "General Departure")),
            day_of_week=str(data.get("day_of_week", "Monday")),
            event_importance=str(data.get("event_importance", data.get("importance", "medium"))),
            estimated_prep_duration=float(data.get("estimated_prep_duration", data.get("estimated_prep", 30.0))),
            estimated_travel_duration=float(data.get("estimated_travel_duration", data.get("estimated_travel", 20.0))),
            bag_prepared=bool(data.get("bag_prepared", False)),
            prediction_timestamp=parsed_ts,
            historical_behavior_features=data.get("historical_behavior_features", {}),
            active_model_version=data.get("active_model_version"),
            metadata=data.get("metadata", {})
        )
