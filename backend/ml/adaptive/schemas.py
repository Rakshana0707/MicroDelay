"""Output schemas, enums, and data contracts for MicroDelay Adaptive Intelligence."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


class ModelAvailability(str, Enum):
    """Explicit model availability states."""
    AVAILABLE = "available"
    TRAINING = "training"
    INSUFFICIENT_DATA = "insufficient_data"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class RiskCategory(str, Enum):
    """Categorized departure lateness risk levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class ConfidenceLevel(str, Enum):
    """Confidence levels for behavioral predictions."""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class ReliabilityStatus(str, Enum):
    """Documented prediction reliability states (Prompt 2.9)."""
    INSUFFICIENT_HISTORY = "insufficient_history"
    LOW_RELIABILITY = "low_reliability"
    MODERATE_RELIABILITY = "moderate_reliability"
    HIGH_RELIABILITY = "high_reliability"
    DEGRADED_MODEL = "degraded_model"


class CalibrationStatus(str, Enum):
    """Classification probability calibration status."""
    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"
    INSUFFICIENT_SAMPLES = "insufficient_samples"


class DataSufficiencyLevel(str, Enum):
    """Documented data sufficiency levels."""
    INSUFFICIENT = "insufficient"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    ROBUST = "robust"


@dataclass
class ContributingFactor:
    """An individual behavioral or situational factor contributing to delay."""
    factor: str
    impact_minutes: float
    direction: str  # "increases_delay", "decreases_delay", "neutral"
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "impact_minutes": round(self.impact_minutes, 1),
            "direction": self.direction,
            "description": self.description
        }


@dataclass
class RecommendedIntervention:
    """An adaptive behavioral intervention proposed to reduce departure delay."""
    action: str
    description: str
    expected_delay_reduction_min: float
    category: str  # e.g., "preparation", "timing", "distraction"
    priority: str = "medium"  # "high", "medium", "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "description": self.description,
            "expected_delay_reduction_min": round(self.expected_delay_reduction_min, 1),
            "category": self.category,
            "priority": self.priority
        }


@dataclass
class ConfidenceStatus:
    """Detailed reliability and confidence evaluation of the prediction."""
    score: float  # 0.0 to 1.0
    level: str  # "high", "moderate", "low"
    is_reliable: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "level": self.level,
            "is_reliable": self.is_reliable,
            "reasons": self.reasons
        }


@dataclass
class DataQualityStatus:
    """Status of the underlying data backing the intelligence models."""
    provenance: str  # "real", "synthetic", "none", "mixed"
    sample_count: int
    is_sufficient: bool
    has_real_data: bool
    quality_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance": self.provenance,
            "sample_count": self.sample_count,
            "is_sufficient": self.is_sufficient,
            "has_real_data": self.has_real_data,
            "quality_notes": self.quality_notes
        }


@dataclass
class AdaptiveIntelligenceOutput:
    """Standardized output contract for the MicroDelay Adaptive Behavioral Intelligence Engine."""
    predicted_delay_minutes: Optional[float] = None
    baseline_predicted_delay_minutes: float = 4.5
    late_probability: Optional[float] = None
    risk_category: Optional[str] = None
    contributing_factors: List[Dict[str, Any]] = field(default_factory=list)
    recommended_interventions: List[Dict[str, Any]] = field(default_factory=list)
    confidence_status: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "default_fallback"
    model_type: str = "statistical_baseline"
    model_availability: str = ModelAvailability.AVAILABLE.value
    data_quality_status: Dict[str, Any] = field(default_factory=dict)
    personalization_info: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[Dict[str, Any]] = None
    fallback_applied: bool = False
    component_status: Dict[str, str] = field(default_factory=dict)
    # Prediction Confidence & Reliability extensions (Prompt 2.9)
    prediction_interval_lower: Optional[float] = None
    prediction_interval_upper: Optional[float] = None
    coverage_level: Optional[float] = 0.80
    reliability_status: str = "moderate_reliability"
    calibration_status: str = "uncalibrated"
    data_sufficiency: str = "minimal"
    limitations: List[str] = field(default_factory=list)
    is_abstained: bool = False
    abstention_reason: Optional[str] = None
    reliability_assessment: Optional[Dict[str, Any]] = None
    execution_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize output to a JSON-safe dictionary."""
        return {
            "predicted_delay_minutes": round(self.predicted_delay_minutes, 1) if self.predicted_delay_minutes is not None else None,
            "baseline_predicted_delay_minutes": round(self.baseline_predicted_delay_minutes, 1),
            "prediction_interval_lower": round(self.prediction_interval_lower, 1) if self.prediction_interval_lower is not None else None,
            "prediction_interval_upper": round(self.prediction_interval_upper, 1) if self.prediction_interval_upper is not None else None,
            "coverage_level": round(self.coverage_level, 2) if self.coverage_level is not None else None,
            "late_probability": round(self.late_probability, 2) if self.late_probability is not None else None,
            "risk_category": self.risk_category,
            "reliability_status": self.reliability_status,
            "calibration_status": self.calibration_status,
            "data_sufficiency": self.data_sufficiency,
            "limitations": self.limitations,
            "is_abstained": self.is_abstained,
            "abstention_reason": self.abstention_reason,
            "reliability_assessment": self.reliability_assessment,
            "contributing_factors": self.contributing_factors,
            "recommended_interventions": self.recommended_interventions,
            "confidence_status": self.confidence_status,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "model_availability": self.model_availability,
            "data_quality_status": self.data_quality_status,
            "personalization_info": self.personalization_info,
            "explanation": self.explanation,
            "fallback_applied": self.fallback_applied,
            "component_status": self.component_status,
            "execution_timestamp": self.execution_timestamp,
            "warnings": self.warnings
        }

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Provide 100% backward compatibility for endpoints expecting Phase 1 predict output."""
        return {
            "predicted_delay_min": round(self.predicted_delay_minutes, 1) if self.predicted_delay_minutes is not None else None,
            "baseline_predicted_delay_min": round(self.baseline_predicted_delay_minutes, 1),
            "prediction_interval_lower": round(self.prediction_interval_lower, 1) if self.prediction_interval_lower is not None else None,
            "prediction_interval_upper": round(self.prediction_interval_upper, 1) if self.prediction_interval_upper is not None else None,
            "coverage_level": round(self.coverage_level, 2) if self.coverage_level is not None else None,
            "model_used": self.model_type,
            "model_version": self.model_version,
            "data_provenance": self.data_quality_status.get("provenance", "unknown"),
            "training_mode": self.data_quality_status.get("provenance", "unknown"),
            "is_personalized_real": self.data_quality_status.get("provenance") == "real",
            # Extended fields cleanly attached
            "late_probability": round(self.late_probability, 2) if self.late_probability is not None else None,
            "risk_category": self.risk_category,
            "reliability_status": self.reliability_status,
            "calibration_status": self.calibration_status,
            "data_sufficiency": self.data_sufficiency,
            "limitations": self.limitations,
            "is_abstained": self.is_abstained,
            "abstention_reason": self.abstention_reason,
            "reliability_assessment": self.reliability_assessment,
            "model_availability": self.model_availability,
            "confidence_status": self.confidence_status,
            "personalization_info": self.personalization_info,
            "explanation": self.explanation
        }
