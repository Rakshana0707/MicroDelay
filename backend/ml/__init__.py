"""MicroDelay Machine Learning Package."""

from ml.preprocessing import (
    ValidationConfig,
    ValidationReport,
    validate_single_observation,
    validate_observations_dataframe,
    MicroDelayPreprocessor
)
from ml.features import (
    HistoricalFeatureExtractor,
    FEATURE_CATALOG,
    compare_feature_sets
)
from ml.validation import (
    ChronologicalSplitter,
    ExpandingWindowCV,
    ChronologicalEvaluator,
    EvaluationReport,
    PartitionMetadata
)
from ml.benchmark import (
    ModelBenchmarkSuite,
    BenchmarkSuiteResult,
    ModelBenchmarkResult,
    calculate_prediction_tolerance
)
from ml.reporting import (
    calculate_standardized_metrics,
    perform_residual_analysis,
    segment_performance,
    determine_model_status,
    ModelEvaluationStatus,
    EvaluationReporter,
    REPORTS_DIR,
    REAL_REPORTS_DIR,
    SYNTHETIC_REPORTS_DIR
)
from ml.attribution import (
    DelayAttributionEngine,
    DepartureAttributionResult,
    CategoryAttribution,
    AttributionType,
    AttributionConfidence,
    AttributionSource
)
from ml.registry import (
    ModelRegistry,
    ModelStatus,
    RetrainingPolicy,
    get_user_lock,
    VERSIONS_DIR
)
try:
    from ml.classifier import (
        LatenessClassificationConfig,
        LatenessClassifierSuite,
        LatenessClassifierManager,
        ClassificationEvaluationMetrics
    )
except ModuleNotFoundError:
    from backend.ml.classifier import (
        LatenessClassificationConfig,
        LatenessClassifierSuite,
        LatenessClassifierManager,
        ClassificationEvaluationMetrics
    )
try:
    from ml.adaptive import (
        PredictionContext,
        AdaptiveIntelligenceOutput,
        AdaptiveIntelligenceOrchestrator,
        ModelAvailability,
        RiskCategory,
        ConfidenceStatus
    )
except ModuleNotFoundError:
    from backend.ml.adaptive import (
        PredictionContext,
        AdaptiveIntelligenceOutput,
        AdaptiveIntelligenceOrchestrator,
        ModelAvailability,
        RiskCategory,
        ConfidenceStatus
    )

__all__ = [
    "ValidationConfig",
    "ValidationReport",
    "validate_single_observation",
    "validate_observations_dataframe",
    "MicroDelayPreprocessor",
    "HistoricalFeatureExtractor",
    "FEATURE_CATALOG",
    "compare_feature_sets",
    "ChronologicalSplitter",
    "ExpandingWindowCV",
    "ChronologicalEvaluator",
    "EvaluationReport",
    "PartitionMetadata",
    "ModelBenchmarkSuite",
    "BenchmarkSuiteResult",
    "ModelBenchmarkResult",
    "calculate_prediction_tolerance",
    "calculate_standardized_metrics",
    "perform_residual_analysis",
    "segment_performance",
    "determine_model_status",
    "ModelEvaluationStatus",
    "EvaluationReporter",
    "REPORTS_DIR",
    "REAL_REPORTS_DIR",
    "SYNTHETIC_REPORTS_DIR",
    "DelayAttributionEngine",
    "DepartureAttributionResult",
    "CategoryAttribution",
    "AttributionType",
    "AttributionConfidence",
    "AttributionSource",
    "ModelRegistry",
    "ModelStatus",
    "RetrainingPolicy",
    "get_user_lock",
    "VERSIONS_DIR",
    "PredictionContext",
    "AdaptiveIntelligenceOutput",
    "AdaptiveIntelligenceOrchestrator",
    "ModelAvailability",
    "RiskCategory",
    "ConfidenceStatus",
    "LatenessClassificationConfig",
    "LatenessClassifierSuite",
    "LatenessClassifierManager",
    "ClassificationEvaluationMetrics"
]
