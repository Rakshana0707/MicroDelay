"""Centralized Data Validation and Preprocessing Pipeline for MicroDelay ML.

This module enforces data integrity and prepares features for regression models.
It validates input records, categorizes records (valid, potential_outlier, invalid),
handles missing values, encodes categories, and encapsulates all transformations
within scikit-learn ColumnTransformer / Pipeline architectures.
"""

import logging
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger("microdelay.ml.preprocessing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [ML.Preprocessing] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

VALID_DAYS_OF_WEEK = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
}
VALID_IMPORTANCE_LEVELS = {"low", "medium", "high"}
VALID_PROVENANCE_VALUES = {"synthetic", "real", "unknown"}

NUMERICAL_FEATURE_COLS = [
    "estimated_prep_duration_min",
    "estimated_travel_duration_min",
    "bag_prepared"
]
CATEGORICAL_FEATURE_COLS = [
    "event_name",
    "day_of_week",
    "importance"
]
ALL_INPUT_FEATURE_COLS = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS


@dataclass
class ValidationConfig:
    """Configurable thresholds for data validation and outlier classification."""
    min_prep_duration_min: float = 0.0
    max_valid_prep_duration_min: float = 720.0       # 12 hours max physically possible
    outlier_prep_threshold_min: float = 120.0        # 2+ hours flagged as potential outlier

    min_travel_duration_min: float = 0.0
    max_valid_travel_duration_min: float = 720.0     # 12 hours max physically possible
    outlier_travel_threshold_min: float = 180.0      # 3+ hours travel flagged as outlier

    min_valid_delay_min: float = -180.0              # max 3 hours early departure
    max_valid_delay_min: float = 720.0               # max 12 hours late departure
    outlier_delay_threshold_min: float = 60.0        # 1+ hour late flagged as potential outlier

    allow_imputation: bool = True
    strict_provenance: bool = True


@dataclass
class RecordValidationResult:
    is_valid: bool
    status: str  # 'valid', 'potential_outlier', 'invalid'
    issues: List[str] = field(default_factory=list)
    cleaned_record: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    total_records: int = 0
    valid_records: int = 0
    potential_outliers: int = 0
    invalid_records: int = 0
    duplicate_records: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    flagged_reasons: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "potential_outliers": self.potential_outliers,
            "invalid_records": self.invalid_records,
            "duplicate_records": self.duplicate_records,
            "acceptance_rate_pct": round(
                ((self.valid_records + self.potential_outliers) / self.total_records * 100), 1
            ) if self.total_records > 0 else 0.0,
            "rejection_reasons": self.rejection_reasons,
            "flagged_reasons": self.flagged_reasons
        }


def _parse_time_str(time_val: Any) -> Optional[datetime.time]:
    """Attempt to parse various time string formats into datetime.time."""
    if not time_val or pd.isna(time_val):
        return None
    time_str = str(time_val).strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
        try:
            return datetime.datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue
    return None


def validate_single_observation(
    record: Dict[str, Any],
    config: Optional[ValidationConfig] = None
) -> RecordValidationResult:
    """Validate a single historical or inference observation.
    
    Distinguishes:
      - 'invalid': critical errors violating physical reality, corrupt types, unparseable required values.
      - 'potential_outlier': extreme yet physically possible observations (e.g., massive prep delay).
      - 'valid': normal valid record.
    """
    if config is None:
        config = ValidationConfig()

    issues: List[str] = []
    flags: List[str] = []
    cleaned: Dict[str, Any] = dict(record)

    # 1. user_id validation (optional for inference row if missing, but if present must be int > 0)
    user_id = record.get("user_id")
    if user_id is not None and not pd.isna(user_id):
        try:
            cleaned["user_id"] = int(user_id)
            if cleaned["user_id"] <= 0:
                issues.append(f"Invalid user_id: {user_id} must be positive")
        except (ValueError, TypeError):
            issues.append(f"Invalid user_id format: {user_id}")

    # 2. event_name validation
    event_name = record.get("event_name")
    if not event_name or pd.isna(event_name) or str(event_name).strip() == "":
        issues.append("Missing required field: event_name")
    else:
        cleaned["event_name"] = str(event_name).strip()

    # 3. day_of_week validation
    day = record.get("day_of_week")
    if not day or pd.isna(day):
        issues.append("Missing required field: day_of_week")
    else:
        day_str = str(day).strip().capitalize()
        if day_str not in VALID_DAYS_OF_WEEK:
            issues.append(f"Invalid day_of_week '{day}'. Must be one of {sorted(VALID_DAYS_OF_WEEK)}")
        else:
            cleaned["day_of_week"] = day_str

    # 4. importance validation
    importance = record.get("importance")
    if importance is None or pd.isna(importance):
        cleaned["importance"] = "medium"  # Safe default fallback
    else:
        imp_str = str(importance).strip().lower()
        if imp_str not in VALID_IMPORTANCE_LEVELS:
            cleaned["importance"] = "medium"
            flags.append(f"Unknown importance '{importance}', normalized to 'medium'")
        else:
            cleaned["importance"] = imp_str

    # 5. estimated_prep_duration_min validation
    est_prep = record.get("estimated_prep_duration_min")
    if est_prep is None or pd.isna(est_prep):
        if config.allow_imputation:
            cleaned["estimated_prep_duration_min"] = 20.0
            flags.append("Missing estimated_prep_duration_min; defaulted to 20.0")
        else:
            issues.append("Missing required field: estimated_prep_duration_min")
    else:
        try:
            val = float(est_prep)
            if np.isnan(val) or np.isinf(val):
                issues.append("Invalid numerical value: estimated_prep_duration_min is NaN or Inf")
            elif val < config.min_prep_duration_min:
                issues.append(f"Negative estimated_prep_duration_min: {val} < 0")
            elif val > config.max_valid_prep_duration_min:
                issues.append(f"Physically implausible estimated_prep_duration_min: {val} > {config.max_valid_prep_duration_min}")
            else:
                if val >= config.outlier_prep_threshold_min:
                    flags.append(f"Extreme estimated_prep_duration_min: {val} >= {config.outlier_prep_threshold_min}")
                cleaned["estimated_prep_duration_min"] = val
        except (ValueError, TypeError):
            issues.append(f"Non-numeric estimated_prep_duration_min: {est_prep}")

    # 6. actual_prep_duration_min validation (if present, e.g. in training data)
    if "actual_prep_duration_min" in record and not pd.isna(record["actual_prep_duration_min"]):
        try:
            act_prep = float(record["actual_prep_duration_min"])
            if np.isnan(act_prep) or np.isinf(act_prep):
                issues.append("Invalid actual_prep_duration_min: NaN or Inf")
            elif act_prep < config.min_prep_duration_min:
                issues.append(f"Negative actual_prep_duration_min: {act_prep} < 0")
            elif act_prep > config.max_valid_prep_duration_min:
                issues.append(f"Physically implausible actual_prep_duration_min: {act_prep} > {config.max_valid_prep_duration_min}")
            else:
                if act_prep >= config.outlier_prep_threshold_min:
                    flags.append(f"Extreme actual_prep_duration_min: {act_prep}")
                cleaned["actual_prep_duration_min"] = act_prep
        except (ValueError, TypeError):
            issues.append(f"Non-numeric actual_prep_duration_min: {record['actual_prep_duration_min']}")

    # 7. estimated_travel_duration_min validation
    est_travel = record.get("estimated_travel_duration_min")
    if est_travel is None or pd.isna(est_travel):
        if config.allow_imputation:
            cleaned["estimated_travel_duration_min"] = 20.0
            flags.append("Missing estimated_travel_duration_min; defaulted to 20.0")
        else:
            issues.append("Missing required field: estimated_travel_duration_min")
    else:
        try:
            val = float(est_travel)
            if np.isnan(val) or np.isinf(val):
                issues.append("Invalid numerical value: estimated_travel_duration_min is NaN or Inf")
            elif val < config.min_travel_duration_min:
                issues.append(f"Negative estimated_travel_duration_min: {val} < 0")
            elif val > config.max_valid_travel_duration_min:
                issues.append(f"Physically implausible estimated_travel_duration_min: {val} > {config.max_valid_travel_duration_min}")
            else:
                if val >= config.outlier_travel_threshold_min:
                    flags.append(f"Extreme estimated_travel_duration_min: {val}")
                cleaned["estimated_travel_duration_min"] = val
        except (ValueError, TypeError):
            issues.append(f"Non-numeric estimated_travel_duration_min: {est_travel}")

    # 8. bag_prepared validation
    bag = record.get("bag_prepared")
    if bag is None or pd.isna(bag):
        cleaned["bag_prepared"] = 0
    else:
        cleaned["bag_prepared"] = 1 if bool(bag) else 0

    # 9. total_delay_min target validation (if present in dataset)
    if "total_delay_min" in record and not pd.isna(record["total_delay_min"]):
        try:
            delay = float(record["total_delay_min"])
            if np.isnan(delay) or np.isinf(delay):
                issues.append("Invalid total_delay_min: NaN or Inf")
            elif delay < config.min_valid_delay_min or delay > config.max_valid_delay_min:
                issues.append(f"Physically implausible total_delay_min: {delay} (valid range [{config.min_valid_delay_min}, {config.max_valid_delay_min}])")
            else:
                if abs(delay) >= config.outlier_delay_threshold_min:
                    flags.append(f"Substantial delay observation: {delay}m")
                cleaned["total_delay_min"] = delay
        except (ValueError, TypeError):
            issues.append(f"Non-numeric total_delay_min: {record['total_delay_min']}")

    # 10. planned_departure & actual_departure timestamps (if provided)
    if "planned_departure" in record and record["planned_departure"]:
        t_plan = _parse_time_str(record["planned_departure"])
        if t_plan is None:
            issues.append(f"Unparseable planned_departure time: {record['planned_departure']}")
    if "actual_departure" in record and record["actual_departure"]:
        t_act = _parse_time_str(record["actual_departure"])
        if t_act is None:
            issues.append(f"Unparseable actual_departure time: {record['actual_departure']}")

    # 11. Data provenance validation
    prov = record.get("provenance")
    if prov is not None and not pd.isna(prov):
        prov_str = str(prov).strip().lower()
        if prov_str not in VALID_PROVENANCE_VALUES:
            if config.strict_provenance:
                flags.append(f"Non-standard provenance value: '{prov}', normalized to 'unknown'")
            cleaned["provenance"] = "unknown"
        else:
            cleaned["provenance"] = prov_str
    else:
        cleaned["provenance"] = "unknown"

    # Determine final record status
    if issues:
        status = "invalid"
        is_valid = False
    elif flags:
        status = "potential_outlier"
        is_valid = True
    else:
        status = "valid"
        is_valid = True

    all_notes = issues + flags
    return RecordValidationResult(
        is_valid=is_valid,
        status=status,
        issues=all_notes,
        cleaned_record=cleaned if is_valid else None
    )


def validate_observations_dataframe(
    df: pd.DataFrame,
    config: Optional[ValidationConfig] = None,
    drop_duplicates: bool = True,
    drop_invalid: bool = True
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Validate, clean, and deduplicate a full DataFrame of observations.
    
    Maintains structured validation statistics. Retains valid unusual observations
    (potential outliers) so true behavioral lateness patterns are not lost.
    """
    if config is None:
        config = ValidationConfig()

    report = ValidationReport(total_records=len(df))
    if df.empty:
        return pd.DataFrame(), report

    # 1. Deduplication on key columns if present
    work_df = df.copy()
    initial_count = len(work_df)
    dedup_cols = [c for c in ["user_id", "date", "planned_departure", "event_name"] if c in work_df.columns]
    if drop_duplicates and len(dedup_cols) >= 2:
        work_df = work_df.drop_duplicates(subset=dedup_cols, keep="last")
        report.duplicate_records = initial_count - len(work_df)
        if report.duplicate_records > 0:
            logger.info(f"Deduplicated {report.duplicate_records} redundant observations")

    valid_rows: List[Dict[str, Any]] = []

    # 2. Row-by-row validation
    for idx, row in work_df.iterrows():
        record_dict = row.to_dict()
        res = validate_single_observation(record_dict, config=config)

        if not res.is_valid:
            report.invalid_records += 1
            for issue in res.issues:
                report.rejection_reasons[issue] = report.rejection_reasons.get(issue, 0) + 1
            if not drop_invalid:
                valid_rows.append(record_dict)
        else:
            if res.status == "potential_outlier":
                report.potential_outliers += 1
                for flag in res.issues:
                    report.flagged_reasons[flag] = report.flagged_reasons.get(flag, 0) + 1
            else:
                report.valid_records += 1

            if res.cleaned_record is not None:
                valid_rows.append(res.cleaned_record)

    cleaned_df = pd.DataFrame(valid_rows)
    logger.info(
        f"Validation complete: {report.valid_records} valid, "
        f"{report.potential_outliers} flagged outliers, "
        f"{report.invalid_records} rejected, {report.duplicate_records} duplicates."
    )
    return cleaned_df, report


def create_feature_preprocessor() -> ColumnTransformer:
    """Create an un-fitted scikit-learn ColumnTransformer for feature encoding.
    
    Transforms:
      - Numerical columns (estimated_prep, estimated_travel, bag_prepared):
        Median imputation.
      - Categorical columns (event_name, day_of_week, importance):
        Missing string imputation + OneHotEncoder with handle_unknown='ignore'.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURE_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURE_COLS)
        ],
        remainder="drop"
    )
    return preprocessor


class MicroDelayPreprocessor:
    """High-level preprocessing wrapper managing validation, fitting, and transformation."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.pipeline: ColumnTransformer = create_feature_preprocessor()
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "MicroDelayPreprocessor":
        """Fit preprocessing transformations exclusively on training partition data."""
        # Validate training data
        clean_df, report = validate_observations_dataframe(
            df, config=self.config, drop_invalid=True
        )
        if clean_df.empty:
            raise ValueError("Cannot fit preprocessor: no valid records remain after validation.")

        X = clean_df[ALL_INPUT_FEATURE_COLS]
        self.pipeline.fit(X)
        self.is_fitted = True

        # Extract generated feature names
        try:
            self.feature_names_ = list(self.pipeline.get_feature_names_out())
        except Exception:
            # Fallback if get_feature_names_out is unavailable
            self.feature_names_ = [f"feat_{i}" for i in range(self.pipeline.transform(X[:1]).shape[1])]

        logger.info(f"Preprocessor successfully fitted on {len(clean_df)} training samples. Output features: {len(self.feature_names_)}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform observations using the fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        # Ensure all expected columns exist (fill missing with NaN for imputer)
        X = df.copy()
        for col in ALL_INPUT_FEATURE_COLS:
            if col not in X.columns:
                X[col] = np.nan

        X_subset = X[ALL_INPUT_FEATURE_COLS]
        return self.pipeline.transform(X_subset)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Fit on training data and return (X_transformed, y, cleaned_df)."""
        clean_df, report = validate_observations_dataframe(
            df, config=self.config, drop_invalid=True
        )
        if clean_df.empty or "total_delay_min" not in clean_df.columns:
            raise ValueError("Training dataset must contain valid records with 'total_delay_min' target.")

        self.fit(clean_df)
        X_trans = self.transform(clean_df)
        y = clean_df["total_delay_min"].values.astype(float)
        return X_trans, y, clean_df

    def transform_single_inference(self, row_dict: Dict[str, Any]) -> np.ndarray:
        """Validate and transform a single inference query."""
        res = validate_single_observation(row_dict, config=self.config)
        cleaned = res.cleaned_record or row_dict
        single_df = pd.DataFrame([cleaned])
        return self.transform(single_df)
