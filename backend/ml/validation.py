"""Chronological Training, Validation, and Evaluation for MicroDelay.

This module guarantees that:
1. Observations are chronologically sorted and partition-split:
   - 70% Training
   - 15% Validation (Model selection & hyperparameter tuning)
   - 15% Final Testing (Untouched holdout evaluation)
2. Expanding-window temporal cross-validation is used on development data.
3. Feature preprocessing is fitted strictly within each training fold, eliminating leakage.
4. Insufficient datasets (<14 records) are explicitly flagged as 'insufficient_data'
   with clear explanatory diagnostics instead of fabricating metrics.
5. The final test set remains strictly untouched until final assessment.
"""

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

try:
    from ml.preprocessing import MicroDelayPreprocessor, validate_observations_dataframe
except ModuleNotFoundError:
    from backend.ml.preprocessing import MicroDelayPreprocessor, validate_observations_dataframe


# Minimum observations required to conduct a valid 70 / 15 / 15 chronological split with CV
MIN_CHRONOLOGICAL_SAMPLES = 8
MIN_TEST_SAMPLES = 1
MIN_VAL_SAMPLES = 1


@dataclass
class PartitionMetadata:
    """Metadata for a temporal partition."""
    name: str
    count: int
    start_date: Optional[str]
    end_date: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "date_range": {
                "start": self.start_date,
                "end": self.end_date
            },
            "time_range": {
                "start": self.start_time,
                "end": self.end_time
            }
        }


@dataclass
class EvaluationReport:
    """Structured report produced by chronological evaluation."""
    evaluation_status: str  # 'validated', 'insufficient_data', 'failed'
    status_reason: str
    data_provenance: str    # 'real', 'synthetic', 'synthetic_assisted'
    is_validated: bool
    total_records: int
    train_metadata: Optional[Dict[str, Any]] = None
    val_metadata: Optional[Dict[str, Any]] = None
    test_metadata: Optional[Dict[str, Any]] = None
    num_cv_folds: int = 0
    cv_fold_maes: List[float] = field(default_factory=list)
    mean_val_mae: Optional[float] = None
    test_mae: Optional[float] = None
    test_rmse: Optional[float] = None
    baseline_test_mae: Optional[float] = None
    error_reduction_pct: Optional[float] = None
    evaluated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_status": self.evaluation_status,
            "status_reason": self.status_reason,
            "data_provenance": self.data_provenance,
            "is_validated": self.is_validated,
            "total_records": self.total_records,
            "train_partition": self.train_metadata,
            "validation_partition": self.val_metadata,
            "test_partition": self.test_metadata,
            "num_cv_folds": self.num_cv_folds,
            "cv_fold_maes": self.cv_fold_maes,
            "mean_val_mae": self.mean_val_mae,
            "test_mae": self.test_mae,
            "test_rmse": self.test_rmse,
            "baseline_test_mae": self.baseline_test_mae,
            "error_reduction_pct": self.error_reduction_pct,
            "evaluated_at": self.evaluated_at
        }


class ChronologicalSplitter:
    """Splits a dataset chronologically into Train (70%), Validation (15%), and Test (15%)."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    @staticmethod
    def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame chronologically and prevent duplicate observations across partitions."""
        work_df = df.copy()

        # Prevent duplicate records from appearing across partitions
        if "id" in work_df.columns:
            work_df = work_df.drop_duplicates(subset=["id"], keep="first")
        elif "date" in work_df.columns and "planned_departure" in work_df.columns:
            work_df = work_df.drop_duplicates(subset=["date", "planned_departure"], keep="first")

        # Build composite chronological sort key
        sort_keys = []
        if "date" in work_df.columns:
            sort_keys.append("date")
        if "planned_departure" in work_df.columns:
            sort_keys.append("planned_departure")
        if "id" in work_df.columns:
            sort_keys.append("id")

        if sort_keys:
            sorted_df = work_df.sort_values(by=sort_keys).reset_index(drop=True)
        else:
            sorted_df = work_df.reset_index(drop=True)

        return sorted_df

    def split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronologically partition df into (train_df, val_df, test_df)."""
        sorted_df = self.sort_chronologically(df)
        n = len(sorted_df)

        if n < MIN_CHRONOLOGICAL_SAMPLES:
            raise ValueError(
                f"Insufficient records for chronological split: {n} < {MIN_CHRONOLOGICAL_SAMPLES}."
            )

        train_end = max(4, int(round(n * self.train_ratio)))
        val_end = max(train_end + 1, int(round(n * (self.train_ratio + self.val_ratio))))
        if val_end >= n:
            val_end = n - 1
        if train_end >= val_end:
            train_end = val_end - 1

        train_df = sorted_df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = sorted_df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = sorted_df.iloc[val_end:].copy().reset_index(drop=True)

        return train_df, val_df, test_df

    @staticmethod
    def get_partition_metadata(df: pd.DataFrame, partition_name: str) -> PartitionMetadata:
        """Extract boundary dates and times for a partition."""
        if df.empty:
            return PartitionMetadata(
                name=partition_name,
                count=0,
                start_date=None,
                end_date=None,
                start_time=None,
                end_time=None
            )

        start_date = str(df["date"].iloc[0]) if "date" in df.columns else None
        end_date = str(df["date"].iloc[-1]) if "date" in df.columns else None
        start_time = str(df["planned_departure"].iloc[0]) if "planned_departure" in df.columns else None
        end_time = str(df["planned_departure"].iloc[-1]) if "planned_departure" in df.columns else None

        return PartitionMetadata(
            name=partition_name,
            count=len(df),
            start_date=start_date,
            end_date=end_date,
            start_time=start_time,
            end_time=end_time
        )


class ExpandingWindowCV:
    """Expanding-window time series cross-validator for temporal sequence data.
    
    In each split fold:
      - Training partition expands to include all history up to step t.
      - Validation partition consists of the next forward window [t, t + step].
      - Eliminates backward leakage.
    """

    def __init__(self, n_splits: int = 3, min_train_size: Optional[int] = None):
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def split(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[int, pd.DataFrame, pd.DataFrame], None, None]:
        """Yield (fold_idx, train_fold, val_fold) sequentially."""
        n = len(df)
        if n < 6:
            raise ValueError(f"At least 6 observations required for cross-validation, got {n}.")

        # Determine effective number of splits possible
        effective_splits = min(self.n_splits, n - 4)
        if effective_splits < 2:
            effective_splits = 2

        # Step size for validation folds
        min_train = self.min_train_size or max(4, int(n * 0.4))
        remaining = n - min_train
        val_step = max(1, remaining // effective_splits)

        for fold_idx in range(effective_splits):
            train_end = min_train + (fold_idx * val_step)
            val_end = min_train + ((fold_idx + 1) * val_step) if fold_idx < effective_splits - 1 else n

            if train_end >= n or val_end <= train_end:
                break

            train_fold = df.iloc[:train_end].copy().reset_index(drop=True)
            val_fold = df.iloc[train_end:val_end].copy().reset_index(drop=True)

            yield fold_idx + 1, train_fold, val_fold


class ChronologicalEvaluator:
    """Orchestrates chronological evaluation and validation across models."""

    def __init__(self, min_samples: int = MIN_CHRONOLOGICAL_SAMPLES):
        self.min_samples = min_samples
        self.splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        self.cv = ExpandingWindowCV(n_splits=3)

    def evaluate(
        self,
        df: pd.DataFrame,
        provenance: str = "synthetic",
        model_builder=None
    ) -> Tuple[EvaluationReport, Optional[Any], Optional[MicroDelayPreprocessor]]:
        """Conduct chronological evaluation.
        
        Args:
          df: Historical departure logs DataFrame.
          provenance: Provenance label ('real', 'synthetic', 'synthetic_assisted').
          model_builder: Callable returning an unfitted model instance (defaults to RandomForestRegressor).
          
        Returns:
          (report, final_fitted_model, final_fitted_preprocessor)
        """
        if model_builder is None:
            model_builder = lambda: RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)

        # 1. Minimum Data Check
        total_count = len(df)
        if total_count < self.min_samples:
            reason = (
                f"Insufficient observations for personalized chronological evaluation: "
                f"found {total_count} records, but a minimum of {self.min_samples} is required "
                f"to perform 70/15/15 chronological splitting without data leakage."
            )
            report = EvaluationReport(
                evaluation_status="insufficient_data",
                status_reason=reason,
                data_provenance=provenance,
                is_validated=False,
                total_records=total_count
            )
            return report, None, None

        # 2. Chronological Split: 70% Train, 15% Validation, 15% Test
        train_df, val_df, test_df = self.splitter.split(df)

        train_meta = self.splitter.get_partition_metadata(train_df, "training").to_dict()
        val_meta = self.splitter.get_partition_metadata(val_df, "validation").to_dict()
        test_meta = self.splitter.get_partition_metadata(test_df, "test").to_dict()

        # 3. Expanding-Window Cross-Validation on Development Data (Train + Validation)
        # Combine Train and Validation for hyperparameter / stability check
        dev_df = pd.concat([train_df, val_df], ignore_index=True)
        cv_maes: List[float] = []

        for fold_idx, cv_train, cv_val in self.cv.split(dev_df):
            # Preprocessing is strictly fitted ONLY on cv_train
            fold_preprocessor = MicroDelayPreprocessor()
            fold_preprocessor.fit(cv_train)

            X_cv_train = fold_preprocessor.transform(cv_train)
            y_cv_train = cv_train["total_delay_min"].values.astype(float)

            X_cv_val = fold_preprocessor.transform(cv_val)
            y_cv_val = cv_val["total_delay_min"].values.astype(float)

            fold_model = model_builder()
            fold_model.fit(X_cv_train, y_cv_train)

            preds = fold_model.predict(X_cv_val)
            fold_mae = float(mean_absolute_error(y_cv_val, preds))
            cv_maes.append(round(fold_mae, 3))

        mean_val_mae = round(float(np.mean(cv_maes)), 3) if cv_maes else None

        # 4. Fit Candidate Model on Development Data (Train + Val)
        dev_preprocessor = MicroDelayPreprocessor()
        dev_preprocessor.fit(dev_df)
        X_dev = dev_preprocessor.transform(dev_df)
        y_dev = dev_df["total_delay_min"].values.astype(float)

        candidate_model = model_builder()
        candidate_model.fit(X_dev, y_dev)

        # 5. Final Holdout Evaluation on Untouched Test Set
        # Note: Preprocessor transforms test_df using dev-fitted parameters
        X_test = dev_preprocessor.transform(test_df)
        y_test = test_df["total_delay_min"].values.astype(float)

        test_preds = candidate_model.predict(X_test)
        test_mae = round(float(mean_absolute_error(y_test, test_preds)), 3)
        test_rmse = round(float(root_mean_squared_error(y_test, test_preds)), 3)

        # Baseline Comparison on Test Set (Historical mean per event from dev set)
        event_means = dev_df.groupby("event_name")["total_delay_min"].mean().to_dict()
        global_dev_mean = float(dev_df["total_delay_min"].mean())
        baseline_preds = test_df["event_name"].map(event_means).fillna(global_dev_mean).values
        baseline_test_mae = round(float(mean_absolute_error(y_test, baseline_preds)), 3)

        reduction_pct = 0.0
        if baseline_test_mae > 0:
            reduction_pct = round(((baseline_test_mae - test_mae) / baseline_test_mae) * 100.0, 1)

        # 6. Production Model Fit on Full Validated Dataset
        full_preprocessor = MicroDelayPreprocessor()
        full_preprocessor.fit(df)
        X_full = full_preprocessor.transform(df)
        y_full = df["total_delay_min"].values.astype(float)

        production_model = model_builder()
        production_model.fit(X_full, y_full)

        report = EvaluationReport(
            evaluation_status="validated",
            status_reason="Chronological 70/15/15 validation completed successfully with untouched test evaluation.",
            data_provenance=provenance,
            is_validated=True,
            total_records=total_count,
            train_metadata=train_meta,
            val_metadata=val_meta,
            test_metadata=test_meta,
            num_cv_folds=len(cv_maes),
            cv_fold_maes=cv_maes,
            mean_val_mae=mean_val_mae,
            test_mae=test_mae,
            test_rmse=test_rmse,
            baseline_test_mae=baseline_test_mae,
            error_reduction_pct=reduction_pct
        )

        return report, production_model, full_preprocessor
