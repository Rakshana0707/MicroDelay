"""Multi-Algorithm Regression Benchmarking for MicroDelay.

This module benchmarks multiple regression algorithms across identical chronological
splits to determine whether alternative models outperform the existing Random Forest:
1. Historical Statistical Baseline (Event-grouped mean)
2. Linear Regression
3. Random Forest Regressor (Current production baseline)
4. HistGradientBoosting Regressor
5. XGBoost Regressor (Optional dependency fallback)

All models share:
- Identical chronological partitions (70% train, 15% val, 15% test).
- Identical target: total_delay_min.
- Leakage-safe fold preprocessing (fitted exclusively on training partition).
- Evaluation on untouched holdout test set for the selected winner only.
"""

import time
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from ml.preprocessing import MicroDelayPreprocessor, validate_observations_dataframe
    from ml.validation import (
        ChronologicalSplitter,
        ExpandingWindowCV,
        EvaluationReport,
        MIN_CHRONOLOGICAL_SAMPLES,
    )
except ModuleNotFoundError:
    from backend.ml.preprocessing import MicroDelayPreprocessor, validate_observations_dataframe
    from backend.ml.validation import (
        ChronologicalSplitter,
        ExpandingWindowCV,
        EvaluationReport,
        MIN_CHRONOLOGICAL_SAMPLES,
    )


def calculate_prediction_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tolerance_min: float = 5.0) -> float:
    """Calculate percentage of predictions within +/- tolerance_min minutes of actual delay."""
    if len(y_true) == 0:
        return 0.0
    abs_errors = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    within_tolerance = (abs_errors <= tolerance_min).sum()
    return round(float(within_tolerance / len(y_true) * 100.0), 2)


@dataclass
class ModelBenchmarkResult:
    """Benchmark metrics and diagnostic info for a single algorithm."""
    model_name: str
    status: str  # 'evaluated', 'skipped', 'failed', 'dependency_unavailable'
    hyperparameters: Dict[str, Any]
    training_duration_ms: float
    val_mae: Optional[float] = None
    val_rmse: Optional[float] = None
    val_r2: Optional[float] = None
    val_medae: Optional[float] = None
    val_tolerance_5m_pct: Optional[float] = None
    test_mae: Optional[float] = None
    test_rmse: Optional[float] = None
    test_r2: Optional[float] = None
    test_medae: Optional[float] = None
    test_tolerance_5m_pct: Optional[float] = None
    cv_fold_maes: List[float] = field(default_factory=list)
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "hyperparameters": self.hyperparameters,
            "training_duration_ms": round(self.training_duration_ms, 2),
            "validation_metrics": {
                "mae": self.val_mae,
                "rmse": self.val_rmse,
                "r2": self.val_r2,
                "medae": self.val_medae,
                "tolerance_5m_pct": self.val_tolerance_5m_pct,
                "cv_fold_maes": self.cv_fold_maes
            } if self.status == "evaluated" else None,
            "test_metrics": {
                "mae": self.test_mae,
                "rmse": self.test_rmse,
                "r2": self.test_r2,
                "medae": self.test_medae,
                "tolerance_5m_pct": self.test_tolerance_5m_pct
            } if self.test_mae is not None else None,
            "failure_reason": self.failure_reason
        }


@dataclass
class BenchmarkSuiteResult:
    """Aggregated outcome of multi-algorithm benchmark suite."""
    benchmark_status: str  # 'completed', 'insufficient_data', 'failed'
    status_reason: str
    data_provenance: str   # 'real', 'synthetic', 'synthetic_assisted'
    is_demonstration_only: bool
    total_observations: int
    train_samples: int
    val_samples: int
    test_samples: int
    validation_date_range: Dict[str, Optional[str]]
    models: List[ModelBenchmarkResult] = field(default_factory=list)
    selected_best_model: Optional[str] = None
    selection_rationale: Optional[str] = None
    evaluated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_status": self.benchmark_status,
            "status_reason": self.status_reason,
            "data_provenance": self.data_provenance,
            "is_demonstration_only": self.is_demonstration_only,
            "total_observations": self.total_observations,
            "partition_sizes": {
                "train": self.train_samples,
                "validation": self.val_samples,
                "test": self.test_samples
            },
            "validation_date_range": self.validation_date_range,
            "models": [m.to_dict() for m in self.models],
            "selected_best_model": self.selected_best_model,
            "selection_rationale": self.selection_rationale,
            "evaluated_at": self.evaluated_at
        }


class ModelBenchmarkSuite:
    """Manages training, cross-validation, tolerance metrics, and selection of regression models."""

    def __init__(self, min_samples: int = MIN_CHRONOLOGICAL_SAMPLES):
        self.min_samples = min_samples
        self.splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        self.cv = ExpandingWindowCV(n_splits=3)

    def _get_model_candidates(self) -> List[Tuple[str, Dict[str, Any], Optional[Callable[[], Any]]]]:
        """Return configured model candidates with hyperparameter definitions."""
        candidates = [
            (
                "Historical Statistical Baseline",
                {"strategy": "event_grouped_mean", "fallback": "global_mean"},
                None  # Handled specially
            ),
            (
                "Linear Regression",
                {"fit_intercept": True, "copy_X": True},
                lambda: LinearRegression(fit_intercept=True)
            ),
            (
                "Random Forest Regressor",
                {"n_estimators": 50, "max_depth": 5, "random_state": 42},
                lambda: RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            ),
            (
                "HistGradientBoosting Regressor",
                {"max_iter": 100, "max_depth": 5, "random_state": 42},
                lambda: HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)
            )
        ]

        if HAS_XGBOOST:
            candidates.append((
                "XGBoost Regressor",
                {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.08, "random_state": 42},
                lambda: xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.08, random_state=42)
            ))
        else:
            candidates.append((
                "XGBoost Regressor",
                {"dependency_missing": "xgboost"},
                None
            ))

        return candidates

    def _evaluate_baseline(
        self,
        dev_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Tuple[ModelBenchmarkResult, Dict[str, float], float]:
        """Evaluate event-grouped historical baseline across expanding CV and validation partition."""
        start_t = time.perf_counter()

        cv_maes = []
        for fold_idx, cv_train, cv_val in self.cv.split(dev_df):
            event_means = cv_train.groupby("event_name")["total_delay_min"].mean().to_dict()
            global_mean = float(cv_train["total_delay_min"].mean())
            preds = cv_val["event_name"].map(event_means).fillna(global_mean).values
            y_val = cv_val["total_delay_min"].values.astype(float)
            cv_maes.append(float(mean_absolute_error(y_val, preds)))

        # Evaluate on validation partition
        train_portion = dev_df.iloc[:len(dev_df) - len(val_df)] if len(dev_df) > len(val_df) else dev_df
        event_means = train_portion.groupby("event_name")["total_delay_min"].mean().to_dict()
        global_mean = float(train_portion["total_delay_min"].mean())

        val_preds = val_df["event_name"].map(event_means).fillna(global_mean).values
        y_val_all = val_df["total_delay_min"].values.astype(float)

        duration_ms = (time.perf_counter() - start_t) * 1000.0
        val_mae = float(mean_absolute_error(y_val_all, val_preds))
        val_rmse = float(root_mean_squared_error(y_val_all, val_preds))
        val_medae = float(median_absolute_error(y_val_all, val_preds))
        val_tol = calculate_prediction_tolerance(y_val_all, val_preds, tolerance_min=5.0)

        # R2 calculation if sample size allows
        val_r2 = None
        if len(y_val_all) >= 2 and np.var(y_val_all) > 1e-6:
            try:
                val_r2 = round(float(r2_score(y_val_all, val_preds)), 3)
            except Exception:
                val_r2 = None

        result = ModelBenchmarkResult(
            model_name="Historical Statistical Baseline",
            status="evaluated",
            hyperparameters={"strategy": "event_grouped_mean", "fallback": "global_mean"},
            training_duration_ms=duration_ms,
            val_mae=round(val_mae, 3),
            val_rmse=round(val_rmse, 3),
            val_r2=val_r2,
            val_medae=round(val_medae, 3),
            val_tolerance_5m_pct=val_tol,
            cv_fold_maes=[round(m, 3) for m in cv_maes]
        )
        return result, event_means, global_mean

    def run_benchmark(
        self,
        df: pd.DataFrame,
        provenance: str = "synthetic"
    ) -> BenchmarkSuiteResult:
        """Run reproducible multi-algorithm benchmark over chronological partitions."""
        total_count = len(df)
        is_demo = (provenance != "real")

        # 1. Minimum Data Check
        if total_count < self.min_samples:
            reason = (
                f"Insufficient data for benchmarking: found {total_count} valid records, "
                f"minimum {self.min_samples} required for reliable chronological evaluation."
            )
            return BenchmarkSuiteResult(
                benchmark_status="insufficient_data",
                status_reason=reason,
                data_provenance=provenance,
                is_demonstration_only=is_demo,
                total_observations=total_count,
                train_samples=0,
                val_samples=0,
                test_samples=0,
                validation_date_range={"start": None, "end": None},
                models=[],
                selected_best_model=None,
                selection_rationale="Benchmarking aborted due to insufficient sample count."
            )

        # 2. Chronological Split (70% train, 15% val, 15% test)
        clean_sorted = self.splitter.sort_chronologically(df)
        train_df, val_df, test_df = self.splitter.split(clean_sorted)
        dev_df = pd.concat([train_df, val_df], ignore_index=True)

        val_meta = self.splitter.get_partition_metadata(val_df, "validation")
        val_date_range = {"start": val_meta.start_date, "end": val_meta.end_date}

        results: List[ModelBenchmarkResult] = []
        fitted_models: Dict[str, Any] = {}
        fitted_preprocessors: Dict[str, Any] = {}

        # 3. Evaluate Baseline
        baseline_res, baseline_means, baseline_global = self._evaluate_baseline(dev_df, val_df)
        results.append(baseline_res)

        # 4. Evaluate ML Candidates
        candidates = self._get_model_candidates()
        for name, params, builder in candidates:
            if name == "Historical Statistical Baseline":
                continue

            if builder is None:
                results.append(ModelBenchmarkResult(
                    model_name=name,
                    status="dependency_unavailable",
                    hyperparameters=params,
                    training_duration_ms=0.0,
                    failure_reason="Dependency not installed or incompatible in current environment."
                ))
                continue

            start_t = time.perf_counter()
            cv_maes = []
            try:
                # Expanding-window temporal CV on development data
                for fold_idx, cv_train, cv_val in self.cv.split(dev_df):
                    fold_prep = MicroDelayPreprocessor()
                    fold_prep.fit(cv_train)
                    X_tr = fold_prep.transform(cv_train)
                    y_tr = cv_train["total_delay_min"].values.astype(float)
                    X_v = fold_prep.transform(cv_val)
                    y_v = cv_val["total_delay_min"].values.astype(float)

                    fold_m = builder()
                    fold_m.fit(X_tr, y_tr)
                    preds = fold_m.predict(X_v)
                    cv_maes.append(float(mean_absolute_error(y_v, preds)))

                # Fit on development set (Train + Val) for validation evaluation
                dev_prep = MicroDelayPreprocessor()
                dev_prep.fit(train_df)
                X_train = dev_prep.transform(train_df)
                y_train = train_df["total_delay_min"].values.astype(float)

                X_val = dev_prep.transform(val_df)
                y_val = val_df["total_delay_min"].values.astype(float)

                cand_model = builder()
                cand_model.fit(X_train, y_train)
                val_preds = cand_model.predict(X_val)

                dur_ms = (time.perf_counter() - start_t) * 1000.0
                val_mae = float(mean_absolute_error(y_val, val_preds))
                val_rmse = float(root_mean_squared_error(y_val, val_preds))
                val_medae = float(median_absolute_error(y_val, val_preds))
                val_tol = calculate_prediction_tolerance(y_val, val_preds, tolerance_min=5.0)

                val_r2 = None
                if len(y_val) >= 2 and np.var(y_val) > 1e-6:
                    try:
                        val_r2 = round(float(r2_score(y_val, val_preds)), 3)
                    except Exception:
                        val_r2 = None

                model_res = ModelBenchmarkResult(
                    model_name=name,
                    status="evaluated",
                    hyperparameters=params,
                    training_duration_ms=dur_ms,
                    val_mae=round(val_mae, 3),
                    val_rmse=round(val_rmse, 3),
                    val_r2=val_r2,
                    val_medae=round(val_medae, 3),
                    val_tolerance_5m_pct=val_tol,
                    cv_fold_maes=[round(m, 3) for m in cv_maes]
                )
                results.append(model_res)
                fitted_models[name] = cand_model
                fitted_preprocessors[name] = dev_prep

            except Exception as e:
                results.append(ModelBenchmarkResult(
                    model_name=name,
                    status="failed",
                    hyperparameters=params,
                    training_duration_ms=0.0,
                    failure_reason=f"Model training error: {str(e)}"
                ))

        # 5. Model Selection Based on Validation MAE & Prediction Tolerance
        evaluated_models = [m for m in results if m.status == "evaluated" and m.val_mae is not None]
        if not evaluated_models:
            return BenchmarkSuiteResult(
                benchmark_status="failed",
                status_reason="All candidate models failed evaluation.",
                data_provenance=provenance,
                is_demonstration_only=is_demo,
                total_observations=total_count,
                train_samples=len(train_df),
                val_samples=len(val_df),
                test_samples=len(test_df),
                validation_date_range=val_date_range,
                models=results,
                selected_best_model=None,
                selection_rationale="No candidate completed validation successfully."
            )

        # Primary criterion: lowest validation MAE; Secondary criterion: highest 5-min tolerance
        sorted_models = sorted(
            evaluated_models,
            key=lambda m: (m.val_mae, -(m.val_tolerance_5m_pct or 0.0))
        )
        best_candidate = sorted_models[0]
        selected_name = best_candidate.model_name

        # 6. Final Evaluation: Untouched Holdout Test Set Evaluated ONCE for Winner
        y_test = test_df["total_delay_min"].values.astype(float)

        if selected_name == "Historical Statistical Baseline":
            test_preds = test_df["event_name"].map(baseline_means).fillna(baseline_global).values
        else:
            win_prep = fitted_preprocessors[selected_name]
            win_model = fitted_models[selected_name]
            X_test = win_prep.transform(test_df)
            test_preds = win_model.predict(X_test)

        best_candidate.test_mae = round(float(mean_absolute_error(y_test, test_preds)), 3)
        best_candidate.test_rmse = round(float(root_mean_squared_error(y_test, test_preds)), 3)
        best_candidate.test_medae = round(float(median_absolute_error(y_test, test_preds)), 3)
        best_candidate.test_tolerance_5m_pct = calculate_prediction_tolerance(y_test, test_preds, tolerance_min=5.0)
        if len(y_test) >= 2 and np.var(y_test) > 1e-6:
            try:
                best_candidate.test_r2 = round(float(r2_score(y_test, test_preds)), 3)
            except Exception:
                best_candidate.test_r2 = None

        rationale = (
            f"Selected '{selected_name}' with lowest validation MAE ({best_candidate.val_mae}m) "
            f"and validation +/-5m tolerance ({best_candidate.val_tolerance_5m_pct}%). "
            f"Holdout test MAE confirmed at {best_candidate.test_mae}m."
        )

        return BenchmarkSuiteResult(
            benchmark_status="completed",
            status_reason="Multi-algorithm benchmarking and untouched holdout test evaluation completed successfully.",
            data_provenance=provenance,
            is_demonstration_only=is_demo,
            total_observations=total_count,
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            validation_date_range=val_date_range,
            models=results,
            selected_best_model=selected_name,
            selection_rationale=rationale
        )
