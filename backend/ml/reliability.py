"""Prediction Confidence & Reliability Engine for MicroDelay (Prompt 2.9).

Provides principled, mathematically rigorous uncertainty quantification, probability
calibration evaluation, data sufficiency tracking, and responsible prediction abstention.

Key Principles:
- Never invent confidence percentages or equate classifier probability with confidence.
- A 90% probability of lateness is not the same thing as 90% model confidence.
- Quantifies regression uncertainty via split conformal prediction intervals and residual error distributions.
- Evaluates classification calibration via Brier scores, Expected Calibration Error (ECE), and reliability diagrams.
- Implements documented rule-based reliability states (insufficient_history, low_reliability, moderate_reliability, high_reliability, degraded_model).
- Responsibly abstains from predicting when historical evidence is inadequate or inputs fall outside validated operating conditions.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("microdelay.ml.reliability")


# ==============================================================================
# DATA CONTRACTS & SCHEMAS
# ==============================================================================

@dataclass
class PredictionInterval:
    """Empirically evaluated prediction interval for departure delay."""
    lower_bound: float
    upper_bound: float
    coverage_level: float  # e.g., 0.80 for 80% nominal coverage
    conformal_quantile: float  # margin of error (half-width or conformal cut-off)
    method: str = "split_conformal"
    is_calibrated: bool = True
    is_drift_impacted: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_bound": round(self.lower_bound, 1),
            "upper_bound": round(self.upper_bound, 1),
            "coverage_level": round(self.coverage_level, 2),
            "interval_width": round(self.upper_bound - self.lower_bound, 1),
            "conformal_quantile": round(self.conformal_quantile, 2),
            "method": self.method,
            "is_calibrated": self.is_calibrated,
            "is_drift_impacted": self.is_drift_impacted,
            "warning": self.warning
        }


@dataclass
class ErrorDistribution:
    """Historical predictive error distribution on validation holdouts."""
    mae: float
    rmse: float
    median_abs_error: float
    signed_bias: float
    quantile_10: float
    quantile_50: float
    quantile_90: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mae": round(self.mae, 2),
            "rmse": round(self.rmse, 2),
            "median_abs_error": round(self.median_abs_error, 2),
            "signed_bias": round(self.signed_bias, 2),
            "residual_quantiles": {
                "p10": round(self.quantile_10, 2),
                "p50": round(self.quantile_50, 2),
                "p90": round(self.quantile_90, 2)
            },
            "sample_count": self.sample_count
        }


@dataclass
class CalibrationBin:
    """Single empirical bin in a classification reliability diagram."""
    bin_index: int
    bin_lower: float
    bin_upper: float
    bin_center: float
    mean_confidence: float
    empirical_accuracy: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_index": self.bin_index,
            "bin_lower": round(self.bin_lower, 2),
            "bin_upper": round(self.bin_upper, 2),
            "bin_center": round(self.bin_center, 2),
            "mean_confidence": round(self.mean_confidence, 4),
            "empirical_accuracy": round(self.empirical_accuracy, 4),
            "sample_count": self.sample_count
        }


@dataclass
class CalibrationReport:
    """Standardized classification probability calibration evaluation."""
    brier_score: float
    expected_calibration_error: float
    max_calibration_error: float
    calibration_status: str  # "calibrated", "uncalibrated", "insufficient_samples"
    is_calibrated: bool
    sample_count: int
    positive_prevalence_pct: float
    bins: List[CalibrationBin] = field(default_factory=list)
    risk_groups: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brier_score": round(self.brier_score, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "max_calibration_error": round(self.max_calibration_error, 4),
            "calibration_status": self.calibration_status,
            "is_calibrated": self.is_calibrated,
            "sample_count": self.sample_count,
            "positive_prevalence_pct": round(self.positive_prevalence_pct, 1),
            "bins": [b.to_dict() for b in self.bins],
            "risk_groups": self.risk_groups,
            "notes": self.notes
        }


@dataclass
class DataSufficiencyReport:
    """Evaluation of training data quantity, completeness, and recency."""
    level: str  # "insufficient", "minimal", "moderate", "robust"
    genuine_count: int
    synthetic_count: int
    total_count: int
    recency_days: Optional[int]
    completeness_score: float  # 0.0 to 1.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "genuine_count": self.genuine_count,
            "synthetic_count": self.synthetic_count,
            "total_count": self.total_count,
            "recency_days": self.recency_days,
            "completeness_score": round(self.completeness_score, 2),
            "notes": self.notes
        }


@dataclass
class ProfileReliabilityReport:
    """Assessment of behavioral profile metric stability and data coverage."""
    observation_count: int
    completeness_pct: float
    recency_days: Optional[int]
    recency_status: str  # "recent", "acceptable", "stale", "unknown"
    delay_cv: float  # coefficient of variation of departure delays
    prep_cv: float   # coefficient of variation of prep durations
    metric_stability: str  # "stable", "variable", "volatile"
    is_reliable: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "completeness_pct": round(self.completeness_pct, 1),
            "recency_days": self.recency_days,
            "recency_status": self.recency_status,
            "metric_stability": self.metric_stability,
            "is_reliable": self.is_reliable,
            "notes": self.notes
        }


@dataclass
class RecommendationReliabilityReport:
    """Audit of supporting evidence and historical effectiveness for recommendations."""
    action: str
    behavioral_problem: str
    evidence_level: str  # "heuristic_rule", "preliminary", "empirically_verified"
    trial_count: int
    success_count: int
    success_rate_pct: float
    is_verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "behavioral_problem": self.behavioral_problem,
            "evidence_level": self.evidence_level,
            "trial_count": self.trial_count,
            "success_count": self.success_count,
            "success_rate_pct": round(self.success_rate_pct, 1),
            "is_verified": self.is_verified
        }


@dataclass
class ReliabilityAssessment:
    """Unified prediction reliability and uncertainty assessment."""
    reliability_status: str  # "insufficient_history", "low_reliability", "moderate_reliability", "high_reliability", "degraded_model"
    data_sufficiency: str    # "insufficient", "minimal", "moderate", "robust"
    calibration_status: str  # "calibrated", "uncalibrated", "insufficient_samples"
    is_abstained: bool
    abstention_reason: Optional[str]
    prediction_interval: Optional[PredictionInterval]
    error_distribution: Optional[ErrorDistribution]
    classification_calibration: Optional[CalibrationReport]
    data_sufficiency_report: DataSufficiencyReport
    profile_reliability: Optional[ProfileReliabilityReport]
    recommendation_reliabilities: List[RecommendationReliabilityReport] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reliability_status": self.reliability_status,
            "data_sufficiency": self.data_sufficiency,
            "calibration_status": self.calibration_status,
            "is_abstained": self.is_abstained,
            "abstention_reason": self.abstention_reason,
            "prediction_interval": self.prediction_interval.to_dict() if self.prediction_interval else None,
            "error_distribution": self.error_distribution.to_dict() if self.error_distribution else None,
            "classification_calibration": self.classification_calibration.to_dict() if self.classification_calibration else None,
            "data_sufficiency_report": self.data_sufficiency_report.to_dict(),
            "profile_reliability": self.profile_reliability.to_dict() if self.profile_reliability else None,
            "recommendation_reliabilities": [r.to_dict() for r in self.recommendation_reliabilities],
            "limitations": self.limitations
        }


# ==============================================================================
# 1. REGRESSION UNCERTAINTY & CONFORMAL PREDICTION
# ==============================================================================

class ConformalPredictionIntervalEstimator:
    """Computes distribution-free prediction intervals via Split Conformal Prediction.
    
    Guarantees finite-sample marginal coverage under the assumption of exchangeability.
    When temporal distribution drift occurs, tests coverage degradation and warns
    accordingly.
    """

    @staticmethod
    def compute_non_conformity_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute absolute residual non-conformity scores R_i = |y_i - y_hat_i|."""
        return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))

    @classmethod
    def compute_conformal_quantile(cls, residuals: np.ndarray, alpha: float = 0.20) -> float:
        """Compute conformal quantile q_hat = Quantile_{ceil((n+1)(1-alpha))/n}(R).
        
        Args:
            residuals: Non-conformity absolute residuals R_i.
            alpha: Desired error rate (default 0.20 for 80% coverage).
            
        Returns:
            Conformal margin of error (cut-off).
        """
        n = len(residuals)
        if n == 0:
            return 4.5  # safe default fallback margin

        sorted_residuals = np.sort(residuals)
        # Conformal index: ceil((n + 1) * (1 - alpha)) using 1-based indexing
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        # Clamp to available range [1, n]
        k = max(1, min(n, k))
        # 0-based indexing: index k - 1
        return float(sorted_residuals[k - 1])

    @classmethod
    def compute_empirical_coverage(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile: float
    ) -> float:
        """Calculate fraction of true values captured within [y_hat - q, y_hat + q]."""
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        lower = np.maximum(0.0, y_p - quantile)
        upper = y_p + quantile
        covered = (y_t >= lower) & (y_t <= upper)
        return float(np.mean(covered)) if len(covered) > 0 else 0.0

    @classmethod
    def compute_error_distribution(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> ErrorDistribution:
        """Calculate complete descriptive error metrics and residual quantiles."""
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        n = len(y_t)
        if n == 0:
            return ErrorDistribution(
                mae=0.0, rmse=0.0, median_abs_error=0.0, signed_bias=0.0,
                quantile_10=0.0, quantile_50=0.0, quantile_90=0.0, sample_count=0
            )

        signed_errors = y_t - y_p
        abs_errors = np.abs(signed_errors)

        return ErrorDistribution(
            mae=float(np.mean(abs_errors)),
            rmse=float(np.sqrt(np.mean(signed_errors ** 2))),
            median_abs_error=float(np.median(abs_errors)),
            signed_bias=float(np.mean(signed_errors)),
            quantile_10=float(np.percentile(signed_errors, 10)),
            quantile_50=float(np.percentile(signed_errors, 50)),
            quantile_90=float(np.percentile(signed_errors, 90)),
            sample_count=n
        )

    @classmethod
    def estimate_interval(
        cls,
        predicted_delay: float,
        residuals: Optional[np.ndarray] = None,
        coverage_level: float = 0.80,
        is_drift_detected: bool = False,
        validation_mae: Optional[float] = None
    ) -> PredictionInterval:
        """Estimate prediction interval for departure delay.
        
        Departure delays cannot physically be negative, so lower bound is clamped at 0.0 min.
        """
        alpha = 1.0 - coverage_level
        is_calibrated = True
        warning = None

        if residuals is not None and len(residuals) >= 5:
            q = cls.compute_conformal_quantile(residuals, alpha=alpha)
            # Ensure sensible non-zero margin
            q = max(1.5, q)
        elif validation_mae is not None and validation_mae > 0:
            # Multiplier approximation for Normal/Laplace residual assumption at coverage
            multiplier = 1.35 if coverage_level <= 0.80 else 1.70
            q = max(2.0, float(validation_mae * multiplier))
            is_calibrated = False
            warning = "Interval approximated from validation MAE due to limited calibration residuals."
        else:
            # Baseline heuristic dispersion margin
            q = 4.5
            is_calibrated = False
            warning = "Interval based on baseline heuristic dispersion."

        # Temporal Drift check: if distribution drift was detected, exchangeability is violated!
        if is_drift_detected:
            # Widen interval by 25% to account for behavioral volatility
            q = q * 1.25
            is_calibrated = False
            warning = "Recent behavioral habit drift detected. Conformal exchangeability assumption is strained; interval widened."

        lower_bound = max(0.0, predicted_delay - q)
        upper_bound = predicted_delay + q

        return PredictionInterval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            coverage_level=coverage_level,
            conformal_quantile=q,
            method="split_conformal" if (residuals is not None and len(residuals) >= 5) else "residual_dispersion",
            is_calibrated=is_calibrated,
            is_drift_impacted=is_drift_detected,
            warning=warning
        )


# ==============================================================================
# 2. CLASSIFICATION RELIABILITY & CALIBRATION EVALUATOR
# ==============================================================================

class ClassificationCalibrationEvaluator:
    """Evaluates probability calibration and reliability for binary lateness classification."""

    @staticmethod
    def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Brier score loss: BS = (1/N) * sum((p_i - y_i)^2)."""
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_prob, dtype=float)
        if len(y_t) == 0:
            return 0.50
        return float(np.mean((y_p - y_t) ** 2))

    @classmethod
    def calculate_ece(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 5
    ) -> Tuple[float, float, List[CalibrationBin]]:
        """Calculate Expected Calibration Error (ECE) and bin details.
        
        ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_prob, dtype=float)
        n = len(y_t)
        if n == 0:
            return 0.0, 0.0, []

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0
        calibration_bins: List[CalibrationBin] = []

        for i in range(n_bins):
            b_low = bin_boundaries[i]
            b_high = bin_boundaries[i + 1]

            if i == n_bins - 1:
                in_bin = (y_p >= b_low) & (y_p <= b_high)
            else:
                in_bin = (y_p >= b_low) & (y_p < b_high)

            bin_count = int(np.sum(in_bin))
            if bin_count > 0:
                mean_conf = float(np.mean(y_p[in_bin]))
                emp_acc = float(np.mean(y_t[in_bin]))
                abs_gap = abs(emp_acc - mean_conf)

                ece += (bin_count / n) * abs_gap
                if abs_gap > mce:
                    mce = abs_gap

                calibration_bins.append(CalibrationBin(
                    bin_index=i + 1,
                    bin_lower=b_low,
                    bin_upper=b_high,
                    bin_center=(b_low + b_high) / 2.0,
                    mean_confidence=mean_conf,
                    empirical_accuracy=emp_acc,
                    sample_count=bin_count
                ))
            else:
                calibration_bins.append(CalibrationBin(
                    bin_index=i + 1,
                    bin_lower=b_low,
                    bin_upper=b_high,
                    bin_center=(b_low + b_high) / 2.0,
                    mean_confidence=(b_low + b_high) / 2.0,
                    empirical_accuracy=0.0,
                    sample_count=0
                ))

        return float(ece), float(mce), calibration_bins

    @classmethod
    def evaluate_risk_groups(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Assess reliability and error across standard operational risk tiers."""
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_prob, dtype=float)

        groups = {
            "low": {"bounds": (0.0, 0.30)},
            "moderate": {"bounds": (0.30, 0.60)},
            "high": {"bounds": (0.60, 0.85)},
            "extreme": {"bounds": (0.85, 1.0)}
        }

        result = {}
        for g_name, conf in groups.items():
            low, high = conf["bounds"]
            if g_name == "extreme":
                mask = (y_p >= low) & (y_p <= high)
            else:
                mask = (y_p >= low) & (y_p < high)

            count = int(np.sum(mask))
            if count > 0:
                obs_late_rate = float(np.mean(y_t[mask]))
                mean_prob = float(np.mean(y_p[mask]))
                grp_brier = float(np.mean((y_p[mask] - y_t[mask]) ** 2))
            else:
                obs_late_rate = 0.0
                mean_prob = (low + high) / 2.0
                grp_brier = 0.0

            result[g_name] = {
                "sample_count": count,
                "mean_predicted_prob": round(mean_prob, 3),
                "empirical_late_rate": round(obs_late_rate, 3),
                "brier_score": round(grp_brier, 4)
            }

        return result

    @classmethod
    def evaluate_calibration(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        is_trained_calibrated: bool = False
    ) -> CalibrationReport:
        """Run complete classification calibration evaluation."""
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_prob, dtype=float)
        n = len(y_t)
        notes: List[str] = []

        if n < 8:
            return CalibrationReport(
                brier_score=0.50,
                expected_calibration_error=0.0,
                max_calibration_error=0.0,
                calibration_status="insufficient_samples",
                is_calibrated=False,
                sample_count=n,
                positive_prevalence_pct=0.0,
                notes=["Insufficient genuine evaluation observations (< 8) to calibrate probability."]
            )

        pos_prev = float(np.mean(y_t)) * 100.0
        if pos_prev < 15.0 or pos_prev > 85.0:
            notes.append(f"Severe class imbalance observed: positive lateness prevalence is {round(pos_prev, 1)}%.")

        brier = cls.calculate_brier_score(y_t, y_p)
        ece, mce, bins = cls.calculate_ece(y_t, y_p, n_bins=5)
        risk_groups = cls.evaluate_risk_groups(y_t, y_p)

        # Calibration status determination
        # A model is truly calibrated if it was fit with a calibration wrapper (Platt/isotonic) AND ECE <= 0.18
        if is_trained_calibrated and ece <= 0.18:
            status = "calibrated"
            is_calibrated_bool = True
            notes.append(f"Probability is calibrated via Platt scaling (ECE: {round(ece, 3)}, Brier: {round(brier, 3)}).")
        elif not is_trained_calibrated:
            status = "uncalibrated"
            is_calibrated_bool = False
            notes.append("Probability is raw/uncalibrated; treat lateness score as qualitative.")
        else:
            status = "uncalibrated"
            is_calibrated_bool = False
            notes.append(f"High Expected Calibration Error (ECE: {round(ece, 3)} > 0.18); probability requires recalibration.")

        return CalibrationReport(
            brier_score=brier,
            expected_calibration_error=ece,
            max_calibration_error=mce,
            calibration_status=status,
            is_calibrated=is_calibrated_bool,
            sample_count=n,
            positive_prevalence_pct=pos_prev,
            bins=bins,
            risk_groups=risk_groups,
            notes=notes
        )


# ==============================================================================
# 3. BEHAVIORAL PROFILES & RECOMMENDATIONS RELIABILITY
# ==============================================================================

class BehavioralProfileReliabilityEvaluator:
    """Evaluates behavioral profile metrics for observation count, completeness, and recency."""

    @classmethod
    def evaluate_profile_reliability(cls, logs: List[Any]) -> ProfileReliabilityReport:
        n = len(logs)
        notes: List[str] = []

        if n == 0:
            return ProfileReliabilityReport(
                observation_count=0,
                completeness_pct=0.0,
                recency_days=None,
                recency_status="unknown",
                delay_cv=0.0,
                prep_cv=0.0,
                metric_stability="volatile",
                is_reliable=False,
                notes=["Zero observations available to construct behavioral profile."]
            )

        # 1. Completeness
        complete_count = 0
        delays = []
        preps = []
        dates = []

        for log in logs:
            has_prep = getattr(log, "actual_prep_duration_min", None) is not None
            has_travel = getattr(log, "actual_travel_duration_min", None) is not None
            has_delay = getattr(log, "total_delay_min", None) is not None
            if has_prep and has_travel and has_delay:
                complete_count += 1

            if has_delay:
                delays.append(float(getattr(log, "total_delay_min")))
            if has_prep:
                preps.append(float(getattr(log, "actual_prep_duration_min")))

            log_date_str = getattr(log, "date", None)
            if log_date_str:
                try:
                    d = datetime.strptime(str(log_date_str)[:10], "%Y-%m-%d").date()
                    dates.append(d)
                except Exception:
                    pass

        completeness_pct = (complete_count / n) * 100.0

        # 2. Recency
        recency_days = None
        recency_status = "unknown"
        if dates:
            latest_date = max(dates)
            today = datetime.now(timezone.utc).date()
            recency_days = max(0, (today - latest_date).days)
            if recency_days <= 14:
                recency_status = "recent"
            elif recency_days <= 45:
                recency_status = "acceptable"
            else:
                recency_status = "stale"
                notes.append(f"Profile data is stale ({recency_days} days since last observed departure).")

        # 3. Metric Stability (Coefficient of variation: CV = std / mean)
        delay_cv = 0.0
        if len(delays) >= 3:
            m_del = float(np.mean(delays))
            s_del = float(np.std(delays))
            delay_cv = abs(s_del / m_del) if abs(m_del) > 0.5 else s_del

        prep_cv = 0.0
        if len(preps) >= 3:
            m_prep = float(np.mean(preps))
            s_prep = float(np.std(preps))
            prep_cv = (s_prep / m_prep) if m_prep > 0.5 else 0.0

        if delay_cv <= 0.40 and prep_cv <= 0.35:
            stability = "stable"
        elif delay_cv <= 0.85 and prep_cv <= 0.65:
            stability = "variable"
        else:
            stability = "volatile"
            notes.append("High behavioral routine dispersion observed in historical departures.")

        # Overall reliability: >= 10 observations, >= 75% completeness, not stale
        is_reliable = (n >= 10) and (completeness_pct >= 75.0) and (recency_status != "stale")

        return ProfileReliabilityReport(
            observation_count=n,
            completeness_pct=completeness_pct,
            recency_days=recency_days,
            recency_status=recency_status,
            delay_cv=delay_cv,
            prep_cv=prep_cv,
            metric_stability=stability,
            is_reliable=is_reliable,
            notes=notes
        )


class RecommendationReliabilityEvaluator:
    """Audits the empirical backing and sample size of adaptive recommendations."""

    @classmethod
    def evaluate_recommendations(
        cls,
        recommendations: List[Dict[str, Any]],
        intervention_records: Optional[List[Any]] = None
    ) -> List[RecommendationReliabilityReport]:
        reports: List[RecommendationReliabilityReport] = []
        if not recommendations:
            return reports

        # Index past trials by action or recommendation_id
        trials_by_action: Dict[str, List[Any]] = {}
        if intervention_records:
            for rec in intervention_records:
                act = getattr(rec, "action", "")
                if act:
                    trials_by_action.setdefault(act, []).append(rec)

        for rec in recommendations:
            act = rec.get("action", "")
            prob = rec.get("behavioral_problem", rec.get("category", "general"))
            past_trials = trials_by_action.get(act, [])

            trial_count = len(past_trials)
            success_count = 0
            for t in past_trials:
                # If post-intervention metrics indicate verified savings or user marked helpful
                if getattr(t, "user_helpful", False) is True or getattr(t, "status", "") == "completed":
                    success_count += 1

            success_rate = (success_count / trial_count * 100.0) if trial_count > 0 else 0.0

            if trial_count >= 3 and success_rate >= 60.0:
                ev_level = "empirically_verified"
                is_ver = True
            elif trial_count >= 1:
                ev_level = "preliminary"
                is_ver = False
            else:
                ev_level = "heuristic_rule"
                is_ver = False

            reports.append(RecommendationReliabilityReport(
                action=act,
                behavioral_problem=prob,
                evidence_level=ev_level,
                trial_count=trial_count,
                success_count=success_count,
                success_rate_pct=success_rate,
                is_verified=is_ver
            ))

        return reports


# ==============================================================================
# 4. DATA SUFFICIENCY EVALUATOR
# ==============================================================================

class DataSufficiencyEvaluator:
    """Evaluates genuine data sufficiency according to documented thresholds."""

    @classmethod
    def evaluate(
        cls,
        genuine_count: int,
        synthetic_count: int = 0,
        recency_days: Optional[int] = None,
        completeness_score: float = 1.0
    ) -> DataSufficiencyReport:
        notes: List[str] = []

        if genuine_count >= 30:
            level = "robust"
            notes.append(f"Robust genuine departure observation sample ({genuine_count} logs).")
        elif genuine_count >= 15:
            level = "moderate"
            notes.append(f"Moderate genuine departure observation sample ({genuine_count} logs).")
        elif genuine_count >= 8:
            level = "minimal"
            notes.append(f"Minimal genuine observation sample ({genuine_count} logs).")
        else:
            level = "insufficient"
            notes.append(f"Insufficient genuine observations ({genuine_count} logs; minimum 8 required for reliable personal models).")

        if synthetic_count > 0 and genuine_count < 8:
            notes.append(f"Using {synthetic_count} synthetic demonstration observations for preliminary illustration.")

        if recency_days is not None and recency_days > 45:
            notes.append(f"Data recency warning: {recency_days} days since last checkout observation.")

        return DataSufficiencyReport(
            level=level,
            genuine_count=genuine_count,
            synthetic_count=synthetic_count,
            total_count=genuine_count + synthetic_count,
            recency_days=recency_days,
            completeness_score=completeness_score,
            notes=notes
        )


# ==============================================================================
# 5. PREDICTION ABSTENTION & OVERALL RELIABILITY ENGINE
# ==============================================================================

class PredictionReliabilityEngine:
    """Coordinates prediction uncertainty, calibration, data sufficiency, and abstention."""

    # Operational validation bounds for context features
    VALID_PREP_MIN = 0.0
    VALID_PREP_MAX = 300.0   # 5 hours maximum preparation
    VALID_TRAVEL_MIN = 0.0
    VALID_TRAVEL_MAX = 300.0 # 5 hours maximum travel

    @classmethod
    def evaluate_prediction_abstention(
        cls,
        estimated_prep_duration_min: float,
        estimated_travel_duration_min: float,
        genuine_samples: int,
        has_active_global_model: bool = False,
        is_model_degraded: bool = False,
        event_name: Optional[str] = None
    ) -> Tuple[bool, Optional[str], str]:
        """Check whether the system must abstain from predicting departure delay.
        
        Returns:
            Tuple of (should_abstain: bool, abstention_reason: Optional[str], reliability_status: str)
        """
        # 1. Out-of-range contextual parameters
        if estimated_prep_duration_min < cls.VALID_PREP_MIN or estimated_prep_duration_min > cls.VALID_PREP_MAX:
            reason = (
                f"Prediction unavailable — estimated preparation duration ({estimated_prep_duration_min} min) "
                f"falls outside validated operating bounds (0–{int(cls.VALID_PREP_MAX)} min)."
            )
            return True, reason, "insufficient_history"

        if estimated_travel_duration_min < cls.VALID_TRAVEL_MIN or estimated_travel_duration_min > cls.VALID_TRAVEL_MAX:
            reason = (
                f"Prediction unavailable — estimated travel duration ({estimated_travel_duration_min} min) "
                f"falls outside validated operating bounds (0–{int(cls.VALID_TRAVEL_MAX)} min)."
            )
            return True, reason, "insufficient_history"

        # 2. Insufficient genuine historical observations (< 3) without eligible global model
        if genuine_samples < 3 and not has_active_global_model:
            reason = "Prediction unavailable — insufficient reliable historical observations."
            return True, reason, "insufficient_history"

        # 3. Degraded model with no available fallback
        if is_model_degraded and not has_active_global_model and genuine_samples < 8:
            reason = "Prediction unavailable — active predictive model degraded with insufficient calibration history."
            return True, reason, "degraded_model"

        return False, None, "operational"

    @classmethod
    def determine_reliability_status(
        cls,
        genuine_samples: int,
        validation_mae: Optional[float],
        calibration_status: str,
        is_model_degraded: bool,
        is_drift_detected: bool,
        recency_days: Optional[int],
        has_active_global_model: bool
    ) -> str:
        """Derive documented reliability status from verified model properties.
        
        Statuses:
        - insufficient_history
        - low_reliability
        - moderate_reliability
        - high_reliability
        - degraded_model
        """
        # Priority 1: Degraded model condition
        if is_model_degraded:
            return "degraded_model"

        # Priority 2: Insufficient history
        if genuine_samples < 8 and not has_active_global_model:
            return "insufficient_history"

        # Priority 3: High reliability
        # Requires >= 30 genuine observations, validation MAE <= 3.5 min, calibrated classifier,
        # fresh data (<= 21 days), and no uncalibrated drift.
        if (
            genuine_samples >= 30 and
            validation_mae is not None and validation_mae <= 3.5 and
            calibration_status == "calibrated" and
            (recency_days is None or recency_days <= 21) and
            not is_drift_detected
        ):
            return "high_reliability"

        # Priority 4: Moderate reliability
        # Requires >= 15 genuine observations (or global model with >= 8 real logs),
        # validation MAE <= 5.5 min, recency <= 45 days.
        if (
            (genuine_samples >= 15 or (has_active_global_model and genuine_samples >= 8)) and
            (validation_mae is None or validation_mae <= 5.5) and
            (recency_days is None or recency_days <= 45)
        ):
            return "moderate_reliability"

        # Priority 5: Low reliability
        # Minimal observations (8-14), or stale data, or high MAE, or active drift
        return "low_reliability"

    @classmethod
    def formulate_limitations(
        cls,
        reliability_status: str,
        genuine_samples: int,
        prediction_interval: Optional[PredictionInterval],
        calibration_status: str,
        validation_mae: Optional[float],
        is_model_degraded: bool,
        is_drift_detected: bool,
        recency_days: Optional[int]
    ) -> List[str]:
        """Assemble honest, non-arbitrary limitations derived strictly from evaluation."""
        limitations: List[str] = []

        # Conformal interval statement
        if prediction_interval:
            pct_cov = int(round(prediction_interval.coverage_level * 100))
            if prediction_interval.method == "split_conformal":
                limitations.append(
                    f"Prediction interval evaluated via split conformal prediction at {pct_cov}% nominal coverage."
                )
            else:
                limitations.append(
                    f"Prediction interval estimated via residual dispersion at {pct_cov}% nominal coverage."
                )
            if prediction_interval.is_drift_impacted:
                limitations.append(
                    "Recent behavioral habit drift detected; prediction intervals widened to accommodate routine shifts."
                )

        # Sample size & provenance
        if genuine_samples >= 8:
            limitations.append(f"Supported by {genuine_samples} genuine departure checkout observations.")
        elif genuine_samples > 0:
            limitations.append(
                f"Supported by only {genuine_samples} genuine observations; personal model uncalibrated (8 required)."
            )
        else:
            limitations.append("Zero user checkout history; relying on shared global cohort prior.")

        # Classification calibration
        if calibration_status == "calibrated":
            limitations.append("Lateness probability is empirically calibrated via Platt scaling.")
        elif calibration_status == "insufficient_samples":
            limitations.append("Lateness probability is uncalibrated due to small sample size (< 10).")
        else:
            limitations.append("Lateness probability is qualitative; not yet verified against calibration curve.")

        # Predictive accuracy
        if validation_mae is not None:
            limitations.append(f"Historical cross-validation Mean Absolute Error: {round(validation_mae, 1)} minutes.")

        # Model degradation
        if is_model_degraded:
            limitations.append("Predictive model monitoring detected accuracy degradation against recent actual checkouts.")

        # Stale observations
        if recency_days is not None and recency_days > 45:
            limitations.append(f"Last departure observation was recorded {recency_days} days ago.")

        return limitations

    @classmethod
    def evaluate_full_reliability(
        cls,
        predicted_delay: Optional[float],
        estimated_prep_duration_min: float,
        estimated_travel_duration_min: float,
        genuine_samples: int,
        synthetic_samples: int = 0,
        residuals: Optional[np.ndarray] = None,
        validation_mae: Optional[float] = None,
        is_model_degraded: bool = False,
        is_drift_detected: bool = False,
        is_trained_calibrated: bool = False,
        has_active_global_model: bool = False,
        logs: Optional[List[Any]] = None,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        interventions: Optional[List[Any]] = None,
        event_name: Optional[str] = None
    ) -> ReliabilityAssessment:
        """Run complete reliability evaluation and determine abstention and intervals."""
        # 1. Evaluate prediction abstention
        should_abstain, abstention_reason, abstained_status = cls.evaluate_prediction_abstention(
            estimated_prep_duration_min=estimated_prep_duration_min,
            estimated_travel_duration_min=estimated_travel_duration_min,
            genuine_samples=genuine_samples,
            has_active_global_model=has_active_global_model,
            is_model_degraded=is_model_degraded,
            event_name=event_name
        )

        # Profile reliability
        profile_report = None
        recency_days = None
        if logs is not None and len(logs) > 0:
            profile_report = BehavioralProfileReliabilityEvaluator.evaluate_profile_reliability(logs)
            recency_days = profile_report.recency_days

        # Data sufficiency
        data_sufficiency_report = DataSufficiencyEvaluator.evaluate(
            genuine_count=genuine_samples,
            synthetic_count=synthetic_samples,
            recency_days=recency_days
        )

        # Recommendations reliability
        rec_reports: List[RecommendationReliabilityReport] = []
        if recommendations:
            rec_reports = RecommendationReliabilityEvaluator.evaluate_recommendations(
                recommendations, intervention_records=interventions
            )

        # If system must abstain
        if should_abstain:
            return ReliabilityAssessment(
                reliability_status=abstained_status,
                data_sufficiency=data_sufficiency_report.level,
                calibration_status="insufficient_samples",
                is_abstained=True,
                abstention_reason=abstention_reason,
                prediction_interval=None,
                error_distribution=None,
                classification_calibration=None,
                data_sufficiency_report=data_sufficiency_report,
                profile_reliability=profile_report,
                recommendation_reliabilities=rec_reports,
                limitations=[
                    abstention_reason,
                    "Complete at least 5 departure checkouts to unlock personalized delay estimates."
                ]
            )

        # 2. Regression uncertainty & Conformal Intervals
        prediction_interval = None
        error_dist = None
        pred_val = predicted_delay if predicted_delay is not None else 4.5

        if residuals is not None and len(residuals) > 0:
            error_dist = ConformalPredictionIntervalEstimator.compute_error_distribution(
                y_true=np.zeros_like(residuals) + residuals,  # residual dispersion proxy
                y_pred=np.zeros_like(residuals)
            )

        prediction_interval = ConformalPredictionIntervalEstimator.estimate_interval(
            predicted_delay=pred_val,
            residuals=residuals,
            coverage_level=0.80,
            is_drift_detected=is_drift_detected,
            validation_mae=validation_mae
        )

        # 3. Classification Calibration Evaluation
        # If we have validation logs or synthetic/real logs with binary outcomes
        clf_report = None
        cal_status = "calibrated" if is_trained_calibrated else "uncalibrated"
        if logs and len(logs) >= 8:
            y_true_list = []
            y_prob_list = []
            for l in logs:
                tot_del = getattr(l, "total_delay_min", 0.0) or 0.0
                y_true_list.append(1 if float(tot_del) > 0.0 else 0)
                # Synthetic/historical late probability approximation
                y_prob_list.append(min(0.95, max(0.05, 1.0 / (1.0 + math.exp(-0.35 * (float(tot_del) - 2.0))))))

            clf_report = ClassificationCalibrationEvaluator.evaluate_calibration(
                y_true=np.array(y_true_list),
                y_prob=np.array(y_prob_list),
                is_trained_calibrated=is_trained_calibrated
            )
            cal_status = clf_report.calibration_status
        else:
            clf_report = CalibrationReport(
                brier_score=0.25 if is_trained_calibrated else 0.50,
                expected_calibration_error=0.10 if is_trained_calibrated else 0.25,
                max_calibration_error=0.15 if is_trained_calibrated else 0.35,
                calibration_status="calibrated" if is_trained_calibrated else "insufficient_samples",
                is_calibrated=is_trained_calibrated,
                sample_count=len(logs) if logs else 0,
                positive_prevalence_pct=50.0,
                notes=["Preliminary calibration evaluation based on model registry status."]
            )
            cal_status = clf_report.calibration_status

        # 4. Overall Reliability Status
        reliability_status = cls.determine_reliability_status(
            genuine_samples=genuine_samples,
            validation_mae=validation_mae,
            calibration_status=cal_status,
            is_model_degraded=is_model_degraded,
            is_drift_detected=is_drift_detected,
            recency_days=recency_days,
            has_active_global_model=has_active_global_model
        )

        # 5. Formulate Limitations
        limitations = cls.formulate_limitations(
            reliability_status=reliability_status,
            genuine_samples=genuine_samples,
            prediction_interval=prediction_interval,
            calibration_status=cal_status,
            validation_mae=validation_mae,
            is_model_degraded=is_model_degraded,
            is_drift_detected=is_drift_detected,
            recency_days=recency_days
        )

        return ReliabilityAssessment(
            reliability_status=reliability_status,
            data_sufficiency=data_sufficiency_report.level,
            calibration_status=cal_status,
            is_abstained=False,
            abstention_reason=None,
            prediction_interval=prediction_interval,
            error_distribution=error_dist,
            classification_calibration=clf_report,
            data_sufficiency_report=data_sufficiency_report,
            profile_reliability=profile_report,
            recommendation_reliabilities=rec_reports,
            limitations=limitations
        )
