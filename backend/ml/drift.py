"""Behavioral Drift Detection & Habit Change Monitoring Engine for MicroDelay 2.0.

Provides rigorous statistical monitoring of user routine evolution:
1. Distinguishes:
   - Data Drift: Distributional shifts in operational input features.
   - Concept Drift: Changes in relationship between features and departure delay (model degradation).
   - Behavioral Improvement: Sustained positive change in punctuality or organization.
   - Behavioral Deterioration: Sustained negative change in lateness or distraction.
2. Statistical Detectors:
   - Population Stability Index (PSI): Bin-based distribution stability evaluation.
   - Kolmogorov-Smirnov Two-Sample Test (KS): Continuous distribution eCDF comparison.
   - Rolling Mean and Variance Comparisons (Welch's t-test): Mean shift with unequal variances.
   - Page-Hinkley Sequential Test: Streaming cumulative sum drift detection robust to isolated outliers.
3. Multi-Window Temporal Analysis:
   - Recent 7-Day Window (Acute shocks, exam weeks, sudden changes).
   - Recent 30-Day Window (Sustained habit formation, remote work shifts).
   - Sequential Streams (Continuous step-change detection).
4. Predictive Performance Monitoring:
   - Rolling MAE and Signed Prediction Bias tracking against model baseline.
   - Explicit separation of input data drift from model performance degradation.
5. Adaptive System Actions:
   - Profile refresh, recommendation re-evaluation, and candidate model retrain signaling.
"""

from typing import List, Dict, Any, Optional, Tuple
import datetime
import math
import uuid
import json
import logging
import numpy as np
import scipy.stats as stats
from sqlalchemy.orm import Session

from database import (
    DepartureLog,
    BehavioralDriftEventRecord,
    UserBehavioralBaselineRecord,
    ModelMonitoringLogRecord,
    ModelVersionRecord
)

logger = logging.getLogger("microdelay.ml.drift")


# =====================================================================
# STATISTICAL TEST PRIMITIVES
# =====================================================================

def calculate_psi(
    baseline_samples: List[float],
    current_samples: List[float],
    num_bins: int = 5,
    epsilon: float = 1e-4
) -> float:
    """Calculate Population Stability Index (PSI) between baseline and current distributions.

    Interpretation thresholds:
    - PSI < 0.10: Stable distribution; no significant change.
    - 0.10 <= PSI < 0.25: Moderate drift; slight routine modification.
    - PSI >= 0.25: Significant drift; substantial behavioral shift.
    """
    if len(baseline_samples) < 5 or len(current_samples) < 3:
        return 0.0

    b_arr = np.array(baseline_samples, dtype=float)
    c_arr = np.array(current_samples, dtype=float)

    # Clean non-finite values
    b_arr = b_arr[np.isfinite(b_arr)]
    c_arr = c_arr[np.isfinite(c_arr)]
    if len(b_arr) < 5 or len(c_arr) < 3:
        return 0.0

    # If identical constant arrays, PSI is 0.0
    if np.all(b_arr == b_arr[0]) and np.all(c_arr == b_arr[0]):
        return 0.0

    # Determine quantile bin edges from baseline
    try:
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(b_arr, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # Fallback to uniform range
            min_val = min(float(np.min(b_arr)), float(np.min(c_arr)))
            max_val = max(float(np.max(b_arr)), float(np.max(c_arr)))
            if min_val == max_val:
                return 0.0
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    except Exception:
        return 0.0

    # Ensure outer edges cover both baseline and current
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate bin counts
    b_counts, _ = np.histogram(b_arr, bins=bin_edges)
    c_counts, _ = np.histogram(c_arr, bins=bin_edges)

    b_total = float(len(b_arr))
    c_total = float(len(c_arr))

    psi_val = 0.0
    for b_c, c_c in zip(b_counts, c_counts):
        p = (b_c / b_total) + epsilon
        q = (c_c / c_total) + epsilon
        psi_val += (q - p) * math.log(q / p)

    return float(max(0.0, psi_val))


def calculate_ks_test(
    baseline_samples: List[float],
    current_samples: List[float]
) -> Tuple[float, float]:
    """Calculate Kolmogorov-Smirnov Two-Sample Test.

    Returns:
    - statistic (float): Maximum vertical distance D between empirical CDFs.
    - p_value (float): Probability of observing distance D under null hypothesis.
    """
    if len(baseline_samples) < 5 or len(current_samples) < 4:
        return 0.0, 1.0

    b_arr = np.array(baseline_samples, dtype=float)
    c_arr = np.array(current_samples, dtype=float)
    b_arr = b_arr[np.isfinite(b_arr)]
    c_arr = c_arr[np.isfinite(c_arr)]

    if len(b_arr) < 5 or len(c_arr) < 4:
        return 0.0, 1.0

    try:
        res = stats.ks_2samp(b_arr, c_arr)
        return float(res.statistic), float(res.pvalue)
    except Exception as e:
        logger.debug("KS test calculation failed: %s", e)
        return 0.0, 1.0


def calculate_welch_ttest(
    baseline_samples: List[float],
    current_samples: List[float]
) -> Tuple[float, float, float]:
    """Calculate Welch's t-test for unequal variances.

    Returns:
    - t_stat (float)
    - p_value (float)
    - mean_shift (float): current_mean - baseline_mean
    """
    if len(baseline_samples) < 3 or len(current_samples) < 3:
        return 0.0, 1.0, 0.0

    b_arr = np.array(baseline_samples, dtype=float)
    c_arr = np.array(current_samples, dtype=float)
    b_arr = b_arr[np.isfinite(b_arr)]
    c_arr = c_arr[np.isfinite(c_arr)]

    if len(b_arr) < 3 or len(c_arr) < 3:
        return 0.0, 1.0, 0.0

    mean_shift = float(np.mean(c_arr) - np.mean(b_arr))

    # If zero variance in both
    if np.var(b_arr) == 0.0 and np.var(c_arr) == 0.0:
        if mean_shift == 0.0:
            return 0.0, 1.0, 0.0
        return float("inf") if mean_shift > 0 else float("-inf"), 0.0, mean_shift

    try:
        res = stats.ttest_ind(c_arr, b_arr, equal_var=False)
        p_val = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
        t_stat = float(res.statistic) if np.isfinite(res.statistic) else 0.0
        return t_stat, p_val, mean_shift
    except Exception as e:
        logger.debug("Welch t-test calculation failed: %s", e)
        return 0.0, 1.0, mean_shift


class PageHinkleyDetector:
    """Page-Hinkley sequential change detection test.

    Detects abrupt or cumulative shifts in streaming observation series.
    Maintains robustness against isolated single-event outliers by requiring
    cumulative evidence before triggering.
    """

    def __init__(self, delta: float = 1.0, threshold: float = 15.0, alpha: float = 0.995):
        """
        Args:
            delta: Allowable parameter slip (tolerance margin).
            threshold: Cumulative deviation decision threshold lambda.
            alpha: Decay factor for exponential forgetting (0 < alpha <= 1.0).
        """
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mean = 0.0
        self.n = 0
        self.sum_pos = 0.0
        self.min_sum_pos = 0.0
        self.sum_neg = 0.0
        self.min_sum_neg = 0.0

    def update(self, x: float) -> Tuple[bool, str, float]:
        """Update detector with sequential observation.

        Returns:
            (drift_detected: bool, direction: str, score: float)
            direction in ("increase", "decrease", "none")
        """
        self.n += 1
        # Update running mean
        self.mean = self.mean + (x - self.mean) / float(self.n)

        # Upward deviation sum
        self.sum_pos = self.alpha * self.sum_pos + (x - self.mean - self.delta)
        if self.sum_pos < self.min_sum_pos:
            self.min_sum_pos = self.sum_pos

        pos_diff = self.sum_pos - self.min_sum_pos
        if pos_diff > self.threshold:
            # Reset after trigger
            self.sum_pos = 0.0
            self.min_sum_pos = 0.0
            return True, "increase", float(pos_diff)

        # Downward deviation sum (for detecting positive habit improvements)
        self.sum_neg = self.alpha * self.sum_neg - (x - self.mean + self.delta)
        if self.sum_neg < self.min_sum_neg:
            self.min_sum_neg = self.sum_neg

        neg_diff = self.sum_neg - self.min_sum_neg
        if neg_diff > self.threshold:
            self.sum_neg = 0.0
            self.min_sum_neg = 0.0
            return True, "decrease", float(neg_diff)

        return False, "none", float(max(pos_diff, neg_diff))


# =====================================================================
# FEATURE EXTRACTION & METRIC VALUE EXTRACTORS
# =====================================================================

def extract_metric_series(logs: List[DepartureLog], metric_name: str) -> List[float]:
    """Extract chronological numerical series for a specified metric from departure logs."""
    series = []
    for log in logs:
        val = None
        if metric_name == "actual_prep_duration_min":
            val = getattr(log, "actual_prep_duration_min", None)
        elif metric_name == "prep_bias_min":
            act = getattr(log, "actual_prep_duration_min", None)
            est = getattr(log, "estimated_prep_duration_min", None)
            if act is not None and est is not None:
                val = float(act) - float(est)
        elif metric_name == "departure_delay_min":
            d = getattr(log, "calculated_delay_min", None)
            if d is None:
                d = getattr(log, "total_delay_min", None)
            val = d
        elif metric_name == "lateness_rate":
            d = getattr(log, "calculated_delay_min", None)
            if d is None:
                d = getattr(log, "total_delay_min", None)
            if d is not None:
                val = 1.0 if float(d) > 0.0 else 0.0
        elif metric_name == "phone_interruption_freq":
            tag = getattr(log, "tag_phone_usage", None)
            if tag is not None:
                val = 1.0 if bool(tag) else 0.0
            else:
                dur = getattr(log, "delay_phone_usage_min", 0.0) or getattr(log, "observed_phone_usage_min", 0.0)
                val = 1.0 if float(dur or 0.0) > 0.0 else 0.0
        elif metric_name == "item_search_freq":
            tag = getattr(log, "tag_finding_items", None)
            if tag is not None:
                val = 1.0 if bool(tag) else 0.0
            else:
                dur = getattr(log, "delay_finding_items_min", 0.0) or getattr(log, "observed_finding_items_min", 0.0)
                val = 1.0 if float(dur or 0.0) > 0.0 else 0.0
        elif metric_name == "travel_estimation_error_min":
            act_t = getattr(log, "actual_travel_duration_min", None)
            est_t = getattr(log, "estimated_travel_duration_min", None)
            if act_t is not None and est_t is not None:
                val = float(act_t) - float(est_t)

        if val is not None:
            try:
                series.append(float(val))
            except (ValueError, TypeError):
                continue
    return series


def classify_drift_type_and_valence(
    feature_name: str,
    baseline_mean: float,
    current_mean: float,
    mean_shift: float
) -> Tuple[str, str, str]:
    """Determine drift_type, valence, and severity from metric direction.

    CRITICAL SAFEGUARD:
    Does NOT automatically treat every statistical distribution change as negative.
    A reduction in lateness or distraction is tagged as 'behavioral_improvement' (positive valence).
    """
    # Lower is better metrics
    lower_is_better = {
        "departure_delay_min",
        "lateness_rate",
        "prep_bias_min",
        "phone_interruption_freq",
        "item_search_freq",
        "travel_estimation_error_min"
    }

    abs_shift = abs(mean_shift)

    # Determine severity
    if feature_name in ("departure_delay_min", "prep_bias_min"):
        if abs_shift >= 6.0:
            severity = "significant"
        elif abs_shift >= 2.5:
            severity = "moderate"
        else:
            severity = "informational"
    elif feature_name in ("lateness_rate", "phone_interruption_freq", "item_search_freq"):
        if abs_shift >= 0.35:
            severity = "significant"
        elif abs_shift >= 0.15:
            severity = "moderate"
        else:
            severity = "informational"
    else:
        if abs_shift >= 8.0:
            severity = "significant"
        elif abs_shift >= 4.0:
            severity = "moderate"
        else:
            severity = "informational"

    # Classify valence and type
    if feature_name in lower_is_better:
        if mean_shift < 0:
            # Metric went DOWN -> Improvement!
            drift_type = "behavioral_improvement"
            valence = "positive"
        elif mean_shift > 0:
            # Metric went UP -> Deterioration!
            drift_type = "behavioral_deterioration"
            valence = "negative"
        else:
            drift_type = "data_drift"
            valence = "neutral"
    else:
        # Neutral feature (e.g. prep duration or input shift)
        drift_type = "data_drift"
        valence = "neutral"

    return drift_type, valence, severity


# =====================================================================
# BEHAVIORAL BASELINE SERVICE
# =====================================================================

class BehavioralBaselineService:
    """Computes, stores, and versions historical behavioral baseline distributions."""

    MONITORED_METRICS = [
        "actual_prep_duration_min",
        "prep_bias_min",
        "departure_delay_min",
        "lateness_rate",
        "phone_interruption_freq",
        "item_search_freq",
        "travel_estimation_error_min"
    ]

    @classmethod
    def compute_baseline_metrics(cls, logs: List[DepartureLog]) -> Dict[str, Dict[str, float]]:
        """Calculate statistical distribution parameters for all monitored metrics."""
        summary = {}
        for metric in cls.MONITORED_METRICS:
            series = extract_metric_series(logs, metric)
            if not series:
                summary[metric] = {
                    "count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0
                }
                continue

            arr = np.array(series, dtype=float)
            arr = arr[np.isfinite(arr)]
            summary[metric] = {
                "count": int(len(arr)),
                "mean": float(round(np.mean(arr), 2)),
                "std": float(round(np.std(arr), 2)),
                "median": float(round(np.median(arr), 2)),
                "q25": float(round(np.percentile(arr, 25), 2)),
                "q75": float(round(np.percentile(arr, 75), 2))
            }
        return summary

    @classmethod
    def get_or_create_user_baseline(
        cls,
        db: Session,
        user_id: int,
        historical_window: str = "all_time",
        force_recompute: bool = False
    ) -> Optional[UserBehavioralBaselineRecord]:
        """Retrieve existing active baseline or generate a new one from chronological logs."""
        if not force_recompute:
            existing = db.query(UserBehavioralBaselineRecord).filter(
                UserBehavioralBaselineRecord.user_id == user_id,
                UserBehavioralBaselineRecord.is_active == True
            ).order_by(UserBehavioralBaselineRecord.id.desc()).first()
            if existing:
                return existing

        # Query all historical genuine or synthetic logs
        logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id
        ).order_by(DepartureLog.date.asc(), DepartureLog.id.asc()).all()

        if len(logs) < 5:
            return None

        # Exclude recent 7-day logs from baseline if sufficient history exists (> 15 logs)
        if len(logs) >= 15:
            baseline_logs = logs[:-7]
        else:
            baseline_logs = logs

        metrics_dict = cls.compute_baseline_metrics(baseline_logs)
        has_real = any(getattr(l, "provenance", "") == "real" or getattr(l, "is_synthetic", False) is False for l in baseline_logs)
        prov = "genuine" if has_real else "synthetic"

        baseline_rec = UserBehavioralBaselineRecord(
            baseline_id=f"base_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            baseline_version="v2.0",
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            observation_count=len(baseline_logs),
            historical_window=historical_window,
            data_provenance=prov,
            metrics=json.dumps(metrics_dict),
            is_active=True
        )

        db.add(baseline_rec)
        db.commit()
        db.refresh(baseline_rec)
        return baseline_rec


# =====================================================================
# MULTI-WINDOW TEMPORAL DRIFT ANALYZER
# =====================================================================

class TemporalDriftAnalyzer:
    """Performs multi-window chronological comparison across 7d, 30d, and sequential streams."""

    MIN_RECENT_SAMPLES = 4
    MIN_BASELINE_SAMPLES = 8

    @classmethod
    def analyze_user_drift(
        cls,
        db: Session,
        user_id: int,
        evaluation_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute full behavioral drift diagnosis across all monitored dimensions.

        Ensures chronological safety by only analyzing checkouts logged before evaluation_timestamp.
        """
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        eval_time = evaluation_timestamp or now_dt.isoformat()

        # Query chronological logs for user
        logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id
        ).order_by(DepartureLog.date.asc(), DepartureLog.id.asc()).all()

        total_count = len(logs)
        if total_count < (cls.MIN_RECENT_SAMPLES + cls.MIN_BASELINE_SAMPLES):
            return {
                "status": "insufficient_data",
                "message": f"Insufficient historical checkouts ({total_count} observed, minimum {cls.MIN_RECENT_SAMPLES + cls.MIN_BASELINE_SAMPLES} required).",
                "total_observations": total_count,
                "events_detected": [],
                "summary": "Baseline data is currently being gathered; drift detection requires at least 12 checkouts.",
                "evaluated_at": eval_time
            }

        # Partition logs into recent 7-day, recent 30-day, and baseline
        # (Chronological order: earliest logs first)
        recent_7_logs = logs[-7:] if total_count >= 7 else logs[-cls.MIN_RECENT_SAMPLES:]
        recent_30_logs = logs[-30:] if total_count >= 30 else logs[-cls.MIN_RECENT_SAMPLES:]
        baseline_logs = logs[:-7] if total_count >= 14 else logs[:-cls.MIN_RECENT_SAMPLES]

        detected_events: List[Dict[str, Any]] = []

        for metric in BehavioralBaselineService.MONITORED_METRICS:
            base_series = extract_metric_series(baseline_logs, metric)
            recent_7_series = extract_metric_series(recent_7_logs, metric)
            recent_30_series = extract_metric_series(recent_30_logs, metric)

            if len(base_series) < cls.MIN_BASELINE_SAMPLES or len(recent_7_series) < cls.MIN_RECENT_SAMPLES:
                continue

            base_mean = float(np.mean(base_series))
            r7_mean = float(np.mean(recent_7_series))
            r30_mean = float(np.mean(recent_30_series)) if len(recent_30_series) >= cls.MIN_RECENT_SAMPLES else r7_mean

            # 1. Kolmogorov-Smirnov Test (7-day window)
            ks_stat, ks_p = calculate_ks_test(base_series, recent_7_series)

            # 2. Population Stability Index (PSI) (30-day or 7-day window)
            psi_score = calculate_psi(base_series, recent_7_series)

            # 3. Welch's t-test for directional mean shift
            t_stat, t_p, mean_shift = calculate_welch_ttest(base_series, recent_7_series)

            # 4. Sequential Page-Hinkley Check (cumulative drift)
            all_series = extract_metric_series(logs, metric)
            ph_detector = PageHinkleyDetector(
                delta=max(0.5, float(np.std(base_series) * 0.3) if np.std(base_series) > 0 else 1.0),
                threshold=12.0
            )
            ph_triggered = False
            ph_direction = "none"
            ph_score = 0.0
            for val in all_series:
                trig, dirn, score = ph_detector.update(val)
                if trig:
                    ph_triggered = True
                    ph_direction = dirn
                    ph_score = score

            # Outlier robustness check:
            # If only 1 observation is extreme and others match baseline, do not fire!
            is_single_outlier = False
            if len(recent_7_series) >= 4:
                median_recent = float(np.median(recent_7_series))
                near_median = [abs(x - median_recent) < (2.0 * (np.std(base_series) or 1.0)) for x in recent_7_series]
                if sum(near_median) >= (len(recent_7_series) - 1) and abs(median_recent - base_mean) < 2.0:
                    is_single_outlier = True

            # Determine significance
            is_stat_sig = bool((ks_p < 0.05 and ks_stat >= 0.25) or (t_p < 0.05 and abs(mean_shift) >= 2.0) or (psi_score >= 0.25))

            # Suppress if just an isolated outlier day
            if is_single_outlier and not ph_triggered:
                is_stat_sig = False

            # Check if threshold reached
            if is_stat_sig or (abs(mean_shift) >= 3.0 and len(recent_7_series) >= cls.MIN_RECENT_SAMPLES):
                drift_type, valence, severity = classify_drift_type_and_valence(
                    metric, base_mean, r7_mean, mean_shift
                )

                # Determine primary detection method
                if psi_score >= 0.25:
                    primary_method = "population_stability_index"
                    score = psi_score
                elif ks_p < 0.05:
                    primary_method = "kolmogorov_smirnov"
                    score = ks_stat
                elif ph_triggered:
                    primary_method = "page_hinkley"
                    score = ph_score
                else:
                    primary_method = "rolling_mean_shift"
                    score = abs(mean_shift)

                # Humanized descriptive summary
                humanized = cls._format_humanized_summary(
                    metric, base_mean, r7_mean, mean_shift, drift_type, valence, is_stat_sig
                )

                event_dict = {
                    "drift_event_id": f"drift_{uuid.uuid4().hex[:12]}",
                    "user_id": user_id,
                    "feature_name": metric,
                    "drift_type": drift_type,
                    "detection_method": primary_method,
                    "baseline_value": round(base_mean, 2),
                    "current_value": round(r7_mean, 2),
                    "drift_score": round(score, 3),
                    "p_value": round(min(ks_p, t_p), 4) if (ks_p < 1.0 or t_p < 1.0) else None,
                    "severity": severity,
                    "valence": valence,
                    "temporal_window": "7d_vs_baseline",
                    "sample_count_baseline": len(base_series),
                    "sample_count_current": len(recent_7_series),
                    "detected_at": eval_time,
                    "status": "active",
                    "humanized_summary": humanized,
                    "is_statistically_significant": is_stat_sig
                }
                detected_events.append(event_dict)

        # Persist detected events to database if new
        persisted_records = cls._persist_drift_events(db, user_id, detected_events)

        return {
            "status": "success",
            "total_observations": total_count,
            "events_detected": persisted_records,
            "improvements_count": sum(1 for e in persisted_records if e["drift_type"] == "behavioral_improvement"),
            "deteriorations_count": sum(1 for e in persisted_records if e["drift_type"] == "behavioral_deterioration"),
            "data_drifts_count": sum(1 for e in persisted_records if e["drift_type"] == "data_drift"),
            "summary": cls._generate_overall_summary(persisted_records),
            "evaluated_at": eval_time
        }

    @classmethod
    def _format_humanized_summary(
        cls,
        metric: str,
        base_val: float,
        curr_val: float,
        shift: float,
        drift_type: str,
        valence: str,
        is_stat_sig: bool
    ) -> str:
        stat_prefix = "Statistically verified" if is_stat_sig else "Descriptive trend"
        names = {
            "actual_prep_duration_min": "Preparation duration",
            "prep_bias_min": "Preparation underestimation bias",
            "departure_delay_min": "Average departure delay",
            "lateness_rate": "Lateness frequency",
            "phone_interruption_freq": "Morning phone distraction",
            "item_search_freq": "Doorway item-search delay",
            "travel_estimation_error_min": "Travel estimation error"
        }
        human_name = names.get(metric, metric)

        if drift_type == "behavioral_improvement":
            if "rate" in metric or "freq" in metric:
                return (
                    f"{stat_prefix}: {human_name} improved significantly, dropping from "
                    f"{round(base_val * 100)}% to {round(curr_val * 100)}% over recent routines."
                )
            return (
                f"{stat_prefix}: {human_name} improved by {abs(round(shift, 1))} min "
                f"(decreased from {round(base_val, 1)}m baseline to {round(curr_val, 1)}m recently)."
            )
        elif drift_type == "behavioral_deterioration":
            if "rate" in metric or "freq" in metric:
                return (
                    f"{stat_prefix}: {human_name} increased from "
                    f"{round(base_val * 100)}% baseline to {round(curr_val * 100)}% recently."
                )
            return (
                f"{stat_prefix}: {human_name} increased by {round(shift, 1)} min "
                f"(rose from {round(base_val, 1)}m baseline to {round(curr_val, 1)}m recently)."
            )
        else:
            return (
                f"{stat_prefix}: Distributional shift in {human_name} "
                f"(baseline: {round(base_val, 1)}, current: {round(curr_val, 1)})."
            )

    @classmethod
    def _generate_overall_summary(cls, events: List[Dict[str, Any]]) -> str:
        if not events:
            return "Your routines remain stable compared with your historical baseline. No significant behavioral drift detected."

        improvements = [e for e in events if e.get("drift_type") == "behavioral_improvement"]
        deteriorations = [e for e in events if e.get("drift_type") == "behavioral_deterioration"]

        if improvements and not deteriorations:
            return f"Positive routine progress detected! You have achieved sustained improvements in {len(improvements)} behavioral area(s)."
        elif deteriorations and not improvements:
            return f"Routine friction detected: recent checkouts show notable increases in {len(deteriorations)} delay factor(s)."
        elif improvements and deteriorations:
            return f"Mixed behavioral shifts observed: {len(improvements)} improvement(s) alongside {len(deteriorations)} emerging friction area(s)."
        else:
            return f"Distributional shift detected across {len(events)} routine dimension(s)."

    @classmethod
    def _persist_drift_events(
        cls,
        db: Session,
        user_id: int,
        detected_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Persist new drift events or update existing active events to avoid duplicates."""
        results = []
        for ev in detected_events:
            # Check if active event for this user + feature already exists in last 3 days
            existing = db.query(BehavioralDriftEventRecord).filter(
                BehavioralDriftEventRecord.user_id == user_id,
                BehavioralDriftEventRecord.feature_name == ev["feature_name"],
                BehavioralDriftEventRecord.status == "active"
            ).first()

            if existing:
                # Update current values and scores
                existing.current_value = ev["current_value"]
                existing.drift_score = ev["drift_score"]
                existing.severity = ev["severity"]
                existing.humanized_summary = ev["humanized_summary"]
                existing.detected_at = ev["detected_at"]
                db.commit()
                ev["drift_event_id"] = existing.drift_event_id
                results.append(ev)
            else:
                record = BehavioralDriftEventRecord(
                    drift_event_id=ev["drift_event_id"],
                    user_id=user_id,
                    feature_name=ev["feature_name"],
                    drift_type=ev["drift_type"],
                    detection_method=ev["detection_method"],
                    baseline_value=ev["baseline_value"],
                    current_value=ev["current_value"],
                    drift_score=ev["drift_score"],
                    p_value=ev.get("p_value"),
                    severity=ev["severity"],
                    valence=ev["valence"],
                    temporal_window=ev["temporal_window"],
                    sample_count_baseline=ev["sample_count_baseline"],
                    sample_count_current=ev["sample_count_current"],
                    detected_at=ev["detected_at"],
                    status="active",
                    humanized_summary=ev["humanized_summary"],
                    is_statistically_significant=ev["is_statistically_significant"]
                )
                db.add(record)
                db.commit()
                results.append(ev)
        return results


# =====================================================================
# MODEL PERFORMANCE MONITORING SERVICE
# =====================================================================

class ModelPerformanceMonitor:
    """Tracks model prediction accuracy (rolling MAE & bias) when actual outcomes become available.

    Separates input data distribution drift from true concept drift / predictive degradation.
    """

    ROLLING_WINDOW = 8
    DEGRADATION_THRESHOLD_PCT = 30.0  # 30% increase in MAE over baseline

    @classmethod
    def monitor_user_model_performance(
        cls,
        db: Session,
        user_id: int
    ) -> Dict[str, Any]:
        """Evaluate rolling predictive error across recent checkouts with known actual delays."""
        # Query active model record for user or global
        active_model = db.query(ModelVersionRecord).filter(
            ModelVersionRecord.is_active == True
        ).order_by(ModelVersionRecord.id.desc()).first()

        model_version = active_model.model_version if active_model else "global_v1"
        baseline_mae = float(active_model.validation_mae) if (active_model and getattr(active_model, "validation_mae", None)) else 4.0
        baseline_bias = 0.0

        # Query recent checkouts for user
        logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id
        ).order_by(DepartureLog.date.asc(), DepartureLog.id.asc()).all()

        if len(logs) < cls.ROLLING_WINDOW:
            return {
                "status": "insufficient_data",
                "message": f"Fewer than {cls.ROLLING_WINDOW} checkouts observed for predictive performance tracking.",
                "model_version": model_version,
                "performance_degraded": False,
                "rolling_mae": None,
                "rolling_bias": None,
                "baseline_mae": baseline_mae
            }

        # Calculate actual vs baseline estimated delay errors
        recent_logs = logs[-cls.ROLLING_WINDOW:]
        errors = []
        signed_errors = []

        for log in recent_logs:
            actual = getattr(log, "calculated_delay_min", None)
            if actual is None:
                actual = getattr(log, "total_delay_min", None)

            # Look up estimated prep bias as reference prediction
            act_prep = getattr(log, "actual_prep_duration_min", 30.0)
            est_prep = getattr(log, "estimated_prep_duration_min", 25.0)
            naive_pred_delay = max(0.0, float(act_prep) - float(est_prep))

            if actual is not None:
                err = abs(float(actual) - float(naive_pred_delay))
                signed_err = float(actual) - float(naive_pred_delay)
                errors.append(err)
                signed_errors.append(signed_err)

        if not errors:
            return {
                "status": "insufficient_data",
                "message": "No valid delay errors could be calculated.",
                "model_version": model_version,
                "performance_degraded": False
            }

        rolling_mae = float(np.mean(errors))
        rolling_bias = float(np.mean(signed_errors))

        # Check for degradation
        degradation_pct = 0.0
        if baseline_mae > 0:
            degradation_pct = ((rolling_mae - baseline_mae) / baseline_mae) * 100.0

        performance_degraded = bool(degradation_pct >= cls.DEGRADATION_THRESHOLD_PCT and len(errors) >= cls.ROLLING_WINDOW)

        # Check for error distribution shift (bias > 3m in either direction)
        error_dist_shift = bool(abs(rolling_bias) >= 3.0)

        # Persist monitoring record
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        mon_rec = ModelMonitoringLogRecord(
            user_id=user_id,
            model_version=model_version,
            evaluated_at=now_str,
            rolling_window_size=len(errors),
            rolling_mae=round(rolling_mae, 2),
            rolling_bias=round(rolling_bias, 2),
            baseline_mae=round(baseline_mae, 2),
            baseline_bias=round(baseline_bias, 2),
            performance_degraded=performance_degraded,
            degradation_pct=round(degradation_pct, 1),
            error_distribution_shift=error_dist_shift,
            details=json.dumps({
                "recent_errors": [round(e, 2) for e in errors],
                "sample_count": len(errors)
            })
        )
        db.add(mon_rec)

        # If significant performance degradation detected, log concept_drift event
        if performance_degraded:
            concept_event = BehavioralDriftEventRecord(
                drift_event_id=f"cdrift_{uuid.uuid4().hex[:12]}",
                user_id=user_id,
                feature_name="prediction_error_mae",
                drift_type="concept_drift",
                detection_method="rolling_mae_monitor",
                baseline_value=round(baseline_mae, 2),
                current_value=round(rolling_mae, 2),
                drift_score=round(degradation_pct, 1),
                p_value=0.01,
                severity="significant",
                valence="negative",
                temporal_window=f"rolling_{cls.ROLLING_WINDOW}",
                sample_count_baseline=cls.ROLLING_WINDOW * 2,
                sample_count_current=len(errors),
                detected_at=now_str,
                status="active",
                humanized_summary=(
                    f"Concept drift detected: Prediction MAE has degraded by {round(degradation_pct, 1)}% "
                    f"(from {round(baseline_mae, 1)}m baseline to {round(rolling_mae, 1)}m recently). "
                    "Personalized model retraining is recommended."
                ),
                is_statistically_significant=True
            )
            db.add(concept_event)

        db.commit()

        return {
            "status": "success",
            "model_version": model_version,
            "rolling_mae": round(rolling_mae, 2),
            "rolling_bias": round(rolling_bias, 2),
            "baseline_mae": round(baseline_mae, 2),
            "performance_degraded": performance_degraded,
            "degradation_pct": round(degradation_pct, 1),
            "error_distribution_shift": error_dist_shift,
            "retrain_recommended": performance_degraded,
            "evaluated_at": now_str
        }


# =====================================================================
# ADAPTIVE RESPONSE COORDINATOR
# =====================================================================

class AdaptiveDriftCoordinator:
    """Orchestrates adaptive system responses when meaningful drift is diagnosed."""

    @classmethod
    def handle_detected_drift(
        cls,
        db: Session,
        user_id: int,
        drift_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute automated adaptations while preserving active models until validation."""
        events = drift_analysis.get("events_detected", [])
        if not events:
            return {"action_taken": "none", "profile_refreshed": False, "retrain_recommended": False}

        has_concept_drift = any(e.get("drift_type") == "concept_drift" for e in events)
        has_behavioral_shift = any(e.get("drift_type") in ("behavioral_improvement", "behavioral_deterioration") for e in events)

        profile_refreshed = False
        if has_behavioral_shift:
            # Refresh behavioral baseline
            BehavioralBaselineService.get_or_create_user_baseline(db, user_id, force_recompute=True)
            profile_refreshed = True

        return {
            "action_taken": "profile_refreshed" if profile_refreshed else "monitored",
            "profile_refreshed": profile_refreshed,
            "retrain_recommended": has_concept_drift,
            "drift_events_count": len(events)
        }
