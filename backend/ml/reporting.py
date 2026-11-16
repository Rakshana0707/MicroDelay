"""Comprehensive Model Evaluation, Residual Analysis, and Reporting for MicroDelay.

This module standardizes evaluation metrics, residual diagnostics, performance segmentation,
and human/machine-readable reporting across real and synthetic datasets.

Key Capabilities:
1. Standardized Metrics: MAE, RMSE, R² (with strict statistical validity guardrails),
   Median Absolute Error, and tolerance metrics (within ±5m and ±10m).
2. Performance Segmentation: Dissects error distributions across days, events,
   importance, bag preparation, prep duration buckets, and user personas.
3. Residual Analysis: Diagnoses over/underprediction bias, skewness, and extreme errors.
4. Model Evaluation State Machine:
   - 'insufficient_data'
   - 'training_completed'
   - 'validation_completed'
   - 'evaluation_completed'
   - 'model_underperforms_baseline'
   - 'model_approved'
5. Separate reporting directories for real vs. synthetic records in backend/reports/.
"""

import os
import json
import datetime
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
REAL_REPORTS_DIR = os.path.join(REPORTS_DIR, "real")
SYNTHETIC_REPORTS_DIR = os.path.join(REPORTS_DIR, "synthetic")

os.makedirs(REAL_REPORTS_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_REPORTS_DIR, exist_ok=True)


class ModelEvaluationStatus:
    INSUFFICIENT_DATA = "insufficient_data"
    TRAINING_COMPLETED = "training_completed"
    VALIDATION_COMPLETED = "validation_completed"
    EVALUATION_COMPLETED = "evaluation_completed"
    MODEL_UNDERPERFORMS_BASELINE = "model_underperforms_baseline"
    MODEL_APPROVED = "model_approved"


def calculate_standardized_metrics(
    y_true: Union[List[float], np.ndarray, pd.Series],
    y_pred: Union[List[float], np.ndarray, pd.Series],
    baseline_mae: Optional[float] = None
) -> Dict[str, Any]:
    """Calculate standardized regression metrics with rigorous R² and baseline guardrails."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    n = len(y_t)
    if n == 0:
        return {
            "sample_count": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "r2_meaningful": False,
            "r2_reason": "Zero observations provided.",
            "medae": None,
            "tolerance_5m_pct": 0.0,
            "tolerance_10m_pct": 0.0,
            "baseline_improvement_pct": 0.0
        }

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    medae = float(median_absolute_error(y_t, y_p))

    # Tolerance metrics
    abs_errors = np.abs(y_t - y_p)
    tol_5m = float((abs_errors <= 5.0).sum() / n * 100.0)
    tol_10m = float((abs_errors <= 10.0).sum() / n * 100.0)

    # R² guardrail: Must have >= 5 samples and non-trivial target variance
    target_var = float(np.var(y_t))
    r2_val = None
    r2_meaningful = False
    r2_reason = "Valid"

    if n < 5:
        r2_reason = f"Sample count too small ({n} < 5) for statistically defensible R²."
    elif target_var <= 1e-4:
        r2_reason = f"Target total_delay_min has zero or near-zero variance ({target_var:.6f}); R² undefined or meaningless."
    else:
        try:
            raw_r2 = float(r2_score(y_t, y_p))
            # Bound extremely negative R² values to avoid misleading distortions
            r2_val = round(raw_r2, 4)
            r2_meaningful = True
        except Exception as e:
            r2_reason = f"R² calculation error: {str(e)}"

    # Baseline improvement formula: ((Baseline MAE - Model MAE) / Baseline MAE) * 100
    improvement_pct = 0.0
    if baseline_mae is not None and baseline_mae > 0.0:
        improvement_pct = round(float((baseline_mae - mae) / baseline_mae * 100.0), 2)

    return {
        "sample_count": n,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": r2_val,
        "r2_meaningful": r2_meaningful,
        "r2_reason": r2_reason,
        "medae": round(medae, 3),
        "tolerance_5m_pct": round(tol_5m, 2),
        "tolerance_10m_pct": round(tol_10m, 2),
        "baseline_improvement_pct": improvement_pct
    }


def perform_residual_analysis(
    y_true: Union[List[float], np.ndarray, pd.Series],
    y_pred: Union[List[float], np.ndarray, pd.Series],
    df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Perform rigorous residual analysis: over/underprediction, bias, distributions, and outliers."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    n = len(y_t)
    if n == 0:
        return {"error": "No observations provided for residual analysis."}

    # Residual e = y_pred - y_true: positive = overpredicted delay, negative = underpredicted delay
    residuals = y_p - y_t
    abs_errors = np.abs(residuals)

    mean_bias = float(np.mean(residuals))
    std_residuals = float(np.std(residuals))

    # Over/under predictions with neutral threshold +/- 0.5 minutes
    overpredicted_mask = residuals > 0.5
    underpredicted_mask = residuals < -0.5
    exact_mask = abs_errors <= 0.5

    over_count = int(overpredicted_mask.sum())
    under_count = int(underpredicted_mask.sum())
    exact_count = int(exact_mask.sum())

    mean_over = float(np.mean(residuals[overpredicted_mask])) if over_count > 0 else 0.0
    mean_under = float(np.mean(np.abs(residuals[underpredicted_mask]))) if under_count > 0 else 0.0

    # Large prediction errors (|e| > 10.0 minutes)
    large_error_mask = abs_errors > 10.0
    large_error_count = int(large_error_mask.sum())
    large_error_pct = round(float(large_error_count / n * 100.0), 2)

    # Detailed large errors listing
    large_error_details = []
    if large_error_count > 0:
        indices = np.where(large_error_mask)[0]
        for idx in indices[:10]:  # Cap at top 10 examples
            detail = {
                "index": int(idx),
                "actual_delay": round(float(y_t[idx]), 1),
                "predicted_delay": round(float(y_p[idx]), 1),
                "residual": round(float(residuals[idx]), 1),
                "type": "overprediction" if residuals[idx] > 0 else "underprediction"
            }
            if df is not None and idx < len(df):
                row = df.iloc[idx]
                detail["event_name"] = str(row.get("event_name", "Unknown"))
                detail["day_of_week"] = str(row.get("day_of_week", "Unknown"))
                detail["date"] = str(row.get("date", "Unknown"))
            large_error_details.append(detail)

    # Skewness calculation safely
    skewness = 0.0
    if n >= 3 and std_residuals > 1e-4:
        skewness = float(np.mean(((residuals - mean_bias) / std_residuals) ** 3))

    distribution = {
        "min": round(float(np.min(residuals)), 2),
        "q25": round(float(np.percentile(residuals, 25)), 2),
        "median": round(float(np.median(residuals)), 2),
        "q75": round(float(np.percentile(residuals, 75)), 2),
        "max": round(float(np.max(residuals)), 2),
        "std": round(std_residuals, 2),
        "skewness": round(skewness, 3)
    }

    # Bias characterization
    if abs(mean_bias) <= 1.0:
        bias_label = "Unbiased (near zero)"
    elif mean_bias > 1.0:
        bias_label = "Pessimistic bias (tends to systematically overpredict delay)"
    else:
        bias_label = "Optimistic bias (tends to systematically underpredict delay)"

    # Identify poorly performing condition segments
    poor_conditions: List[Dict[str, Any]] = []
    overall_mae = float(np.mean(abs_errors))

    if df is not None and len(df) == n and overall_mae > 0:
        work_df = df.copy()
        work_df["_abs_err"] = abs_errors

        for col in ["event_name", "day_of_week", "importance", "bag_prepared"]:
            if col in work_df.columns:
                grouped = work_df.groupby(col)["_abs_err"].agg(["count", "mean"])
                for cat, stats in grouped.iterrows():
                    # Flag slice if count >= 2 and MAE is > 1.4x overall MAE
                    if stats["count"] >= 2 and stats["mean"] > (1.4 * overall_mae):
                        poor_conditions.append({
                            "segment_dimension": col,
                            "segment_value": str(cat),
                            "sample_count": int(stats["count"]),
                            "segment_mae": round(float(stats["mean"]), 2),
                            "overall_mae": round(overall_mae, 2),
                            "error_inflation_pct": round(float((stats["mean"] - overall_mae) / overall_mae * 100.0), 1)
                        })

    return {
        "mean_signed_bias": round(mean_bias, 3),
        "bias_characterization": bias_label,
        "overpredictions": {
            "count": over_count,
            "percentage": round(float(over_count / n * 100.0), 2),
            "mean_overprediction_min": round(mean_over, 2)
        },
        "underpredictions": {
            "count": under_count,
            "percentage": round(float(under_count / n * 100.0), 2),
            "mean_underprediction_min": round(mean_under, 2)
        },
        "exact_predictions": {
            "count": exact_count,
            "percentage": round(float(exact_count / n * 100.0), 2)
        },
        "large_errors": {
            "threshold_min": 10.0,
            "count": large_error_count,
            "percentage": large_error_pct,
            "sample_cases": large_error_details
        },
        "error_distribution": distribution,
        "poor_performing_conditions": poor_conditions
    }


def segment_performance(
    df: pd.DataFrame,
    y_true: Union[List[float], np.ndarray, pd.Series],
    y_pred: Union[List[float], np.ndarray, pd.Series]
) -> Dict[str, List[Dict[str, Any]]]:
    """Segment prediction accuracy across days, events, importance, bag state, prep duration, and personas."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(df) != len(y_t):
        return {}

    work_df = df.copy()
    work_df["_y_true"] = y_t
    work_df["_y_pred"] = y_p
    work_df["_abs_err"] = np.abs(y_t - y_p)
    work_df["_sq_err"] = (y_t - y_p) ** 2
    work_df["_within_5m"] = (work_df["_abs_err"] <= 5.0).astype(int)

    # Bin preparation duration into meaningful behavioral buckets
    if "estimated_prep_duration_min" in work_df.columns:
        prep_col = work_df["estimated_prep_duration_min"].astype(float)
        conditions = [
            (prep_col < 15.0),
            (prep_col >= 15.0) & (prep_col <= 30.0),
            (prep_col > 30.0) & (prep_col <= 45.0),
            (prep_col > 45.0)
        ]
        choices = ["< 15m", "15 - 30m", "30 - 45m", "> 45m"]
        work_df["prep_duration_bucket"] = np.select(conditions, choices, default="15 - 30m")

    # Dimensions to segment
    dimensions = ["day_of_week", "event_name", "importance", "bag_prepared", "prep_duration_bucket", "profile_type"]
    segmented_results: Dict[str, List[Dict[str, Any]]] = {}

    for dim in dimensions:
        if dim not in work_df.columns:
            continue

        groups = work_df.groupby(dim)
        dim_stats = []

        for val, group in groups:
            count = len(group)
            mae = float(group["_abs_err"].mean())
            rmse = float(np.sqrt(group["_sq_err"].mean()))
            tol_5m = float(group["_within_5m"].mean() * 100.0)

            dim_stats.append({
                "segment": str(val),
                "count": count,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "tolerance_5m_pct": round(tol_5m, 1)
            })

        # Sort descending by count
        dim_stats.sort(key=lambda s: s["count"], reverse=True)
        segmented_results[dim] = dim_stats

    return segmented_results


def determine_model_status(
    train_count: int,
    is_validated: bool,
    baseline_mae: Optional[float],
    model_mae: Optional[float],
    min_samples: int = 8,
    tolerance_5m_pct: float = 0.0,
    large_error_pct: float = 0.0
) -> Tuple[str, str]:
    """Determine standardized evaluation status and explanatory justification."""
    if train_count < min_samples:
        return (
            ModelEvaluationStatus.INSUFFICIENT_DATA,
            f"Insufficient observations ({train_count} < {min_samples}). Personalized evaluation is disabled to avoid misleading metrics."
        )

    if not is_validated or model_mae is None:
        return (
            ModelEvaluationStatus.TRAINING_COMPLETED,
            "Model training finished, but chronological validation has not been completed."
        )

    # Check baseline comparison
    if baseline_mae is not None and model_mae > baseline_mae:
        return (
            ModelEvaluationStatus.MODEL_UNDERPERFORMS_BASELINE,
            f"Model MAE ({model_mae:.2f}m) underperformed historical event baseline MAE ({baseline_mae:.2f}m). Fallback to baseline recommended."
        )

    # High error rate check
    if large_error_pct > 40.0:
        return (
            ModelEvaluationStatus.EVALUATION_COMPLETED,
            f"Model completed validation with MAE {model_mae:.2f}m, but exhibited {large_error_pct:.1f}% large errors (>10m)."
        )

    return (
        ModelEvaluationStatus.MODEL_APPROVED,
        f"Model approved: successfully outperformed baseline (MAE {model_mae:.2f}m vs baseline {baseline_mae:.2f}m) with {tolerance_5m_pct:.1f}% +/-5m accuracy."
    )


class EvaluationReporter:
    """Generates structured JSON and human-readable Markdown evaluation reports."""

    def __init__(self, user_id: int, model_version: str = "2.0.0", provenance: str = "synthetic"):
        self.user_id = user_id
        self.model_version = model_version
        self.provenance = provenance.lower()
        self.target_dir = REAL_REPORTS_DIR if self.provenance == "real" else SYNTHETIC_REPORTS_DIR

    def generate_report(
        self,
        metrics: Dict[str, Any],
        residual_analysis: Dict[str, Any],
        segmentation: Dict[str, Any],
        model_name: str,
        hyperparameters: Dict[str, Any],
        training_metadata: Dict[str, Any],
        evaluation_status: str,
        status_reason: str
    ) -> Tuple[str, str]:
        """Generate and save both JSON and Markdown evaluation artifacts.
        
        Returns:
          (json_filepath, markdown_filepath)
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")

        report_data = {
            "metadata": {
                "user_id": self.user_id,
                "model_version": self.model_version,
                "model_name": model_name,
                "data_provenance": self.provenance,
                "is_demonstration_only": (self.provenance != "real"),
                "generated_at": now.isoformat(),
                "evaluation_status": evaluation_status,
                "status_reason": status_reason,
                "hyperparameters": hyperparameters,
                "training_metadata": training_metadata
            },
            "metrics": metrics,
            "residual_analysis": residual_analysis,
            "segmentation": segmentation
        }

        # 1. Save JSON Report
        filename_base = f"evaluation_user_{self.user_id}_{timestamp_str}"
        json_path = os.path.join(self.target_dir, f"{filename_base}.json")
        latest_json_path = os.path.join(self.target_dir, f"latest_user_{self.user_id}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 2. Save Markdown Report
        md_path = os.path.join(self.target_dir, f"{filename_base}.md")
        markdown_content = self._render_markdown(report_data)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return json_path, md_path

    def _render_markdown(self, data: Dict[str, Any]) -> str:
        """Render human-readable Markdown evaluation report."""
        meta = data["metadata"]
        m = data["metrics"]
        res = data["residual_analysis"]
        seg = data["segmentation"]

        is_demo_badge = "⚠️ SYNTHETIC DEMONSTRATION DATA (NOT REAL PERFORMANCE)" if meta["is_demonstration_only"] else "✅ GENUINE USER OBSERVATIONS"

        r2_display = f"{m['r2']}" if m["r2_meaningful"] else f"N/A ({m['r2_reason']})"

        lines = [
            f"# MicroDelay ML Model Evaluation Report",
            f"",
            f"**Data Provenance Status:** {is_demo_badge}  ",
            f"**User ID:** {meta['user_id']} | **Model Version:** {meta['model_version']}  ",
            f"**Model Architecture:** `{meta['model_name']}`  ",
            f"**Generated:** {meta['generated_at']}  ",
            f"**Status:** `{meta['evaluation_status'].upper()}` — *{meta['status_reason']}*",
            f"",
            f"---",
            f"",
            f"## 1. Executive Metrics Summary",
            f"",
            f"| Metric | Value | Reference / Target |",
            f"| :--- | :--- | :--- |",
            f"| **Mean Absolute Error (MAE)** | **{m['mae']} min** | Lower is better |",
            f"| **Root Mean Squared Error (RMSE)** | {m['rmse']} min | Penalizes large errors |",
            f"| **Median Absolute Error (MedAE)** | {m['medae']} min | 50% of errors fall below this |",
            f"| **Prediction Tolerance (±5 min)** | **{m['tolerance_5m_pct']}%** | Practical usability window |",
            f"| **Prediction Tolerance (±10 min)** | {m['tolerance_10m_pct']}% | High-certainty window |",
            f"| **R² Score (Holdout)** | {r2_display} | Variance explained |",
            f"| **Baseline Improvement** | **{m['baseline_improvement_pct']}%** | vs Historical Event Baseline |",
            f"",
            f"---",
            f"",
            f"## 2. Residual & Bias Analysis",
            f"",
            f"- **Systemic Bias:** {res.get('bias_characterization', 'N/A')} (Mean signed error: `{res.get('mean_signed_bias', 0.0)}m`)",
            f"- **Overpredictions ($\\\\hat{{y}} > y$):** {res.get('overpredictions', {}).get('count', 0)} ({res.get('overpredictions', {}).get('percentage', 0.0)}%), Avg: `{res.get('overpredictions', {}).get('mean_overprediction_min', 0.0)}m`",
            f"- **Underpredictions ($\\\\hat{{y}} < y$):** {res.get('underpredictions', {}).get('count', 0)} ({res.get('underpredictions', {}).get('percentage', 0.0)}%), Avg: `{res.get('underpredictions', {}).get('mean_underprediction_min', 0.0)}m`",
            f"- **Exact / Neutral ($\\\\pm 0.5m$):** {res.get('exact_predictions', {}).get('count', 0)} ({res.get('exact_predictions', {}).get('percentage', 0.0)}%)",
            f"- **Large Outlier Errors (>10m):** {res.get('large_errors', {}).get('count', 0)} ({res.get('large_errors', {}).get('percentage', 0.0)}%)",
            f"",
            f"### Error Distribution (Residuals)",
            f"",
            f"```text",
            f"Min: {res.get('error_distribution', {}).get('min', 0.0)}m | Q25: {res.get('error_distribution', {}).get('q25', 0.0)}m | Median: {res.get('error_distribution', {}).get('median', 0.0)}m | Q75: {res.get('error_distribution', {}).get('q75', 0.0)}m | Max: {res.get('error_distribution', {}).get('max', 0.0)}m",
            f"Std Dev: {res.get('error_distribution', {}).get('std', 0.0)}m | Skewness: {res.get('error_distribution', {}).get('skewness', 0.0)}",
            f"```",
            f""
        ]

        # Poor performance highlights
        poor_conds = res.get("poor_performing_conditions", [])
        if poor_conds:
            lines.append("### ⚠️ Flagged Behavioral Bottlenecks (High Error Subgroups)")
            lines.append("")
            lines.append("| Dimension | Subgroup | Samples | Subgroup MAE | Overall MAE | Error Inflation |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for pc in poor_conds:
                lines.append(f"| {pc['segment_dimension']} | **{pc['segment_value']}** | {pc['sample_count']} | {pc['segment_mae']}m | {pc['overall_mae']}m | +{pc['error_inflation_pct']}% |")
            lines.append("")

        # Performance Segmentation Tables
        lines.append("---")
        lines.append("")
        lines.append("## 3. Performance Segmentation")
        lines.append("")

        for dim_key, dim_title in [
            ("day_of_week", "Day of the Week"),
            ("event_name", "Event Routine"),
            ("importance", "Importance Level"),
            ("bag_prepared", "Bag Prepared State (1 = Prepared, 0 = Unprepared)"),
            ("prep_duration_bucket", "Preparation Duration Ranges")
        ]:
            if dim_key in seg and seg[dim_key]:
                lines.append(f"### By {dim_title}")
                lines.append("")
                lines.append("| Category | Observations | MAE | RMSE | Within ±5m |")
                lines.append("| :--- | :--- | :--- | :--- | :--- |")
                for row in seg[dim_key]:
                    lines.append(f"| **{row['segment']}** | {row['count']} | {row['mae']}m | {row['rmse']}m | {row['tolerance_5m_pct']}% |")
                lines.append("")

        return "\n".join(lines)
