"""Behavioral Drift Detection Foundation for MicroDelay.

Detects shifts in user departure timing, prep estimation bias, and lateness rates
by comparing recent operational windows against baseline historical distributions.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("microdelay.ml.adaptive.drift")


class BehavioralDriftDetector:
    """Detects shifts in habits between long-term baseline and recent departure routines."""

    def __init__(self, recent_window_size: int = 5, drift_threshold_min: float = 3.0):
        self.recent_window_size = recent_window_size
        self.drift_threshold_min = drift_threshold_min

    def detect_drift(self, logs: List[Any]) -> Dict[str, Any]:
        """Assess whether recent user behavior has significantly drifted from historical baseline."""
        if not logs or len(logs) < (self.recent_window_size * 2):
            return {
                "drift_detected": False,
                "status": "insufficient_data",
                "direction": "stable",
                "mean_shift_min": 0.0,
                "recent_mean_delay": 0.0,
                "baseline_mean_delay": 0.0,
                "description": "Insufficient historical observations for drift comparison."
            }

        try:
            # Extract delay values
            delays = []
            for log in logs:
                d = getattr(log, "total_delay_min", None)
                if d is None and isinstance(log, dict):
                    d = log.get("total_delay_min")
                if d is not None:
                    delays.append(float(d))

            if len(delays) < (self.recent_window_size * 2):
                return {
                    "drift_detected": False,
                    "status": "insufficient_data",
                    "direction": "stable",
                    "mean_shift_min": 0.0,
                    "description": "Insufficient delay metrics for drift comparison."
                }

            recent = delays[:self.recent_window_size]
            baseline = delays[self.recent_window_size:]

            recent_mean = sum(recent) / float(len(recent))
            baseline_mean = sum(baseline) / float(len(baseline))
            shift = recent_mean - baseline_mean

            abs_shift = abs(shift)
            if abs_shift >= self.drift_threshold_min:
                status = "significant_drift"
                drift_detected = True
            elif abs_shift >= 1.5:
                status = "minor_drift"
                drift_detected = False
            else:
                status = "stable"
                drift_detected = False

            if shift < -1.5:
                direction = "improving"
            elif shift > 1.5:
                direction = "degrading"
            else:
                direction = "stable"

            return {
                "drift_detected": drift_detected,
                "status": status,
                "direction": direction,
                "mean_shift_min": round(shift, 1),
                "recent_mean_delay": round(recent_mean, 1),
                "baseline_mean_delay": round(baseline_mean, 1),
                "description": (
                    f"Recent routines show an average delay of {round(recent_mean, 1)} min vs baseline of "
                    f"{round(baseline_mean, 1)} min (shift: {round(shift, 1):+g} min). Trend is {direction}."
                )
            }
        except Exception as e:
            logger.warning("Error evaluating behavioral drift: %s", e)
            return {
                "drift_detected": False,
                "status": "error",
                "direction": "stable",
                "mean_shift_min": 0.0,
                "description": f"Drift analysis skipped: {e}"
            }
