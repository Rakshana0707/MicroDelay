"""Individual Behavioral Profile Foundation for MicroDelay.

Aggregates persistent behavioral patterns, estimation biases, and delay bottlenecks
from a user's verified departure history.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("microdelay.ml.adaptive.profile")


class BehavioralProfileEngine:
    """Builds and summarizes a user's persistent behavioral signature."""

    def build_profile(self, logs: List[Any]) -> Dict[str, Any]:
        """Aggregate behavioral traits from user departure logs."""
        if not logs:
            return {
                "has_profile": False,
                "total_records": 0,
                "prep_bias_min": 0.0,
                "item_search_frequency": 0.0,
                "phone_distraction_frequency": 0.0,
                "last_task_frequency": 0.0,
                "punctuality_rate": 1.0,
                "dominant_bottleneck": "none",
                "summary": "No historical observations recorded yet."
            }

        n = len(logs)
        prep_biases = []
        item_searches = 0
        phone_usages = 0
        last_tasks = 0
        on_time_count = 0

        total_phone_delay = 0.0
        total_items_delay = 0.0
        total_task_delay = 0.0
        total_prep_bias_delay = 0.0

        for log in logs:
            # Safe extraction handling both objects and dicts
            actual_prep = getattr(log, "actual_prep_duration_min", None)
            if actual_prep is None and isinstance(log, dict):
                actual_prep = log.get("actual_prep_duration_min")

            est_prep = getattr(log, "estimated_prep_duration_min", None)
            if est_prep is None and isinstance(log, dict):
                est_prep = log.get("estimated_prep_duration_min")

            tot_delay = getattr(log, "total_delay_min", None)
            if tot_delay is None and isinstance(log, dict):
                tot_delay = log.get("total_delay_min")

            if actual_prep is not None and est_prep is not None:
                prep_biases.append(float(actual_prep) - float(est_prep))

            if tot_delay is not None and float(tot_delay) <= 0.0:
                on_time_count += 1

            tag_items = getattr(log, "tag_finding_items", False) or (isinstance(log, dict) and log.get("tag_finding_items"))
            if tag_items:
                item_searches += 1

            tag_phone = getattr(log, "tag_phone_usage", False) or (isinstance(log, dict) and log.get("tag_phone_usage"))
            if tag_phone:
                phone_usages += 1

            tag_task = getattr(log, "tag_one_last_task", False) or (isinstance(log, dict) and log.get("tag_one_last_task"))
            if tag_task:
                last_tasks += 1

            phone_d = getattr(log, "delay_phone_usage_min", 0.0) or (isinstance(log, dict) and log.get("delay_phone_usage_min", 0.0))
            items_d = getattr(log, "delay_finding_items_min", 0.0) or (isinstance(log, dict) and log.get("delay_finding_items_min", 0.0))
            task_d = getattr(log, "delay_one_last_task_min", 0.0) or (isinstance(log, dict) and log.get("delay_one_last_task_min", 0.0))
            bias_d = getattr(log, "delay_prep_bias_min", 0.0) or (isinstance(log, dict) and log.get("delay_prep_bias_min", 0.0))

            total_phone_delay += float(phone_d or 0.0)
            total_items_delay += float(items_d or 0.0)
            total_task_delay += float(task_d or 0.0)
            total_prep_bias_delay += float(bias_d or 0.0)

        avg_bias = round(sum(prep_biases) / len(prep_biases), 1) if prep_biases else 0.0
        punctuality_rate = round(on_time_count / float(n), 2)

        # Determine dominant bottleneck
        delays = {
            "preparation_underestimation": total_prep_bias_delay,
            "item_searching": total_items_delay,
            "phone_distraction": total_phone_delay,
            "last_minute_tasks": total_task_delay
        }
        dominant_bottleneck = max(delays, key=delays.get) if any(v > 0 for v in delays.values()) else "none"

        return {
            "has_profile": True,
            "total_records": n,
            "prep_bias_min": avg_bias,
            "item_search_frequency": round(item_searches / float(n), 2),
            "phone_distraction_frequency": round(phone_usages / float(n), 2),
            "last_task_frequency": round(last_tasks / float(n), 2),
            "punctuality_rate": punctuality_rate,
            "dominant_bottleneck": dominant_bottleneck,
            "summary": (
                f"User displays an average preparation underestimation of {avg_bias} min. "
                f"Punctuality rate is {round(punctuality_rate * 100)}%. "
                f"Primary departure bottleneck is {dominant_bottleneck.replace('_', ' ')}."
            )
        }
