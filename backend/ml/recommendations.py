"""Adaptive Personalized Recommendation & Intervention Tracking Engine for MicroDelay.

Transforms static and heuristic suggestions into evidence-driven, tracked interventions
that learn from observed user behavior, track lifecycle states, and measure empirical outcomes.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import datetime
import json
import uuid
import logging
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

try:
    from database import UserInterventionRecord, DepartureLog, User
except ModuleNotFoundError:
    from backend.database import UserInterventionRecord, DepartureLog, User

logger = logging.getLogger("microdelay.ml.recommendations")

INTERVENTION_CAUSATION_DISCLAIMER = (
    "Empirical Notice: Measured intervention outcomes reflect observed behavioral differences "
    "across routine departure logs. They demonstrate empirical associations rather than isolated "
    "laboratory causation. External environmental conditions (e.g., transit disruptions, seasonal "
    "schedules, personal events) may contribute to observed changes."
)


# =====================================================================
# 1. RECOMMENDATION CATALOG
# =====================================================================

@dataclass
class CatalogItem:
    """A structured behavioral intervention template in the catalog."""
    catalog_id: str
    category: str  # "preparation", "distraction", "timing", "routine_consistency"
    action: str
    description: str
    implementation_steps: List[str]
    behavioral_problem: str
    base_delay_reduction_min: float
    implementation_effort: str = "medium"  # "low", "medium", "high"
    target_metric: str = "total_delay_min"  # primary metric expected to improve

    def to_dict(self) -> Dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "category": self.category,
            "action": self.action,
            "description": self.description,
            "implementation_steps": self.implementation_steps,
            "behavioral_problem": self.behavioral_problem,
            "base_delay_reduction_min": self.base_delay_reduction_min,
            "implementation_effort": self.implementation_effort,
            "target_metric": self.target_metric
        }


RECOMMENDATION_CATALOG: Dict[str, CatalogItem] = {
    # Category: Preparation
    "prep_bag_previous_night": CatalogItem(
        catalog_id="prep_bag_previous_night",
        category="preparation",
        action="Evening Bag Packing Checklist",
        description="Pack your work bag, keys, badges, chargers, and essentials the night before.",
        implementation_steps=[
            "Designate 5 minutes immediately after dinner to assemble bag contents.",
            "Place the packed bag beside your entryway door before going to bed.",
            "Verify electronic devices (laptop, tablet, power banks) are charged or packed."
        ],
        behavioral_problem="unprepared_bag_friction",
        base_delay_reduction_min=3.8,
        implementation_effort="low",
        target_metric="bag_prepared"
    ),
    "prep_fixed_item_location": CatalogItem(
        catalog_id="prep_fixed_item_location",
        category="preparation",
        action="Doorway Launchpad (Keys & Wallet Drop)",
        description="Establish a dedicated tray or wall hook for keys, identification, wallet, and transit card.",
        implementation_steps=[
            "Place an entryway drop tray or mount a hook within 1 meter of your front door.",
            "Commit to depositing keys and transit card immediately upon crossing your threshold.",
            "Never place everyday essentials on desks, couches, or bedroom nightstands."
        ],
        behavioral_problem="frequent_item_search_delays",
        base_delay_reduction_min=3.5,
        implementation_effort="low",
        target_metric="delay_finding_items_min"
    ),
    "prep_clothing_in_advance": CatalogItem(
        catalog_id="prep_clothing_in_advance",
        category="preparation",
        action="Pre-Stage Morning Attire",
        description="Select, inspect, and lay out tomorrow's clothing before going to sleep.",
        implementation_steps=[
            "Check tomorrow's weather forecast before sleeping.",
            "Lay out your full outfit including socks, footwear, and outer layer.",
            "Eliminate morning wardrobe indecision and last-minute ironing overhead."
        ],
        behavioral_problem="preparation_duration_variability",
        base_delay_reduction_min=2.5,
        implementation_effort="low",
        target_metric="actual_prep_duration_min"
    ),
    "prep_materials_digital_sync": CatalogItem(
        catalog_id="prep_materials_digital_sync",
        category="preparation",
        action="Pre-Departure Material Staging",
        description="Stage work documents, presentation slides, notes, or classroom supplies the evening before.",
        implementation_steps=[
            "Download and cache needed files or presentations locally on your laptop.",
            "Confirm presentation adapters (HDMI/USB-C) and dongles are in your bag.",
            "Avoid morning panic searching for course materials or meeting notes."
        ],
        behavioral_problem="departure_material_friction",
        base_delay_reduction_min=3.0,
        implementation_effort="medium",
        target_metric="total_delay_min"
    ),

    # Category: Distraction
    "distract_prep_focus_mode": CatalogItem(
        catalog_id="distract_prep_focus_mode",
        category="distraction",
        action="15-Minute Pre-Departure Screen Lock",
        description="Switch phone to 'Do Not Disturb' or preparation focus mode 15 minutes before departure cutoff.",
        implementation_steps=[
            "Schedule an automated Do-Not-Disturb focus window starting 20 minutes before departure.",
            "Keep your phone screen face-down during grooming and dressing.",
            "Only allow emergency phone calls to bypass the silence filter."
        ],
        behavioral_problem="phone_notification_distraction",
        base_delay_reduction_min=3.5,
        implementation_effort="medium",
        target_metric="phone_distraction_freq"
    ),
    "distract_delay_phone_check": CatalogItem(
        catalog_id="distract_delay_phone_check",
        category="distraction",
        action="Postpone Nonessential Communications",
        description="Defer replying to non-urgent emails, chats, or social media until arriving at transit or office.",
        implementation_steps=[
            "Adopt the rule: 'No replying to work chats or emails within 15 minutes of departure'.",
            "Close Slack/Teams/WhatsApp desktop apps when beginning your final prep sequence.",
            "Jot down urgent thoughts on a sticky note rather than unlocking your smartphone."
        ],
        behavioral_problem="last_minute_communication_rush",
        base_delay_reduction_min=4.0,
        implementation_effort="medium",
        target_metric="delay_one_last_task_min"
    ),
    "distract_reduce_notifications": CatalogItem(
        catalog_id="distract_reduce_notifications",
        category="distraction",
        action="Doorway Notification Gate",
        description="Silence lock-screen alert banners that break routine momentum right before stepping out.",
        implementation_steps=[
            "Identify top 2 messaging apps sending non-urgent morning notifications.",
            "Disable lock-screen push banners for news, social feeds, and casual group chats.",
            "Step out of your front door immediately once shoes and jacket are on."
        ],
        behavioral_problem="doorway_distraction_loop",
        base_delay_reduction_min=2.8,
        implementation_effort="low",
        target_metric="delay_phone_min"
    ),

    # Category: Time Estimation
    "time_increase_prep_buffer": CatalogItem(
        catalog_id="time_increase_prep_buffer",
        category="timing",
        action="Calibrate Prep Buffer to Historical Reality",
        description="Increase your scheduled preparation duration by your observed average underestimation margin.",
        implementation_steps=[
            "Review your personal prep bias metric in your MicroDelay profile.",
            "Increase your default event preparation timer by the observed underestimation gap.",
            "Treat the earlier preparation start time as your non-negotiable departure anchor."
        ],
        behavioral_problem="repeated_prep_underestimation",
        base_delay_reduction_min=4.5,
        implementation_effort="medium",
        target_metric="prep_bias_min"
    ),
    "time_use_historical_prep_avg": CatalogItem(
        catalog_id="time_use_historical_prep_avg",
        category="timing",
        action="Benchmark Routine on Empirical Prep Average",
        description="Replace aspirational best-case prep assumptions with your empirical 30-day average prep time.",
        implementation_steps=[
            "Calibrate your morning timeline around your median actual prep duration.",
            "Avoid planning based on your 'fastest ever morning' rush experience.",
            "Incorporate a 5-minute safety buffer for minor friction."
        ],
        behavioral_problem="unrealistic_prep_expectations",
        base_delay_reduction_min=3.0,
        implementation_effort="low",
        target_metric="actual_prep_duration_min"
    ),
    "time_realistic_travel_buffer": CatalogItem(
        catalog_id="time_realistic_travel_buffer",
        category="timing",
        action="Transit Congestion Buffer Injection",
        description="Add a 15% transit congestion margin to departure time calculations.",
        implementation_steps=[
            "Check live transit navigation 30 minutes before leaving.",
            "Add a 5-minute parking or platform walking buffer to your planned travel duration.",
            "Target arriving at your destination 10 minutes prior to scheduled start."
        ],
        behavioral_problem="travel_estimation_compression",
        base_delay_reduction_min=3.5,
        implementation_effort="medium",
        target_metric="total_delay_min"
    ),
    "time_monday_morning_rule": CatalogItem(
        catalog_id="time_monday_morning_rule",
        category="timing",
        action="Monday Schedule Re-Alignment Rule",
        description="Start Monday morning preparation 10 minutes earlier to counteract weekend schedule shifts.",
        implementation_steps=[
            "Set your Monday morning wake-up alarm 10 minutes earlier than mid-week alarms.",
            "Review your Monday itinerary and schedule on Sunday evening.",
            "Keep morning grooming and breakfast simple on the first workday of the week."
        ],
        behavioral_problem="monday_departure_inertia",
        base_delay_reduction_min=4.0,
        implementation_effort="medium",
        target_metric="total_delay_min"
    ),

    # Category: Routine Consistency
    "routine_start_recurring_earlier": CatalogItem(
        catalog_id="routine_start_recurring_earlier",
        category="routine_consistency",
        action="Anchor Earliest Daily Routine",
        description="Shift recurring morning events that frequently trigger late departures 10 minutes earlier.",
        implementation_steps=[
            "Identify the specific morning event that generates your highest lateness rate.",
            "Adjust your departure alarm 10 minutes earlier for that recurring event.",
            "Reserve the final 10 minutes strictly for departure without starting new chores."
        ],
        behavioral_problem="recurrent_late_activity_anchor",
        base_delay_reduction_min=3.5,
        implementation_effort="medium",
        target_metric="lateness_rate"
    ),
    "routine_prepare_materials_before_sleep": CatalogItem(
        catalog_id="routine_prepare_materials_before_sleep",
        category="routine_consistency",
        action="Evening Departure Staging Ritual",
        description="Complete all departure logistics before sleeping to enable grab-and-go morning exits.",
        implementation_steps=[
            "Stage keys, badge, water bottle, transit card, and outerwear together by 10 PM.",
            "Review your morning timeline and set target departure alarms.",
            "Wake up with zero pre-departure logistics left to solve."
        ],
        behavioral_problem="morning_gathering_friction",
        base_delay_reduction_min=4.2,
        implementation_effort="medium",
        target_metric="total_delay_min"
    ),
    "routine_fixed_departure_sequence": CatalogItem(
        catalog_id="routine_fixed_departure_sequence",
        category="routine_consistency",
        action="Standardized Departure Sequence",
        description="Follow a strict, repeatable order of operations during the final 15 minutes before leaving.",
        implementation_steps=[
            "Follow a rigid closing sequence: 1. Pack devices -> 2. Put on shoes/coat -> 3. Grab keys/phone -> 4. Exit door.",
            "Never initiate cleaning or email reading once the departure sequence begins.",
            "Maintain the identical sequence on every weekday morning."
        ],
        behavioral_problem="routine_inconsistency",
        base_delay_reduction_min=3.0,
        implementation_effort="low",
        target_metric="actual_prep_duration_min"
    )
}


# =====================================================================
# 2. ELIGIBILITY EVALUATOR & EVIDENCE LINKER
# =====================================================================

@dataclass
class EligibleIntervention:
    """An intervention matched to concrete empirical user behavior."""
    catalog_item: CatalogItem
    behavioral_problem: str
    supporting_evidence: Dict[str, Any]
    calibrated_delay_reduction_min: float
    evidence_strength: float  # 0.5 to 2.0


class RecommendationEligibilityEvaluator:
    """Evaluates user historical data to find eligible interventions backed by evidence."""

    @staticmethod
    def evaluate_user_eligibility(
        user_logs: List[DepartureLog],
        profile_data: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None
    ) -> List[EligibleIntervention]:
        """Match behavioral problems in user history to eligible catalog interventions."""
        eligible: List[EligibleIntervention] = []
        n_logs = len(user_logs)

        # Convert logs to DataFrame for vectorized behavioral diagnosis
        if n_logs == 0:
            # Cold-start fallback: Recommend low-effort foundational habits
            eligible.append(EligibleIntervention(
                catalog_item=RECOMMENDATION_CATALOG["prep_bag_previous_night"],
                behavioral_problem="unprepared_bag_friction",
                supporting_evidence={
                    "problem_frequency_pct": 0.0,
                    "observed_overhead_min": 3.8,
                    "sample_count": 0,
                    "metric_name": "cold_start_baseline",
                    "metric_value": "No logs recorded",
                    "supporting_statement": "Foundational pacer habit: packing the night before eliminates morning gathering friction."
                },
                calibrated_delay_reduction_min=3.8,
                evidence_strength=0.8
            ))
            eligible.append(EligibleIntervention(
                catalog_item=RECOMMENDATION_CATALOG["prep_fixed_item_location"],
                behavioral_problem="frequent_item_search_delays",
                supporting_evidence={
                    "problem_frequency_pct": 0.0,
                    "observed_overhead_min": 3.2,
                    "sample_count": 0,
                    "metric_name": "cold_start_baseline",
                    "metric_value": "No logs recorded",
                    "supporting_statement": "Foundational habit: an entryway drop tray eliminates morning doorway searches."
                },
                calibrated_delay_reduction_min=3.2,
                evidence_strength=0.8
            ))
            return eligible

        df = pd.DataFrame([{
            "total_delay": max(0.0, getattr(l, "total_delay_min", 0.0)),
            "bag_prepared": bool(getattr(l, "bag_prepared", False)),
            "finding_items": bool(getattr(l, "tag_finding_items", False) or getattr(l, "finding_items", False)),
            "delay_finding_items_min": getattr(l, "delay_finding_items_min", 0.0) or (2.5 if (getattr(l, "tag_finding_items", False) or getattr(l, "finding_items", False)) else 0.0),
            "phone_distraction": bool(getattr(l, "tag_phone_usage", False) or getattr(l, "phone_distraction", False)),
            "delay_phone_min": getattr(l, "delay_phone_usage_min", 0.0) or getattr(l, "delay_phone_min", 0.0) or (3.0 if (getattr(l, "tag_phone_usage", False) or getattr(l, "phone_distraction", False)) else 0.0),
            "one_last_task": bool(getattr(l, "tag_one_last_task", False) or getattr(l, "one_last_task", False)),
            "delay_one_last_task_min": getattr(l, "delay_one_last_task_min", 0.0) or (3.0 if (getattr(l, "tag_one_last_task", False) or getattr(l, "one_last_task", False)) else 0.0),
            "actual_prep_duration": getattr(l, "actual_prep_duration_min", 30.0),
            "estimated_prep_duration": getattr(l, "estimated_prep_duration_min", 25.0),
            "day_of_week": getattr(l, "day_of_week", "Monday")
        } for l in user_logs])

        # 1. Evaluate Item Search Delays
        finding_count = len(df[df["finding_items"] == True])
        finding_rate = finding_count / n_logs
        avg_search_min = df[df["finding_items"] == True]["delay_finding_items_min"].mean() if finding_count > 0 else 0.0
        if finding_rate >= 0.15 or avg_search_min >= 2.0:
            calibrated_impact = round(max(2.5, avg_search_min), 1)
            strength = min(2.0, 0.9 + finding_rate + (avg_search_min / 5.0))
            eligible.append(EligibleIntervention(
                catalog_item=RECOMMENDATION_CATALOG["prep_fixed_item_location"],
                behavioral_problem="frequent_item_search_delays",
                supporting_evidence={
                    "problem_frequency_pct": round(finding_rate * 100, 1),
                    "observed_overhead_min": round(avg_search_min, 1),
                    "sample_count": n_logs,
                    "affected_sessions": finding_count,
                    "metric_name": "item_search_frequency",
                    "metric_value": f"{round(finding_rate * 100, 1)}%",
                    "supporting_statement": (
                        f"In {finding_count} of your last {n_logs} departures ({round(finding_rate*100)}%), "
                        f"item searches added ~{round(avg_search_min, 1)} minutes of doorway friction."
                    )
                },
                calibrated_delay_reduction_min=calibrated_impact,
                evidence_strength=strength
            ))

        # 2. Evaluate Unprepared Bag Habit
        unprepared_count = len(df[df["bag_prepared"] == False])
        unprepared_rate = unprepared_count / n_logs
        unprep_delay = df[df["bag_prepared"] == False]["total_delay"].mean() if unprepared_count > 0 else 0.0
        prep_delay = df[df["bag_prepared"] == True]["total_delay"].mean() if (n_logs - unprepared_count) > 0 else 0.0
        delay_diff = max(0.0, unprep_delay - prep_delay)

        context_unprepared = context and not getattr(context, "bag_prepared", True)
        if unprepared_rate >= 0.25 or delay_diff >= 2.0 or context_unprepared:
            calibrated_impact = round(max(3.0, delay_diff if delay_diff > 0 else 3.8), 1)
            strength = min(2.0, 0.8 + unprepared_rate + (delay_diff / 4.0))
            eligible.append(EligibleIntervention(
                catalog_item=RECOMMENDATION_CATALOG["prep_bag_previous_night"],
                behavioral_problem="unprepared_bag_friction",
                supporting_evidence={
                    "problem_frequency_pct": round(unprepared_rate * 100, 1),
                    "observed_overhead_min": round(delay_diff, 1),
                    "sample_count": n_logs,
                    "affected_sessions": unprepared_count,
                    "metric_name": "unprepared_bag_rate",
                    "metric_value": f"{round(unprepared_rate * 100, 1)}%",
                    "supporting_statement": (
                        f"You left your bag unprepared on {round(unprepared_rate*100)}% of mornings, "
                        f"which correlates with ~{calibrated_impact}m higher departure delays."
                    )
                },
                calibrated_delay_reduction_min=calibrated_impact,
                evidence_strength=strength
            ))

        # 3. Evaluate Phone Notifications & Digital Distractions
        phone_count = len(df[df["phone_distraction"] == True])
        phone_rate = phone_count / n_logs
        avg_phone_min = df[df["phone_distraction"] == True]["delay_phone_min"].mean() if phone_count > 0 else 0.0
        if phone_rate >= 0.15 or avg_phone_min >= 2.0:
            calibrated_impact = round(max(2.5, avg_phone_min), 1)
            strength = min(2.0, 0.85 + phone_rate + (avg_phone_min / 5.0))
            eligible.append(EligibleIntervention(
                catalog_item=RECOMMENDATION_CATALOG["distract_prep_focus_mode"],
                behavioral_problem="phone_notification_distraction",
                supporting_evidence={
                    "problem_frequency_pct": round(phone_rate * 100, 1),
                    "observed_overhead_min": round(avg_phone_min, 1),
                    "sample_count": n_logs,
                    "affected_sessions": phone_count,
                    "metric_name": "phone_distraction_freq",
                    "metric_value": f"{round(phone_rate * 100, 1)}%",
                    "supporting_statement": (
                        f"Smartphone checks delayed {phone_count} of your last {n_logs} departures "
                        f"by an average of {round(avg_phone_min, 1)} minutes."
                    )
                },
                calibrated_delay_reduction_min=calibrated_impact,
                evidence_strength=strength
            ))

        # 4. Evaluate Last-Minute Tasks / Communications
        olt_count = len(df[df["one_last_task"] == True])
        olt_rate = olt_count / n_logs
        avg_olt_min = df[df["one_last_task"] == True]["delay_one_last_task_min"].mean() if olt_count > 0 else 0.0
        if olt_rate >= 0.15 or avg_olt_min >= 2.5:
            calibrated_impact = round(max(3.0, avg_olt_min), 1)
            strength = min(2.0, 0.9 + olt_rate + (avg_olt_min / 4.0))
            eligible.append(EligibleIntervention(
                catalog_item=RECOMMENDATION_CATALOG["distract_delay_phone_check"],
                behavioral_problem="last_minute_communication_rush",
                supporting_evidence={
                    "problem_frequency_pct": round(olt_rate * 100, 1),
                    "observed_overhead_min": round(avg_olt_min, 1),
                    "sample_count": n_logs,
                    "affected_sessions": olt_count,
                    "metric_name": "one_last_task_frequency",
                    "metric_value": f"{round(olt_rate * 100, 1)}%",
                    "supporting_statement": (
                        f"Nonessential last-minute tasks occurred in {round(olt_rate * 100)}% of checkouts, "
                        f"adding ~{round(avg_olt_min, 1)} minutes before stepping out."
                    )
                },
                calibrated_delay_reduction_min=calibrated_impact,
                evidence_strength=strength
            ))

        # 5. Evaluate Preparation Estimation Bias
        prep_diff = (df["actual_prep_duration"] - df["estimated_prep_duration"])
        mean_prep_bias = float(prep_diff.mean())
        if mean_prep_bias >= 2.0:
            calibrated_impact = round(mean_prep_bias, 1)
            strength = min(2.0, 0.9 + (mean_prep_bias / 5.0))
            eligible.append(EligibleIntervention(
                catalog_item=RECOMMENDATION_CATALOG["time_increase_prep_buffer"],
                behavioral_problem="repeated_prep_underestimation",
                supporting_evidence={
                    "problem_frequency_pct": round((len(df[prep_diff > 0]) / n_logs) * 100, 1),
                    "observed_overhead_min": round(mean_prep_bias, 1),
                    "sample_count": n_logs,
                    "metric_name": "personal_prep_bias_min",
                    "metric_value": f"+{round(mean_prep_bias, 1)}m",
                    "supporting_statement": (
                        f"You underestimate morning preparation time by an average of "
                        f"{round(mean_prep_bias, 1)} minutes across {n_logs} tracked departures."
                    )
                },
                calibrated_delay_reduction_min=calibrated_impact,
                evidence_strength=strength
            ))

        # 6. Evaluate Monday Morning Inertia
        monday_logs = df[df["day_of_week"] == "Monday"]
        if len(monday_logs) >= 2:
            monday_late_rate = len(monday_logs[monday_logs["total_delay"] > 0]) / len(monday_logs)
            monday_avg_delay = float(monday_logs["total_delay"].mean())
            if monday_late_rate >= 0.35 or monday_avg_delay >= 3.0:
                strength = min(2.0, 0.8 + monday_late_rate + (monday_avg_delay / 6.0))
                eligible.append(EligibleIntervention(
                    catalog_item=RECOMMENDATION_CATALOG["time_monday_morning_rule"],
                    behavioral_problem="monday_departure_inertia",
                    supporting_evidence={
                        "problem_frequency_pct": round(monday_late_rate * 100, 1),
                        "observed_overhead_min": round(monday_avg_delay, 1),
                        "sample_count": len(monday_logs),
                        "metric_name": "monday_lateness_rate",
                        "metric_value": f"{round(monday_late_rate * 100, 1)}%",
                        "supporting_statement": (
                            f"Your Monday lateness rate is {round(monday_late_rate * 100)}% "
                            f"(average delay: {round(monday_avg_delay, 1)}m) due to weekend schedule shift."
                        )
                    },
                    calibrated_delay_reduction_min=round(max(2.5, monday_avg_delay), 1),
                    evidence_strength=strength
                ))

        # 7. Evaluate Preparation Variability / Routine Inconsistency
        prep_std = float(df["actual_prep_duration"].std()) if n_logs > 2 else 0.0
        if prep_std >= 6.0:
            eligible.append(EligibleIntervention(
                catalog_item=RECOMMENDATION_CATALOG["routine_fixed_departure_sequence"],
                behavioral_problem="routine_inconsistency",
                supporting_evidence={
                    "problem_frequency_pct": round(prep_std, 1),
                    "observed_overhead_min": round(prep_std * 0.5, 1),
                    "sample_count": n_logs,
                    "metric_name": "prep_duration_std",
                    "metric_value": f"±{round(prep_std, 1)}m",
                    "supporting_statement": (
                        f"Your preparation duration varies widely (standard deviation: ±{round(prep_std, 1)}m). "
                        "A fixed departure sequence stabilizes morning flow."
                    )
                },
                calibrated_delay_reduction_min=round(max(2.0, prep_std * 0.4), 1),
                evidence_strength=min(1.8, 0.8 + (prep_std / 10.0))
            ))

        # Always ensure at least 2 foundational interventions exist if none triggered
        if len(eligible) < 2:
            eligible.append(EligibleIntervention(
                catalog_item=RECOMMENDATION_CATALOG["routine_prepare_materials_before_sleep"],
                behavioral_problem="morning_gathering_friction",
                supporting_evidence={
                    "problem_frequency_pct": 20.0,
                    "observed_overhead_min": 3.0,
                    "sample_count": n_logs,
                    "metric_name": "routine_standard",
                    "metric_value": "General optimization",
                    "supporting_statement": "Staging departure items the evening before reduces overall morning cognitive load."
                },
                calibrated_delay_reduction_min=3.0,
                evidence_strength=0.75
            ))

        return eligible


# =====================================================================
# 3. ADAPTIVE PRIORITIZER & PERSONALIZED LEARNING
# =====================================================================

class AdaptivePrioritizer:
    """Ranks eligible recommendations using empirical evidence, effort, feedback, and fatigue."""

    @staticmethod
    def calculate_priority_score(
        eligible: EligibleIntervention,
        user_history_records: List[UserInterventionRecord],
        now: Optional[datetime.datetime] = None
    ) -> Tuple[float, str, Dict[str, float]]:
        """Calculate multi-criteria priority score and determine priority tier."""
        now_dt = now or datetime.datetime.now(datetime.timezone.utc)
        cat_id = eligible.catalog_item.catalog_id
        rec_category = eligible.catalog_item.category

        # 1. Base Evidence & Impact Score
        evidence_score = eligible.evidence_strength * 10.0
        impact_score = min(2.0, max(0.8, eligible.calibrated_delay_reduction_min / 3.0))

        # 2. Implementation Effort Weight (low effort gets higher preference for easier adoption)
        effort_weights = {"low": 1.25, "medium": 1.0, "high": 0.8}
        effort_weight = effort_weights.get(eligible.catalog_item.implementation_effort, 1.0)

        # 3. User Feedback & Preference Weight
        feedback_weight = 1.0
        history_weight = 1.0
        fatigue_penalty = 1.0

        for rec in user_history_records:
            # Check for same catalog item or same category
            if rec.recommendation_id == cat_id:
                # Active or snoozed check
                if rec.status in ("accepted", "started"):
                    # Already active! Disqualify from new suggestions
                    return 0.0, "active", {}

                if rec.status == "snoozed" and rec.snoozed_until:
                    try:
                        snooze_dt = datetime.datetime.fromisoformat(rec.snoozed_until)
                        if snooze_dt.tzinfo is None:
                            snooze_dt = snooze_dt.replace(tzinfo=datetime.timezone.utc)
                        if snooze_dt > now_dt:
                            # Still snoozed! Suppress from suggestions
                            return 0.0, "snoozed", {}
                    except Exception:
                        pass

                # Check recent dismissal
                if rec.status == "dismissed" and rec.dismissed_at:
                    try:
                        dismiss_dt = datetime.datetime.fromisoformat(rec.dismissed_at)
                        if dismiss_dt.tzinfo is None:
                            dismiss_dt = dismiss_dt.replace(tzinfo=datetime.timezone.utc)
                        days_since_dismiss = (now_dt - dismiss_dt).total_seconds() / 86400.0
                        if days_since_dismiss < 3.0:
                            # Dismissed very recently: strong cooling fatigue penalty
                            fatigue_penalty *= 0.25
                        elif days_since_dismiss < 7.0:
                            fatigue_penalty *= 0.60
                    except Exception:
                        pass

                # Check past completion & measured outcome
                if rec.status == "completed" and rec.measured_outcome:
                    try:
                        outcome = json.loads(rec.measured_outcome) if isinstance(rec.measured_outcome, str) else rec.measured_outcome
                        delta = outcome.get("delay_delta_min", 0.0)
                        if delta < -1.0:
                            # Proven past effectiveness! Substantial boost
                            history_weight *= 1.35
                        elif delta > 1.0:
                            # Past trial did not improve delay
                            history_weight *= 0.80
                    except Exception:
                        pass

                # User explicit rating / helpfulness feedback
                if rec.user_rating is not None:
                    if rec.user_rating >= 4 or rec.user_helpful is True:
                        feedback_weight *= 1.30
                    elif rec.user_rating <= 2 or rec.user_helpful is False:
                        feedback_weight *= 0.65

            # Category preference reinforcement
            elif rec.category == rec_category and rec.user_helpful is True:
                feedback_weight *= 1.08

        total_score = round(
            evidence_score * impact_score * effort_weight * feedback_weight * history_weight * fatigue_penalty,
            2
        )

        priority = "high" if total_score >= 12.0 else ("medium" if total_score >= 7.0 else "low")

        components = {
            "evidence_score": round(evidence_score, 2),
            "impact_score": round(impact_score, 2),
            "effort_weight": round(effort_weight, 2),
            "feedback_weight": round(feedback_weight, 2),
            "history_weight": round(history_weight, 2),
            "fatigue_penalty": round(fatigue_penalty, 2),
            "total_score": total_score
        }

        return total_score, priority, components

    @classmethod
    def rank_interventions(
        cls,
        eligible_items: List[EligibleIntervention],
        user_history: List[UserInterventionRecord],
        now: Optional[datetime.datetime] = None
    ) -> List[Tuple[EligibleIntervention, float, str, Dict[str, float]]]:
        """Rank eligible interventions by adaptive composite score."""
        ranked = []
        for item in eligible_items:
            score, priority, components = cls.calculate_priority_score(item, user_history, now=now)
            if score > 0.0:
                ranked.append((item, score, priority, components))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


# =====================================================================
# 4. EFFECTIVENESS MEASUREMENT & CONVERSION SERVICE
# =====================================================================

class EffectivenessMeasurementService:
    """Calculates empirical before-and-after outcome metrics for accepted interventions."""

    @staticmethod
    def capture_baseline_period(
        user_logs: List[DepartureLog],
        reference_time: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """Snapshot the baseline behavioral metrics immediately preceding intervention acceptance."""
        ref_dt = reference_time or datetime.datetime.now(datetime.timezone.utc)
        
        # Filter logs strictly recorded prior to reference time
        pre_logs = []
        for l in user_logs:
            rec_str = getattr(l, "recorded_at", None) or getattr(l, "created_at", None)
            if rec_str:
                try:
                    dt = datetime.datetime.fromisoformat(str(rec_str).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=datetime.timezone.utc)
                    if dt <= ref_dt:
                        pre_logs.append(l)
                except Exception:
                    pre_logs.append(l)
            else:
                pre_logs.append(l)

        # Take up to last 10 pre-intervention checkouts
        pre_logs = pre_logs[-10:] if len(pre_logs) > 10 else pre_logs
        n_base = len(pre_logs)

        if n_base == 0:
            return {
                "sample_count": 0,
                "avg_departure_delay": 0.0,
                "lateness_rate": 0.0,
                "avg_prep_duration": 30.0,
                "prep_bias": 0.0,
                "item_search_rate": 0.0,
                "phone_distraction_rate": 0.0,
                "unprepared_bag_rate": 0.0,
                "captured_at": ref_dt.isoformat()
            }

        delays = [max(0.0, getattr(l, "total_delay_min", 0.0)) for l in pre_logs]
        prep_diffs = [
            getattr(l, "actual_prep_duration_min", 30.0) - getattr(l, "estimated_prep_duration_min", 25.0)
            for l in pre_logs
        ]

        return {
            "sample_count": n_base,
            "avg_departure_delay": round(float(np.mean(delays)), 2),
            "lateness_rate": round(float(sum(1 for d in delays if d > 0) / n_base), 3),
            "avg_prep_duration": round(float(np.mean([getattr(l, "actual_prep_duration_min", 30.0) for l in pre_logs])), 1),
            "prep_bias": round(float(np.mean(prep_diffs)), 2),
            "item_search_rate": round(sum(1 for l in pre_logs if (getattr(l, "tag_finding_items", False) or getattr(l, "finding_items", False))) / n_base, 3),
            "phone_distraction_rate": round(sum(1 for l in pre_logs if (getattr(l, "tag_phone_usage", False) or getattr(l, "phone_distraction", False))) / n_base, 3),
            "unprepared_bag_rate": round(sum(1 for l in pre_logs if not getattr(l, "bag_prepared", True)) / n_base, 3),
            "captured_at": ref_dt.isoformat()
        }

    @staticmethod
    def calculate_effectiveness(
        baseline_metrics: Dict[str, Any],
        post_logs: List[DepartureLog],
        catalog_item: Optional[CatalogItem] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """Compute empirical change in behavioral outcomes between baseline and post-intervention periods."""
        n_post = len(post_logs)
        confounders: List[str] = []

        if n_post < 2:
            return (
                {
                    "status": "insufficient_post_observations",
                    "post_sample_count": n_post,
                    "required_samples": 2,
                    "message": "At least 2 departure routines after intervention start are needed to measure outcome changes."
                },
                {"sample_count": n_post},
                ["Insufficient observation period to reliably measure habit adjustment."]
            )

        post_delays = [max(0.0, getattr(l, "total_delay_min", 0.0)) for l in post_logs]
        post_prep_durations = [getattr(l, "actual_prep_duration_min", 30.0) for l in post_logs]
        post_prep_diffs = [
            getattr(l, "actual_prep_duration_min", 30.0) - getattr(l, "estimated_prep_duration_min", 25.0)
            for l in post_logs
        ]

        post_metrics = {
            "sample_count": n_post,
            "avg_departure_delay": round(float(np.mean(post_delays)), 2),
            "lateness_rate": round(float(sum(1 for d in post_delays if d > 0) / n_post), 3),
            "avg_prep_duration": round(float(np.mean(post_prep_durations)), 1),
            "prep_bias": round(float(np.mean(post_prep_diffs)), 2),
            "item_search_rate": round(sum(1 for l in post_logs if (getattr(l, "tag_finding_items", False) or getattr(l, "finding_items", False))) / n_post, 3),
            "phone_distraction_rate": round(sum(1 for l in post_logs if (getattr(l, "tag_phone_usage", False) or getattr(l, "phone_distraction", False))) / n_post, 3),
            "unprepared_bag_rate": round(sum(1 for l in post_logs if not getattr(l, "bag_prepared", True)) / n_post, 3)
        }

        # Calculate Deltas
        base_delay = baseline_metrics.get("avg_departure_delay", 0.0)
        post_delay = post_metrics["avg_departure_delay"]
        delay_delta = round(post_delay - base_delay, 2)

        base_late_rate = baseline_metrics.get("lateness_rate", 0.0)
        post_late_rate = post_metrics["lateness_rate"]
        late_rate_delta = round(post_late_rate - base_late_rate, 3)

        base_prep_bias = baseline_metrics.get("prep_bias", 0.0)
        post_prep_bias = post_metrics["prep_bias"]
        prep_bias_delta = round(post_prep_bias - base_prep_bias, 2)

        # Detect Confounders
        base_n = baseline_metrics.get("sample_count", 0)
        if base_n < 4 or n_post < 4:
            confounders.append("Small sample size: outcome is subject to routine day-to-day variance.")

        # Check for planned prep duration shifts
        base_planned_prep = baseline_metrics.get("avg_prep_duration", 30.0) - base_prep_bias
        post_planned_prep = post_metrics["avg_prep_duration"] - post_prep_bias
        if abs(post_planned_prep - base_planned_prep) >= 5.0:
            confounders.append(
                f"Planned prep duration shifted by {round(post_planned_prep - base_planned_prep, 1)}m between periods."
            )

        # Check target metric specific improvements
        target_improvement = False
        target_metric = catalog_item.target_metric if catalog_item else "total_delay_min"

        if target_metric == "bag_prepared":
            bag_diff = baseline_metrics.get("unprepared_bag_rate", 0.0) - post_metrics["unprepared_bag_rate"]
            target_improvement = bag_diff > 0.15
        elif target_metric == "delay_finding_items_min":
            search_diff = baseline_metrics.get("item_search_rate", 0.0) - post_metrics["item_search_rate"]
            target_improvement = search_diff > 0.10
        elif target_metric == "phone_distraction_freq":
            phone_diff = baseline_metrics.get("phone_distraction_rate", 0.0) - post_metrics["phone_distraction_rate"]
            target_improvement = phone_diff > 0.10
        elif target_metric == "prep_bias_min":
            target_improvement = prep_bias_delta < -1.0
        else:
            target_improvement = delay_delta < -1.0

        is_overall_improved = (delay_delta < -0.5) or (late_rate_delta < -0.10) or target_improvement

        measured_outcome = {
            "status": "evaluated",
            "is_improved": is_overall_improved,
            "delay_delta_min": delay_delta,
            "lateness_rate_delta": late_rate_delta,
            "prep_bias_delta_min": prep_bias_delta,
            "baseline_avg_delay": base_delay,
            "post_avg_delay": post_delay,
            "baseline_lateness_rate": base_late_rate,
            "post_lateness_rate": post_late_rate,
            "baseline_sample_count": base_n,
            "post_sample_count": n_post,
            "target_metric": target_metric,
            "target_improvement": target_improvement,
            "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        return measured_outcome, post_metrics, confounders


# =====================================================================
# 5. INTERVENTION SERVICE (API & LIFECYCLE MANAGEMENT)
# =====================================================================

class InterventionService:
    """Manages creation, lifecycle transitions, effectiveness tracking, and queries."""

    @classmethod
    def get_personalized_recommendations(
        cls,
        db: Session,
        user_id: int,
        context: Optional[Any] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate and rank personalized eligible recommendations for user."""
        # 1. Fetch user's genuine logs
        user_logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id,
            DepartureLog.observation_status == "valid"
        ).order_by(DepartureLog.id.asc()).all()

        # 2. Fetch existing intervention history for user
        history_records = db.query(UserInterventionRecord).filter(
            UserInterventionRecord.user_id == user_id
        ).all()

        # 3. Evaluate eligibility
        eligible_items = RecommendationEligibilityEvaluator.evaluate_user_eligibility(
            user_logs=user_logs,
            context=context
        )

        # 4. Rank using adaptive prioritizer
        ranked = AdaptivePrioritizer.rank_interventions(
            eligible_items=eligible_items,
            user_history=history_records
        )

        output: List[Dict[str, Any]] = []
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 5. Select top suggestions and ensure persistence in DB
        for item, score, priority, components in ranked[:limit]:
            cat = item.catalog_item
            
            # Check if this intervention already exists in DB for this user in suggested state
            existing = db.query(UserInterventionRecord).filter(
                UserInterventionRecord.user_id == user_id,
                UserInterventionRecord.recommendation_id == cat.catalog_id,
                UserInterventionRecord.status == "suggested"
            ).first()

            if not existing:
                intv_id = f"intv_{user_id}_{cat.catalog_id}_{uuid.uuid4().hex[:6]}"
                record = UserInterventionRecord(
                    intervention_id=intv_id,
                    user_id=user_id,
                    recommendation_id=cat.catalog_id,
                    category=cat.category,
                    action=cat.action,
                    description=cat.description,
                    implementation_steps=json.dumps(cat.implementation_steps),
                    behavioral_problem=item.behavioral_problem,
                    supporting_evidence=json.dumps(item.supporting_evidence),
                    status="suggested",
                    expected_delay_reduction_min=item.calibrated_delay_reduction_min,
                    expected_outcome=json.dumps({
                        "metric": cat.target_metric,
                        "expected_delay_reduction_min": item.calibrated_delay_reduction_min
                    }),
                    causation_disclaimer=INTERVENTION_CAUSATION_DISCLAIMER,
                    generated_at=now_str
                )
                db.add(record)
                try:
                    db.commit()
                    db.refresh(record)
                    intv_record = record
                except Exception as e:
                    db.rollback()
                    logger.warning("Could not persist suggested intervention: %s", e)
                    intv_record = None
            else:
                intv_record = existing

            rec_dict = {
                "intervention_id": intv_record.intervention_id if intv_record else f"intv_temp_{cat.catalog_id}",
                "recommendation_id": cat.catalog_id,
                "category": cat.category,
                "action": cat.action,
                "description": cat.description,
                "implementation_steps": cat.implementation_steps,
                "implementation_effort": cat.implementation_effort,
                "behavioral_problem": item.behavioral_problem,
                "supporting_evidence": item.supporting_evidence,
                "expected_delay_reduction_min": item.calibrated_delay_reduction_min,
                "priority": priority,
                "priority_score": score,
                "scoring_breakdown": components,
                "causation_disclaimer": INTERVENTION_CAUSATION_DISCLAIMER,
                "status": "suggested"
            }
            output.append(rec_dict)

        return output

    @classmethod
    def accept_intervention(
        cls,
        db: Session,
        user_id: int,
        intervention_id: str
    ) -> Dict[str, Any]:
        """User accepts an intervention; initializes baseline period snapshot."""
        rec = db.query(UserInterventionRecord).filter(
            UserInterventionRecord.intervention_id == intervention_id,
            UserInterventionRecord.user_id == user_id
        ).first()

        if not rec:
            raise ValueError("Intervention not found or access denied.")

        now_dt = datetime.datetime.now(datetime.timezone.utc)
        now_str = now_dt.isoformat()

        # Capture baseline period from preceding logs
        user_logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id,
            DepartureLog.observation_status == "valid"
        ).order_by(DepartureLog.id.asc()).all()

        baseline = EffectivenessMeasurementService.capture_baseline_period(user_logs, reference_time=now_dt)

        rec.status = "accepted"
        rec.accepted_at = now_str
        rec.started_at = now_str
        rec.baseline_metrics = json.dumps(baseline)

        db.commit()
        db.refresh(rec)
        return cls._serialize_intervention(rec)

    @classmethod
    def dismiss_intervention(
        cls,
        db: Session,
        user_id: int,
        intervention_id: str
    ) -> Dict[str, Any]:
        """User dismisses recommendation; records dismissal timestamp for fatigue cooling."""
        rec = db.query(UserInterventionRecord).filter(
            UserInterventionRecord.intervention_id == intervention_id,
            UserInterventionRecord.user_id == user_id
        ).first()

        if not rec:
            raise ValueError("Intervention not found or access denied.")

        rec.status = "dismissed"
        rec.dismissed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        db.commit()
        db.refresh(rec)
        return cls._serialize_intervention(rec)

    @classmethod
    def snooze_intervention(
        cls,
        db: Session,
        user_id: int,
        intervention_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Snooze recommendation for N days."""
        rec = db.query(UserInterventionRecord).filter(
            UserInterventionRecord.intervention_id == intervention_id,
            UserInterventionRecord.user_id == user_id
        ).first()

        if not rec:
            raise ValueError("Intervention not found or access denied.")

        now_dt = datetime.datetime.now(datetime.timezone.utc)
        snooze_dt = now_dt + datetime.timedelta(days=max(1, days))

        rec.status = "snoozed"
        rec.snoozed_until = snooze_dt.isoformat()

        db.commit()
        db.refresh(rec)
        return cls._serialize_intervention(rec)

    @classmethod
    def complete_intervention(
        cls,
        db: Session,
        user_id: int,
        intervention_id: str,
        rating: Optional[int] = None,
        notes: Optional[str] = None,
        helpful: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Mark intervention as completed and evaluate empirical outcome change."""
        rec = db.query(UserInterventionRecord).filter(
            UserInterventionRecord.intervention_id == intervention_id,
            UserInterventionRecord.user_id == user_id
        ).first()

        if not rec:
            raise ValueError("Intervention not found or access denied.")

        now_dt = datetime.datetime.now(datetime.timezone.utc)
        rec.status = "completed"
        rec.completed_at = now_dt.isoformat()

        if rating is not None:
            rec.user_rating = max(1, min(5, rating))
        if notes is not None:
            rec.user_feedback_notes = str(notes).strip()
        if helpful is not None:
            rec.user_helpful = bool(helpful)

        # Calculate effectiveness against post-intervention logs
        start_time_str = rec.started_at or rec.accepted_at or rec.generated_at
        start_dt = datetime.datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)

        post_logs = []
        user_logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id,
            DepartureLog.observation_status == "valid"
        ).order_by(DepartureLog.id.asc()).all()

        for l in user_logs:
            rec_str = getattr(l, "recorded_at", None) or getattr(l, "created_at", None)
            if rec_str:
                try:
                    dt = datetime.datetime.fromisoformat(str(rec_str).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=datetime.timezone.utc)
                    if dt >= start_dt:
                        post_logs.append(l)
                except Exception:
                    pass

        # Parse baseline metrics
        baseline = json.loads(rec.baseline_metrics) if rec.baseline_metrics else {}
        catalog_item = RECOMMENDATION_CATALOG.get(rec.recommendation_id)

        measured_outcome, post_metrics, confounders = EffectivenessMeasurementService.calculate_effectiveness(
            baseline_metrics=baseline,
            post_logs=post_logs,
            catalog_item=catalog_item
        )

        rec.measured_outcome = json.dumps(measured_outcome)
        rec.post_metrics = json.dumps(post_metrics)
        rec.confounding_factors = json.dumps(confounders)
        rec.completion_status = "success" if measured_outcome.get("is_improved") else "neutral"

        db.commit()
        db.refresh(rec)
        return cls._serialize_intervention(rec)

    @classmethod
    def submit_feedback(
        cls,
        db: Session,
        user_id: int,
        intervention_id: str,
        rating: int,
        notes: Optional[str] = None,
        helpful: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Submit or update user feedback rating on an intervention."""
        rec = db.query(UserInterventionRecord).filter(
            UserInterventionRecord.intervention_id == intervention_id,
            UserInterventionRecord.user_id == user_id
        ).first()

        if not rec:
            raise ValueError("Intervention not found or access denied.")

        rec.user_rating = max(1, min(5, rating))
        if notes is not None:
            rec.user_feedback_notes = str(notes).strip()
        if helpful is not None:
            rec.user_helpful = bool(helpful)

        db.commit()
        db.refresh(rec)
        return cls._serialize_intervention(rec)

    @classmethod
    def get_user_interventions(
        cls,
        db: Session,
        user_id: int,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve user's intervention history with optional status filter."""
        query = db.query(UserInterventionRecord).filter(
            UserInterventionRecord.user_id == user_id
        )
        if status:
            query = query.filter(UserInterventionRecord.status == status)

        records = query.order_by(UserInterventionRecord.id.desc()).all()
        return [cls._serialize_intervention(r) for r in records]

    @classmethod
    def get_intervention_by_id(
        cls,
        db: Session,
        user_id: int,
        intervention_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve specific intervention enforcing ownership isolation."""
        rec = db.query(UserInterventionRecord).filter(
            UserInterventionRecord.intervention_id == intervention_id,
            UserInterventionRecord.user_id == user_id
        ).first()
        return cls._serialize_intervention(rec) if rec else None

    @classmethod
    def get_intervention_effectiveness(
        cls,
        db: Session,
        user_id: int,
        intervention_id: str
    ) -> Dict[str, Any]:
        """Return detailed empirical effectiveness analysis for an intervention."""
        rec = db.query(UserInterventionRecord).filter(
            UserInterventionRecord.intervention_id == intervention_id,
            UserInterventionRecord.user_id == user_id
        ).first()

        if not rec:
            raise ValueError("Intervention not found or access denied.")

        baseline = json.loads(rec.baseline_metrics) if rec.baseline_metrics else {}
        post = json.loads(rec.post_metrics) if rec.post_metrics else {}
        outcome = json.loads(rec.measured_outcome) if rec.measured_outcome else {}
        confounders = json.loads(rec.confounding_factors) if rec.confounding_factors else []

        return {
            "intervention_id": rec.intervention_id,
            "action": rec.action,
            "status": rec.status,
            "baseline_metrics": baseline,
            "post_metrics": post,
            "measured_outcome": outcome,
            "confounding_factors": confounders,
            "causation_disclaimer": rec.causation_disclaimer or INTERVENTION_CAUSATION_DISCLAIMER
        }

    @staticmethod
    def _serialize_intervention(rec: UserInterventionRecord) -> Dict[str, Any]:
        """Serialize a database intervention record into standard response schema."""
        return {
            "intervention_id": rec.intervention_id,
            "recommendation_id": rec.recommendation_id,
            "category": rec.category,
            "action": rec.action,
            "description": rec.description,
            "implementation_steps": json.loads(rec.implementation_steps) if rec.implementation_steps else [],
            "behavioral_problem": rec.behavioral_problem,
            "supporting_evidence": json.loads(rec.supporting_evidence) if rec.supporting_evidence else {},
            "status": rec.status,
            "expected_delay_reduction_min": round(rec.expected_delay_reduction_min, 1),
            "expected_outcome": json.loads(rec.expected_outcome) if rec.expected_outcome else {},
            "measured_outcome": json.loads(rec.measured_outcome) if rec.measured_outcome else None,
            "baseline_metrics": json.loads(rec.baseline_metrics) if rec.baseline_metrics else None,
            "post_metrics": json.loads(rec.post_metrics) if rec.post_metrics else None,
            "confounding_factors": json.loads(rec.confounding_factors) if rec.confounding_factors else [],
            "causation_disclaimer": rec.causation_disclaimer,
            "generated_at": rec.generated_at,
            "accepted_at": rec.accepted_at,
            "started_at": rec.started_at,
            "completed_at": rec.completed_at,
            "dismissed_at": rec.dismissed_at,
            "snoozed_until": rec.snoozed_until,
            "user_rating": rec.user_rating,
            "user_feedback_notes": rec.user_feedback_notes,
            "user_helpful": rec.user_helpful,
            "completion_status": rec.completion_status
        }
