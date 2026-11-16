"""Observed vs. Estimated Delay Attribution for MicroDelay.

This module separates directly observed behavioral durations from system-estimated
contributions, preventing heuristic pro-rata shares from being presented as measured causal effects.

Attribution Types:
1. OBSERVED: Directly measured/timed behavioral durations (e.g., timed phone distraction, item searching).
2. ESTIMATED: Model-inferred category allocations or estimated preparation bias.
3. RESIDUAL: Unexplained departure delay.
4. LEGACY: Historical records allocated using previous fixed pro-rata weights.

Budget & Overlap Handling:
- Departure delay budget B = max(0.0, total_delay_min).
- If sum(observed) > B: Explicitly documents overlapping intervals / buffer absorption.
- If total_delay_min <= 0: Behavioral distractions are logged as prep-phase activities
  with 0.0 departure delay contribution.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AttributionType(str, Enum):
    OBSERVED = "observed"
    ESTIMATED = "estimated"
    RESIDUAL = "residual"
    LEGACY = "legacy_estimated"


class AttributionConfidence(str, Enum):
    MEASURED = "measured"
    INFERRED = "inferred"
    HEURISTIC = "heuristic"
    UNEXPLAINED = "unexplained"


class AttributionSource(str, Enum):
    DIRECT_INPUT = "direct_input"
    TIMER_OBSERVED = "timer_observed"
    MODEL_INFERRED = "model_inferred"
    PRO_RATA_FALLBACK = "pro_rata_fallback"
    LEGACY_RECORD = "legacy_record"


@dataclass
class CategoryAttribution:
    """Metadata for a single behavioral or structural delay contribution."""
    category: str  # 'one_last_task', 'finding_items', 'waiting', 'phone_usage', 'prep_bias', 'unexplained_residual'
    duration_min: float
    departure_impact_min: float
    attribution_type: str  # 'observed', 'estimated', 'residual', 'legacy_estimated'
    data_source: str
    confidence: str        # 'measured', 'inferred', 'heuristic', 'unexplained'
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "duration_min": round(self.duration_min, 2),
            "departure_impact_min": round(self.departure_impact_min, 2),
            "attribution_type": self.attribution_type,
            "data_source": self.data_source,
            "confidence": self.confidence,
            "notes": self.notes
        }


@dataclass
class DepartureAttributionResult:
    """Structured breakdown of departure delay attribution."""
    total_delay_min: float
    departure_delay_budget_min: float
    is_early_or_on_time: bool
    early_departure_buffer_min: float
    has_overlapping_durations: bool
    overlapping_duration_min: float
    overlap_explanation: Optional[str]
    total_observed_min: float
    total_estimated_min: float
    unexplained_residual_min: float
    attribution_completeness_pct: float
    categories: List[CategoryAttribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_delay_min": round(self.total_delay_min, 2),
            "departure_delay_budget_min": round(self.departure_delay_budget_min, 2),
            "is_early_or_on_time": self.is_early_or_on_time,
            "early_departure_buffer_min": round(self.early_departure_buffer_min, 2),
            "has_overlapping_durations": self.has_overlapping_durations,
            "overlapping_duration_min": round(self.overlapping_duration_min, 2),
            "overlap_explanation": self.overlap_explanation,
            "summary": {
                "total_observed_min": round(self.total_observed_min, 2),
                "total_estimated_min": round(self.total_estimated_min, 2),
                "unexplained_residual_min": round(self.unexplained_residual_min, 2),
                "attribution_completeness_pct": round(self.attribution_completeness_pct, 1)
            },
            "breakdown": [c.to_dict() for c in self.categories]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def get_legacy_allocated_dict(self) -> Dict[str, float]:
        """Produce backward-compatible allocated minutes for DepartureLog columns."""
        res = {
            "delay_one_last_task_min": 0.0,
            "delay_finding_items_min": 0.0,
            "delay_phone_usage_min": 0.0,
            "delay_waiting_min": 0.0,
            "delay_prep_bias_min": 0.0
        }
        for cat in self.categories:
            if cat.category == "one_last_task":
                res["delay_one_last_task_min"] = cat.departure_impact_min
            elif cat.category == "finding_items":
                res["delay_finding_items_min"] = cat.departure_impact_min
            elif cat.category == "phone_usage":
                res["delay_phone_usage_min"] = cat.departure_impact_min
            elif cat.category == "waiting":
                res["delay_waiting_min"] = cat.departure_impact_min
            elif cat.category == "prep_bias":
                res["delay_prep_bias_min"] = cat.departure_impact_min
        return res


class DelayAttributionEngine:
    """Computes explainable observed vs. estimated delay attribution."""

    DEFAULT_WEIGHTS = {
        "one_last_task": 6.0,
        "finding_items": 5.0,
        "waiting": 4.0,
        "phone_usage": 3.0
    }

    @classmethod
    def attribute_departure(
        cls,
        total_delay_min: float,
        actual_prep_duration_min: float,
        estimated_prep_duration_min: float,
        tags: Dict[str, bool],
        observed_durations: Optional[Dict[str, float]] = None
    ) -> DepartureAttributionResult:
        """Attribute delay distinguishing measured durations from estimated contributions.
        
        Args:
          total_delay_min: Actual departure time minus planned departure time in minutes.
          actual_prep_duration_min: Total preparation time elapsed.
          estimated_prep_duration_min: Scheduled preparation time.
          tags: Binary flags for ('one_last_task', 'finding_items', 'phone_usage', 'waiting').
          observed_durations: Directly measured minutes for each behavior if available.
        """
        obs_map = observed_durations or {}
        pos_delay = max(0.0, float(total_delay_min))
        is_early_or_on_time = (total_delay_min <= 0.0)
        early_buffer = max(0.0, -float(total_delay_min))

        categories: List[CategoryAttribution] = []

        # 1. Early Departure or On-Time (No Departure Lateness Budget)
        if is_early_or_on_time:
            # Check if any observed behaviors were recorded during preparation
            total_obs = 0.0
            for tag_name in ["one_last_task", "finding_items", "phone_usage", "waiting"]:
                tag_active = bool(tags.get(tag_name, False))
                obs_val = float(obs_map.get(tag_name, 0.0))
                if obs_val > 0.0 or tag_active:
                    dur = obs_val if obs_val > 0.0 else 0.0
                    total_obs += dur
                    categories.append(CategoryAttribution(
                        category=tag_name,
                        duration_min=dur,
                        departure_impact_min=0.0,
                        attribution_type=AttributionType.OBSERVED if obs_val > 0.0 else AttributionType.ESTIMATED,
                        data_source=AttributionSource.DIRECT_INPUT if obs_val > 0.0 else AttributionSource.PRO_RATA_FALLBACK,
                        confidence=AttributionConfidence.MEASURED if obs_val > 0.0 else AttributionConfidence.HEURISTIC,
                        notes="Behavior occurred during preparation but did not cause departure lateness."
                    ))

            overlap_explanation = None
            if total_obs > 0:
                overlap_explanation = f"Observed activities ({total_obs:.1f}m) were fully absorbed by prep buffer without causing departure delay."

            return DepartureAttributionResult(
                total_delay_min=total_delay_min,
                departure_delay_budget_min=0.0,
                is_early_or_on_time=True,
                early_departure_buffer_min=early_buffer,
                has_overlapping_durations=False,
                overlapping_duration_min=0.0,
                overlap_explanation=overlap_explanation,
                total_observed_min=total_obs,
                total_estimated_min=0.0,
                unexplained_residual_min=0.0,
                attribution_completeness_pct=100.0,
                categories=categories
            )

        # 2. Positive Delay: Allocate between Observed and Estimated
        # Identify which categories have directly observed durations
        active_tags = [t for t in ["one_last_task", "finding_items", "phone_usage", "waiting"] if tags.get(t, False)]

        observed_total = 0.0
        observed_cats: Dict[str, float] = {}
        unobserved_active_tags: List[str] = []

        for tag in ["one_last_task", "finding_items", "phone_usage", "waiting"]:
            if tag in obs_map and obs_map[tag] is not None and float(obs_map[tag]) > 0.0:
                val = float(obs_map[tag])
                observed_cats[tag] = val
                observed_total += val
            elif tag in active_tags:
                unobserved_active_tags.append(tag)

        # Check overlapping intervals / budget exceeded
        has_overlap = False
        overlapping_min = 0.0
        overlap_expl = None

        if observed_total > pos_delay:
            has_overlap = True
            overlapping_min = round(observed_total - pos_delay, 2)
            overlap_expl = (
                f"Directly observed behaviors sum to {observed_total:.1f}m, which exceeds the total departure delay "
                f"({pos_delay:.1f}m) by {overlapping_min:.1f}m. The excess duration occurred concurrently (multitasking) "
                f"or was absorbed during planned preparation buffer."
            )

        # Compute departure delay impact for directly observed items
        # If observed_total exceeds pos_delay, scale proportional departure impact to fit budget
        scale_factor = (pos_delay / observed_total) if (observed_total > pos_delay and observed_total > 0) else 1.0

        attributed_departure_sum = 0.0

        for cat_name, raw_dur in observed_cats.items():
            impact = round(raw_dur * scale_factor, 2)
            attributed_departure_sum += impact
            categories.append(CategoryAttribution(
                category=cat_name,
                duration_min=raw_dur,
                departure_impact_min=impact,
                attribution_type=AttributionType.OBSERVED,
                data_source=AttributionSource.DIRECT_INPUT,
                confidence=AttributionConfidence.MEASURED,
                notes=f"Directly measured: {raw_dur:.1f}m" + (f" (scaled to {impact:.1f}m due to overlap)" if has_overlap else "")
            ))

        # Remaining delay budget to allocate to unobserved tags or prep bias
        remaining_budget = max(0.0, pos_delay - attributed_departure_sum)
        estimated_total = 0.0

        if remaining_budget > 0.0 and unobserved_active_tags:
            # Distribute remaining budget across unobserved active tags using default weights
            active_weights = {t: cls.DEFAULT_WEIGHTS.get(t, 4.0) for t in unobserved_active_tags}
            w_sum = sum(active_weights.values())

            for tag in unobserved_active_tags:
                tag_share = round((active_weights[tag] / w_sum) * remaining_budget, 2)
                estimated_total += tag_share
                attributed_departure_sum += tag_share
                categories.append(CategoryAttribution(
                    category=tag,
                    duration_min=tag_share,
                    departure_impact_min=tag_share,
                    attribution_type=AttributionType.ESTIMATED,
                    data_source=AttributionSource.PRO_RATA_FALLBACK,
                    confidence=AttributionConfidence.HEURISTIC,
                    notes=f"System-estimated contribution ({active_weights[tag]:.0f}/{w_sum:.0f} weight ratio)."
                ))

        # 3. Preparation Bias Contribution
        # Planning fallacy: actual prep minus estimated prep
        prep_bias_raw = max(0.0, float(actual_prep_duration_min) - float(estimated_prep_duration_min))
        residual_delay = max(0.0, round(pos_delay - attributed_departure_sum, 2))

        if residual_delay > 0.0:
            # If prep bias accounts for part or all of the residual
            bias_contribution = min(residual_delay, prep_bias_raw)
            if bias_contribution > 0.0:
                estimated_total += bias_contribution
                attributed_departure_sum += bias_contribution
                categories.append(CategoryAttribution(
                    category="prep_bias",
                    duration_min=bias_contribution,
                    departure_impact_min=bias_contribution,
                    attribution_type=AttributionType.ESTIMATED,
                    data_source=AttributionSource.MODEL_INFERRED,
                    confidence=AttributionConfidence.INFERRED,
                    notes=f"Preparation duration overrun (actual {actual_prep_duration_min:.0f}m vs planned {estimated_prep_duration_min:.0f}m)."
                ))

        # Final unexplained residual
        unexplained = max(0.0, round(pos_delay - attributed_departure_sum, 2))
        if unexplained > 0.0:
            categories.append(CategoryAttribution(
                category="unexplained_residual",
                duration_min=unexplained,
                departure_impact_min=unexplained,
                attribution_type=AttributionType.RESIDUAL,
                data_source=AttributionSource.MODEL_INFERRED,
                confidence=AttributionConfidence.UNEXPLAINED,
                notes="Unexplained residual departure delay."
            ))

        total_explained = attributed_departure_sum
        completeness_pct = round(min(100.0, (total_explained / pos_delay * 100.0)), 1) if pos_delay > 0 else 100.0

        return DepartureAttributionResult(
            total_delay_min=total_delay_min,
            departure_delay_budget_min=pos_delay,
            is_early_or_on_time=False,
            early_departure_buffer_min=0.0,
            has_overlapping_durations=has_overlap,
            overlapping_duration_min=overlapping_min,
            overlap_explanation=overlap_expl,
            total_observed_min=round(observed_total, 2),
            total_estimated_min=round(estimated_total, 2),
            unexplained_residual_min=round(unexplained, 2),
            attribution_completeness_pct=completeness_pct,
            categories=categories
        )

    @classmethod
    def parse_legacy_record(cls, log: Any) -> DepartureAttributionResult:
        """Parse historical DepartureLog created under previous algorithm without modifying original record."""
        total_delay = float(getattr(log, "total_delay_min", 0.0))
        pos_delay = max(0.0, total_delay)

        # If already has attribution metadata JSON stored, decode it
        stored_meta = getattr(log, "attribution_metadata", None)
        if stored_meta:
            try:
                data = json.loads(stored_meta) if isinstance(stored_meta, str) else stored_meta
                cats = [
                    CategoryAttribution(
                        category=c["category"],
                        duration_min=c["duration_min"],
                        departure_impact_min=c["departure_impact_min"],
                        attribution_type=c["attribution_type"],
                        data_source=c["data_source"],
                        confidence=c["confidence"],
                        notes=c.get("notes")
                    )
                    for c in data.get("breakdown", [])
                ]
                return DepartureAttributionResult(
                    total_delay_min=data["total_delay_min"],
                    departure_delay_budget_min=data["departure_delay_budget_min"],
                    is_early_or_on_time=data["is_early_or_on_time"],
                    early_departure_buffer_min=data["early_departure_buffer_min"],
                    has_overlapping_durations=data["has_overlapping_durations"],
                    overlapping_duration_min=data["overlapping_duration_min"],
                    overlap_explanation=data.get("overlap_explanation"),
                    total_observed_min=data["summary"]["total_observed_min"],
                    total_estimated_min=data["summary"]["total_estimated_min"],
                    unexplained_residual_min=data["summary"]["unexplained_residual_min"],
                    attribution_completeness_pct=data["summary"]["attribution_completeness_pct"],
                    categories=cats
                )
            except Exception:
                pass

        # Legacy pro-rata columns
        categories = []
        col_mappings = [
            ("one_last_task", float(getattr(log, "delay_one_last_task_min", 0.0))),
            ("finding_items", float(getattr(log, "delay_finding_items_min", 0.0))),
            ("phone_usage", float(getattr(log, "delay_phone_usage_min", 0.0))),
            ("waiting", float(getattr(log, "delay_waiting_min", 0.0))),
            ("prep_bias", float(getattr(log, "delay_prep_bias_min", 0.0)))
        ]

        total_legacy_allocated = 0.0
        for cat_name, dur in col_mappings:
            if dur > 0.0:
                total_legacy_allocated += dur
                categories.append(CategoryAttribution(
                    category=cat_name,
                    duration_min=dur,
                    departure_impact_min=dur,
                    attribution_type=AttributionType.LEGACY.value,
                    data_source=AttributionSource.LEGACY_RECORD.value,
                    confidence=AttributionConfidence.HEURISTIC.value,
                    notes="Historical record allocated via previous fixed pro-rata weights."
                ))

        unexplained = max(0.0, round(pos_delay - total_legacy_allocated, 2))
        if unexplained > 0.0:
            categories.append(CategoryAttribution(
                category="unexplained_residual",
                duration_min=unexplained,
                departure_impact_min=unexplained,
                attribution_type=AttributionType.RESIDUAL.value,
                data_source=AttributionSource.MODEL_INFERRED.value,
                confidence=AttributionConfidence.UNEXPLAINED.value,
                notes="Legacy residual delay."
            ))

        return DepartureAttributionResult(
            total_delay_min=total_delay,
            departure_delay_budget_min=pos_delay,
            is_early_or_on_time=(total_delay <= 0.0),
            early_departure_buffer_min=max(0.0, -total_delay),
            has_overlapping_durations=False,
            overlapping_duration_min=0.0,
            overlap_explanation="Legacy record from earlier version.",
            total_observed_min=0.0,
            total_estimated_min=round(total_legacy_allocated, 2),
            unexplained_residual_min=round(unexplained, 2),
            attribution_completeness_pct=100.0 if pos_delay == 0 else round(min(100.0, (total_legacy_allocated / pos_delay * 100.0)), 1),
            categories=categories
        )
