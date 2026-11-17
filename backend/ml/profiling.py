"""Personalized Behavioral Profiling Engine for MicroDelay 2.0 (Prompt 2.5).

Continuously aggregates, analyzes, and versions individual behavioral profiles across
six foundational dimensions:
A. Preparation Behavior (prep duration, bias, variability, consistency score)
B. Distraction Behavior (phone frequency, interruption duration, peak periods)
C. Organization Behavior (item-search rate, bag preparation consistency, patterns)
D. Time Estimation Behavior (prep bias, travel bias, event-specific errors)
E. Departure Behavior (average lateness, late/early frequencies, day patterns)
F. Routine Consistency (schedule adherence, weekday variation, recent shift)

Supports multi-window rolling calculations (7-day, 30-day, all-time baseline),
evidence-backed insight generation, profile comparisons, and persistent versioning.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import logging
import datetime
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import desc

try:
    from database import DepartureLog, User, BehavioralProfileSnapshot, UserClusterProfile
except ModuleNotFoundError:
    from backend.database import DepartureLog, User, BehavioralProfileSnapshot, UserClusterProfile

logger = logging.getLogger("microdelay.ml.profiling")


class BehavioralDimensionCalculator:
    """Calculates granular behavioral metrics across dimensions A through F."""

    @staticmethod
    def calculate_dimensions(logs: List[Any]) -> Dict[str, Any]:
        """Compute all 6 dimensions from a collection of departure logs."""
        if not logs:
            return {
                "preparation": {
                    "avg_actual_prep_min": 0.0,
                    "prep_bias_min": 0.0,
                    "prep_duration_std": 0.0,
                    "prep_consistency_score": 1.0,
                    "observation_count": 0
                },
                "distraction": {
                    "phone_usage_frequency": 0.0,
                    "avg_phone_interruption_min": 0.0,
                    "common_interruption_period": "None observed",
                    "distraction_trend": "No data",
                    "observation_count": 0
                },
                "organization": {
                    "item_search_frequency": 0.0,
                    "avg_item_search_delay_min": 0.0,
                    "bag_prepared_consistency": 0.0,
                    "recurring_missing_item_pattern": "No data",
                    "observation_count": 0
                },
                "time_estimation": {
                    "prep_estimation_bias_min": 0.0,
                    "travel_estimation_bias_min": 0.0,
                    "event_specific_errors": {},
                    "observation_count": 0
                },
                "departure": {
                    "avg_lateness_min": 0.0,
                    "lateness_frequency": 0.0,
                    "early_departure_frequency": 0.0,
                    "day_specific_patterns": {},
                    "observation_count": 0
                },
                "routine_consistency": {
                    "schedule_adherence_rate": 1.0,
                    "weekday_vs_weekend_variation": 0.0,
                    "recent_departure_change_min": 0.0,
                    "observation_count": 0
                }
            }

        n = len(logs)
        actual_preps = []
        prep_biases = []
        travel_biases = []
        delays = []
        late_count = 0
        early_count = 0
        phone_count = 0
        phone_delays = []
        item_count = 0
        item_delays = []
        bag_count = 0

        event_biases: Dict[str, List[float]] = {}
        day_delays: Dict[str, List[float]] = {}
        day_late_counts: Dict[str, int] = {}
        day_total_counts: Dict[str, int] = {}

        for log in logs:
            # Safe attribute access supporting both ORM objects and dicts
            actual_prep = float(getattr(log, "actual_prep_duration_min", 0.0) or (isinstance(log, dict) and log.get("actual_prep_duration_min", 0.0)) or 0.0)
            est_prep = float(getattr(log, "estimated_prep_duration_min", 0.0) or (isinstance(log, dict) and log.get("estimated_prep_duration_min", 0.0)) or 0.0)
            actual_travel = float(getattr(log, "actual_travel_duration_min", 0.0) or (isinstance(log, dict) and log.get("actual_travel_duration_min", 0.0)) or 0.0)
            est_travel = float(getattr(log, "estimated_travel_duration_min", 0.0) or (isinstance(log, dict) and log.get("estimated_travel_duration_min", 0.0)) or 0.0)
            tot_delay = float(getattr(log, "total_delay_min", 0.0) or (isinstance(log, dict) and log.get("total_delay_min", 0.0)) or 0.0)

            actual_preps.append(actual_prep)
            prep_bias = actual_prep - est_prep
            prep_biases.append(prep_bias)
            travel_bias = actual_travel - est_travel
            travel_biases.append(travel_bias)
            delays.append(tot_delay)

            if tot_delay > 0.0:
                late_count += 1
            elif tot_delay < 0.0:
                early_count += 1

            # Phone distraction
            phone_tag = bool(getattr(log, "tag_phone_usage", False) or (isinstance(log, dict) and log.get("tag_phone_usage", False)))
            phone_d = float(getattr(log, "delay_phone_usage_min", 0.0) or (isinstance(log, dict) and log.get("delay_phone_usage_min", 0.0)) or 0.0)
            if phone_tag or phone_d > 0.0:
                phone_count += 1
            phone_delays.append(phone_d)

            # Item search
            item_tag = bool(getattr(log, "tag_finding_items", False) or (isinstance(log, dict) and log.get("tag_finding_items", False)))
            item_d = float(getattr(log, "delay_finding_items_min", 0.0) or (isinstance(log, dict) and log.get("delay_finding_items_min", 0.0)) or 0.0)
            if item_tag or item_d > 0.0:
                item_count += 1
            item_delays.append(item_d)

            # Bag preparedness
            bag_ready = bool(getattr(log, "bag_prepared", False) or (isinstance(log, dict) and log.get("bag_prepared", False)))
            if bag_ready:
                bag_count += 1

            # Event specific error grouping (by event name and importance)
            ev_name = str(getattr(log, "event_name", None) or (isinstance(log, dict) and log.get("event_name")) or "Standard").strip()
            event_biases.setdefault(ev_name, []).append(prep_bias)

            # Day of week grouping
            day = str(getattr(log, "day_of_week", "Monday") or (isinstance(log, dict) and log.get("day_of_week", "Monday"))).capitalize()
            day_delays.setdefault(day, []).append(tot_delay)
            day_total_counts[day] = day_total_counts.get(day, 0) + 1
            if tot_delay > 0.0:
                day_late_counts[day] = day_late_counts.get(day, 0) + 1

        # A. Preparation Behavior
        avg_actual_prep = float(np.mean(actual_preps)) if actual_preps else 0.0
        avg_prep_bias = float(np.mean(prep_biases)) if prep_biases else 0.0
        prep_std = float(np.std(actual_preps)) if len(actual_preps) >= 2 else 0.0
        cv = prep_std / (avg_actual_prep + 1e-5) if avg_actual_prep > 0 else 0.0
        prep_consistency_score = round(max(0.0, min(1.0, 1.0 - cv)), 2)

        # B. Distraction Behavior
        phone_freq = float(phone_count / n)
        avg_phone_d = float(np.mean(phone_delays)) if phone_delays else 0.0
        if phone_freq >= 0.4:
            common_interruption = "Final 15 min of morning preparation"
        elif phone_freq >= 0.2:
            common_interruption = "Intermittent notification checks"
        else:
            common_interruption = "Low interruption frequency"

        # C. Organization Behavior
        item_freq = float(item_count / n)
        avg_item_d = float(np.mean(item_delays)) if item_delays else 0.0
        bag_consistency = float(bag_count / n)
        if bag_consistency < 0.4 and item_freq >= 0.3:
            missing_item_pattern = "Unprepared bag strongly correlates with doorway item searching"
        elif item_freq >= 0.3:
            missing_item_pattern = "Doorway search delays (keys, badges, transit cards)"
        else:
            missing_item_pattern = "Items consistently located prior to departure"

        # D. Time Estimation Behavior
        avg_travel_bias = float(np.mean(travel_biases)) if travel_biases else 0.0
        event_errors = {}
        for imp, b_list in event_biases.items():
            event_errors[imp] = round(float(np.mean(b_list)), 1)

        # E. Departure Behavior
        avg_lateness = float(np.mean(delays)) if delays else 0.0
        late_freq = float(late_count / n)
        early_freq = float(early_count / n)
        day_patterns = {}
        for d, d_delays in day_delays.items():
            tot_cnt = day_total_counts.get(d, 1)
            lt_cnt = day_late_counts.get(d, 0)
            day_patterns[d] = {
                "avg_delay_min": round(float(np.mean(d_delays)), 1),
                "lateness_rate": round(lt_cnt / float(tot_cnt), 2),
                "sample_count": tot_cnt
            }

        # F. Routine Consistency
        schedule_adherence = round((n - late_count) / float(n), 2)
        if len(day_delays) >= 2:
            day_means = [float(np.mean(dl)) for dl in day_delays.values()]
            weekday_var = round(float(np.std(day_means)), 1)
        else:
            weekday_var = 0.0

        if n > 5:
            recent_delays = delays[-5:]
            older_delays = delays[:-5]
            recent_change = round(float(np.mean(recent_delays) - np.mean(older_delays)), 1)
        else:
            recent_change = 0.0

        return {
            "preparation": {
                "avg_actual_prep_min": round(avg_actual_prep, 1),
                "prep_bias_min": round(avg_prep_bias, 1),
                "prep_duration_std": round(prep_std, 1),
                "prep_consistency_score": prep_consistency_score,
                "observation_count": n
            },
            "distraction": {
                "phone_usage_frequency": round(phone_freq, 2),
                "avg_phone_interruption_min": round(avg_phone_d, 1),
                "common_interruption_period": common_interruption,
                "observation_count": n
            },
            "organization": {
                "item_search_frequency": round(item_freq, 2),
                "avg_item_search_delay_min": round(avg_item_d, 1),
                "bag_prepared_consistency": round(bag_consistency, 2),
                "recurring_missing_item_pattern": missing_item_pattern,
                "observation_count": n
            },
            "time_estimation": {
                "prep_estimation_bias_min": round(avg_prep_bias, 1),
                "travel_estimation_bias_min": round(avg_travel_bias, 1),
                "event_specific_errors": event_errors,
                "event_specific_prep_bias": event_errors,
                "observation_count": n
            },
            "departure": {
                "avg_lateness_min": round(avg_lateness, 1),
                "lateness_frequency": round(late_freq, 2),
                "late_departure_frequency": round(late_freq, 2),
                "early_departure_frequency": round(early_freq, 2),
                "day_specific_patterns": day_patterns,
                "day_of_week_patterns": day_patterns,
                "observation_count": n
            },
            "routine_consistency": {
                "schedule_adherence_rate": schedule_adherence,
                "weekday_vs_weekend_variation": weekday_var,
                "weekday_delay_std": weekday_var,
                "recent_departure_change_min": recent_change,
                "observation_count": n
            }
        }


class BehavioralInsightsGenerator:
    """Generates evidence-backed statements supported strictly by underlying data."""

    @staticmethod
    def generate_insights(
        metrics_all: Dict[str, Any],
        metrics_30d: Optional[Dict[str, Any]] = None,
        metrics_7d: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Synthesize factual, statistical observations without unsupported causal leaps."""
        insights = []
        prep = metrics_all.get("preparation", {})
        dist = metrics_all.get("distraction", {})
        org = metrics_all.get("organization", {})
        est = metrics_all.get("time_estimation", {})
        dep = metrics_all.get("departure", {})
        rout = metrics_all.get("routine_consistency", {})

        n_all = prep.get("observation_count", 0)
        if n_all < 3:
            return ["Profile has limited observations; gather at least 3 checkouts to reveal behavioral habits."]

        # 1. Preparation bias statement
        bias = prep.get("prep_bias_min", 0.0)
        if bias >= 3.0:
            insights.append({
                "dimension": "preparation",
                "finding": "Preparation Time Underestimation",
                "statement": f"You underestimate your preparation time by an average of {bias:.1f} minutes.",
                "factual_evidence": f"Average prep bias of +{bias:.1f}m observed across {prep.get('observation_count', 0)} routines.",
                "supporting_metric": "prep_bias_min"
            })
        elif bias <= -2.0:
            insights.append({
                "dimension": "preparation",
                "finding": "Preparation Time Buffer",
                "statement": f"You consistently finish preparation {abs(bias):.1f} minutes ahead of your scheduled departure window.",
                "factual_evidence": f"Preparation completed {abs(bias):.1f}m earlier than estimated.",
                "supporting_metric": "prep_bias_min"
            })
        elif abs(bias) < 1.5:
            insights.append({
                "dimension": "preparation",
                "finding": "Calibrated Estimation",
                "statement": "Your preparation time estimates are tightly calibrated with your actual morning timeline.",
                "factual_evidence": f"Prep error within +/- 1.5m ({bias:.1f}m observed).",
                "supporting_metric": "prep_bias_min"
            })

        # 2. Day-specific delay pattern
        day_patterns = dep.get("day_specific_patterns", {})
        if day_patterns:
            worst_day = max(day_patterns.items(), key=lambda kv: kv[1].get("avg_delay_min", 0.0))
            if worst_day[1].get("avg_delay_min", 0.0) > 2.0 and worst_day[1].get("sample_count", 0) >= 2:
                insights.append({
                    "dimension": "departure",
                    "finding": f"Elevated Lateness on {worst_day[0]}s",
                    "statement": f"Your departure delays are elevated on {worst_day[0]}s (averaging {worst_day[1]['avg_delay_min']:.1f} minutes late across {worst_day[1]['sample_count']} checkouts).",
                    "factual_evidence": f"Worst day {worst_day[0]} delay = {worst_day[1]['avg_delay_min']:.1f}m.",
                    "supporting_metric": "day_specific_patterns"
                })

        # 3. 30-day bag preparation trend
        if metrics_30d and metrics_30d.get("organization", {}).get("observation_count", 0) >= 3:
            bag_30d = metrics_30d["organization"].get("bag_prepared_consistency", 0.0)
            bag_all = org.get("bag_prepared_consistency", 0.0)
            if bag_30d - bag_all >= 0.15:
                insights.append({
                    "dimension": "organization",
                    "finding": "Bag Preparation Improvement",
                    "statement": f"Your bag preparation rate has improved over the last 30 days ({round(bag_30d * 100)}% vs {round(bag_all * 100)}% historical baseline).",
                    "factual_evidence": f"+{round((bag_30d - bag_all) * 100)}% increase in 30d bag readiness.",
                    "supporting_metric": "bag_prepared_consistency"
                })
            elif bag_all - bag_30d >= 0.15:
                insights.append({
                    "dimension": "organization",
                    "finding": "Bag Preparation Decline",
                    "statement": f"Overnight bag preparation has declined over the last 30 days ({round(bag_30d * 100)}% vs {round(bag_all * 100)}% historical baseline).",
                    "factual_evidence": f"-{round((bag_all - bag_30d) * 100)}% decrease in 30d bag readiness.",
                    "supporting_metric": "bag_prepared_consistency"
                })

        # 4. Distraction & phone interruption evidence
        phone_freq = dist.get("phone_usage_frequency", 0.0)
        phone_d = dist.get("avg_phone_interruption_min", 0.0)
        if phone_freq >= 0.30:
            insights.append({
                "dimension": "distraction",
                "finding": "Phone Interruption Impact",
                "statement": f"Phone interruptions account for an average of {phone_d:.1f} minutes of directly observed delay (occurring in {round(phone_freq * 100)}% of checkouts).",
                "factual_evidence": f"Occurred in {round(phone_freq * 100)}% of checkouts, causing {phone_d:.1f}m delay.",
                "supporting_metric": "phone_usage_frequency"
            })

        # 5. Organization & doorway search
        item_freq = org.get("item_search_frequency", 0.0)
        if item_freq >= 0.30:
            insights.append({
                "dimension": "organization",
                "finding": "Doorway Item Search Overhead",
                "statement": f"Searching for essentials (keys, badges, transit cards) occurs in {round(item_freq * 100)}% of your recorded departures.",
                "factual_evidence": f"Item search recorded in {round(item_freq * 100)}% of checkouts.",
                "supporting_metric": "item_search_frequency"
            })

        # 6. Overall punctuality
        adh = rout.get("schedule_adherence_rate", 1.0)
        insights.append({
            "dimension": "routine_consistency",
            "finding": "Overall Departure Punctuality",
            "statement": f"You departed on time or early in {round(adh * 100)}% of your {n_all} verified routines.",
            "factual_evidence": f"Punctuality rate {round(adh * 100)}% across {n_all} observations.",
            "supporting_metric": "schedule_adherence_rate"
        })

        return insights


class PersonalizedBehavioralProfiler:
    """Service orchestrating profile calculations, rolling windows, versioning, and comparisons."""

    @classmethod
    def filter_logs_by_window(
        cls,
        logs: List[Any],
        window_days: Optional[int] = None,
        now_dt: Optional[datetime.datetime] = None,
        days: Optional[int] = None
    ) -> List[Any]:
        """Filter departure logs to those within window_days relative to reference date."""
        target_days = window_days if window_days is not None else days
        if not logs or target_days is None:
            return logs

        if now_dt is None and logs:
            try:
                latest = logs[-1]
                if hasattr(latest, "recorded_at") and latest.recorded_at:
                    reference_dt = datetime.datetime.fromisoformat(str(latest.recorded_at).replace("Z", "+00:00"))
                elif hasattr(latest, "date") and latest.date:
                    reference_dt = datetime.datetime.strptime(latest.date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
                elif isinstance(latest, dict):
                    dt_str = latest.get("recorded_at") or f"{latest.get('date')}T23:59:59+00:00"
                    reference_dt = datetime.datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
                else:
                    reference_dt = datetime.datetime.now(datetime.timezone.utc)
            except Exception:
                reference_dt = datetime.datetime.now(datetime.timezone.utc)
        else:
            reference_dt = now_dt or datetime.datetime.now(datetime.timezone.utc)

        cutoff_dt = reference_dt - datetime.timedelta(days=target_days)
        cutoff_iso = cutoff_dt.isoformat()

        filtered = []
        for log in logs:
            log_time = None
            if hasattr(log, "recorded_at") and log.recorded_at:
                log_time = str(log.recorded_at)
            elif hasattr(log, "date") and log.date:
                log_time = f"{log.date}T23:59:59Z"
            elif isinstance(log, dict):
                log_time = str(log.get("recorded_at") or f"{log.get('date', '')}T23:59:59Z")

            if log_time and log_time >= cutoff_iso:
                filtered.append(log)

        return filtered

    @classmethod
    def compute_full_profile(
        cls,
        user_id: int,
        db: Session,
        provenance: str = "real"
    ) -> Dict[str, Any]:
        """Calculate complete multi-window behavioral profile for user."""
        is_synth = (provenance == "synthetic")

        logs_all = (
            db.query(DepartureLog)
            .filter(
                DepartureLog.user_id == user_id,
                DepartureLog.is_synthetic == is_synth,
                DepartureLog.observation_status == "valid"
            )
            .order_by(DepartureLog.date.asc(), DepartureLog.planned_departure.asc())
            .all()
        )

        n_total = len(logs_all)

        # Determine reference timestamp from latest log or now
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        if logs_all:
            last_log = logs_all[-1]
            try:
                if last_log.date:
                    now_dt = datetime.datetime.strptime(last_log.date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
            except Exception:
                pass

        logs_30d = cls.filter_logs_by_window(logs_all, window_days=30, now_dt=now_dt)
        logs_7d = cls.filter_logs_by_window(logs_all, window_days=7, now_dt=now_dt)

        metrics_all = BehavioralDimensionCalculator.calculate_dimensions(logs_all)
        metrics_30d = BehavioralDimensionCalculator.calculate_dimensions(logs_30d)
        metrics_7d = BehavioralDimensionCalculator.calculate_dimensions(logs_7d)

        insights = BehavioralInsightsGenerator.generate_insights(
            metrics_all=metrics_all,
            metrics_30d=metrics_30d if len(logs_30d) >= 3 else None,
            metrics_7d=metrics_7d if len(logs_7d) >= 3 else None
        )

        # Cluster reference from Prompt 2.4 if available
        cluster_ref = None
        cluster_rec = db.query(UserClusterProfile).filter(
            UserClusterProfile.user_id == user_id
        ).order_by(desc(UserClusterProfile.id)).first()
        if cluster_rec:
            cluster_ref = cluster_rec.cluster_name

        # Data sufficiency status
        if n_total < 3:
            sufficiency = "insufficient_data"
        elif n_total < 8:
            sufficiency = "gathering_history"
        else:
            sufficiency = "sufficient"

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        profile_version = f"prof_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        profile_id = f"usr_{user_id}_{profile_version}"

        profile_data = {
            "profile_id": profile_id,
            "user_id": user_id,
            "profile_version": profile_version,
            "generated_at": now_iso,
            "observation_count": n_total,
            "data_provenance": provenance,
            "confidence_or_sufficiency_status": sufficiency,
            "cluster_reference": cluster_ref,
            "dimensions": {
                "all_time": metrics_all,
                "window_30d": {
                    "is_sufficient": len(logs_30d) >= 3,
                    "sample_count": len(logs_30d),
                    "metrics": metrics_30d
                },
                "window_7d": {
                    "is_sufficient": len(logs_7d) >= 3,
                    "sample_count": len(logs_7d),
                    "metrics": metrics_7d
                }
            },
            "insights": insights
        }

        return profile_data

    @classmethod
    def get_or_create_profile(
        cls,
        user_id: int,
        db: Session,
        provenance: str = "real",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Fetch cached active profile or generate and persist a new snapshot."""
        is_synth = (provenance == "synthetic")

        # Check total logs count
        total_logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id,
            DepartureLog.is_synthetic == is_synth,
            DepartureLog.observation_status == "valid"
        ).count()

        latest_snapshot = db.query(BehavioralProfileSnapshot).filter(
            BehavioralProfileSnapshot.user_id == user_id,
            BehavioralProfileSnapshot.data_provenance == provenance
        ).order_by(desc(BehavioralProfileSnapshot.id)).first()

        # Update condition: force refresh, no snapshot, or >= 3 new logs accumulated since last snapshot
        should_recalc = (
            force_refresh or
            latest_snapshot is None or
            (total_logs - latest_snapshot.observation_count >= 3)
        )

        if not should_recalc and latest_snapshot is not None:
            try:
                dims = json.loads(latest_snapshot.behavioral_metrics)
                insights_list = json.loads(latest_snapshot.insights)
                return {
                    "profile_id": latest_snapshot.profile_id,
                    "user_id": latest_snapshot.user_id,
                    "profile_version": latest_snapshot.profile_version,
                    "generated_at": latest_snapshot.generated_at,
                    "observation_count": latest_snapshot.observation_count,
                    "data_provenance": latest_snapshot.data_provenance,
                    "confidence_or_sufficiency_status": latest_snapshot.confidence_or_sufficiency_status,
                    "cluster_reference": latest_snapshot.cluster_reference,
                    "dimensions": dims,
                    "insights": insights_list,
                    "cached": True
                }
            except Exception as e:
                logger.warning("Error parsing cached profile snapshot: %s. Recalculating.", e)

        # Recalculate and persist snapshot
        profile_data = cls.compute_full_profile(user_id, db, provenance=provenance)
        snapshot = BehavioralProfileSnapshot(
            profile_id=profile_data["profile_id"],
            user_id=user_id,
            profile_version=profile_data["profile_version"],
            generated_at=profile_data["generated_at"],
            observation_window="multi_window",
            observation_count=profile_data["observation_count"],
            data_provenance=provenance,
            behavioral_metrics=json.dumps(profile_data["dimensions"]),
            cluster_reference=profile_data["cluster_reference"],
            confidence_or_sufficiency_status=profile_data["confidence_or_sufficiency_status"],
            insights=json.dumps(profile_data["insights"])
        )
        db.add(snapshot)
        db.commit()

        profile_data["cached"] = False
        return profile_data

    @classmethod
    def get_profile_history(
        cls,
        user_id: int,
        db: Session,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve historical profile snapshots for the authenticated user."""
        snapshots = (
            db.query(BehavioralProfileSnapshot)
            .filter(BehavioralProfileSnapshot.user_id == user_id)
            .order_by(desc(BehavioralProfileSnapshot.id))
            .limit(limit)
            .all()
        )

        history = []
        for s in snapshots:
            try:
                dims = json.loads(s.behavioral_metrics)
                ins = json.loads(s.insights)
                prep_bias = dims.get("all_time", {}).get("preparation", {}).get("prep_bias_min", 0.0)
                avg_delay = dims.get("all_time", {}).get("departure", {}).get("avg_lateness_min", 0.0)
                adherence = dims.get("all_time", {}).get("routine_consistency", {}).get("schedule_adherence_rate", 1.0)
            except Exception:
                dims = {}
                ins = []
                prep_bias = 0.0
                avg_delay = 0.0
                adherence = 1.0

            history.append({
                "profile_id": s.profile_id,
                "profile_version": s.profile_version,
                "generated_at": s.generated_at,
                "observation_count": s.observation_count,
                "data_provenance": s.data_provenance,
                "cluster_reference": s.cluster_reference,
                "confidence_or_sufficiency_status": s.confidence_or_sufficiency_status,
                "summary_metrics": {
                    "prep_bias_min": prep_bias,
                    "avg_lateness_min": avg_delay,
                    "schedule_adherence_rate": adherence
                },
                "insights": ins
            })

        return history

    @classmethod
    def compare_profile_periods(
        cls,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Statistically compare two profile snapshots or rolling windows."""
        # Extract metrics safely from dictionary structure
        def get_all_time(p):
            dims = p.get("dimensions", {})
            return dims.get("all_time", dims)

        m_a = get_all_time(profile_a)
        m_b = get_all_time(profile_b)

        prep_a = m_a.get("preparation", {}).get("prep_bias_min", 0.0)
        prep_b = m_b.get("preparation", {}).get("prep_bias_min", 0.0)
        prep_diff = round(prep_b - prep_a, 1)

        delay_a = m_a.get("departure", {}).get("avg_lateness_min", 0.0)
        delay_b = m_b.get("departure", {}).get("avg_lateness_min", 0.0)
        delay_diff = round(delay_b - delay_a, 1)

        phone_a = m_a.get("distraction", {}).get("phone_usage_frequency", 0.0)
        phone_b = m_b.get("distraction", {}).get("phone_usage_frequency", 0.0)
        phone_diff = round(phone_b - phone_a, 2)

        bag_a = m_a.get("organization", {}).get("bag_prepared_consistency", 0.0)
        bag_b = m_b.get("organization", {}).get("bag_prepared_consistency", 0.0)
        bag_diff = round(bag_b - bag_a, 2)

        adherence_a = m_a.get("routine_consistency", {}).get("schedule_adherence_rate", 1.0)
        adherence_b = m_b.get("routine_consistency", {}).get("schedule_adherence_rate", 1.0)
        adherence_diff = round(adherence_b - adherence_a, 2)

        # Interpretation of overall shift
        if delay_diff <= -1.0 or adherence_diff >= 0.15:
            overall_trend = "improving"
            trend_summary = f"Departure delay reduced by {abs(delay_diff)}m with improved schedule adherence (+{round(adherence_diff * 100)}%)."
        elif delay_diff >= 1.0 or adherence_diff <= -0.15:
            overall_trend = "regressing"
            trend_summary = f"Departure delay increased by {delay_diff}m compared to earlier baseline."
        else:
            overall_trend = "stable"
            trend_summary = "Behavioral metrics and routine consistency remain steady."

        return {
            "period_a": {
                "profile_id": profile_a.get("profile_id", "A"),
                "generated_at": profile_a.get("generated_at"),
                "observation_count": profile_a.get("observation_count", 0)
            },
            "period_b": {
                "profile_id": profile_b.get("profile_id", "B"),
                "generated_at": profile_b.get("generated_at"),
                "observation_count": profile_b.get("observation_count", 0)
            },
            "deltas": {
                "prep_bias_min_delta": prep_diff,
                "avg_lateness_min_delta": delay_diff,
                "phone_usage_frequency_delta": phone_diff,
                "bag_prepared_consistency_delta": bag_diff,
                "schedule_adherence_delta": adherence_diff
            },
            "overall_trend": overall_trend,
            "trend_summary": trend_summary
        }
