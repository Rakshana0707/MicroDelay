"""Advanced Feature Engineering for MicroDelay.

This module develops leakage-safe calendar, historical behavioral, preparation,
travel, and behavioral trend features for departure delay prediction.

Data Leakage Prevention:
1. Chronological Ordering: All historical rolling and expanding statistics use strictly past observations (via shift(1)).
2. Exclusion of Targets: Current observation's actual departure time, actual prep duration, actual travel duration, and delay tags are NEVER used to predict that observation's delay.
3. Safe Fallbacks: Cold-start / initial observations use population baselines or safe neutral defaults.
4. Inference Equivalence: Exact same feature extraction pipeline applied to historical training records and live prediction requests.
"""

import math
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


@dataclass
class FeatureDefinition:
    """Documentation and metadata for engineered features."""
    name: str
    category: str  # 'calendar', 'behavioral', 'preparation', 'travel', 'trend'
    data_source: str
    calculation: str
    prediction_time_availability: str
    leakage_risk: str
    description: str


FEATURE_CATALOG: Dict[str, FeatureDefinition] = {
    # Calendar features
    "is_weekend": FeatureDefinition(
        name="is_weekend",
        category="calendar",
        data_source="Event / Planned departure date",
        calculation="1 if day_of_week in ['Saturday', 'Sunday'] else 0",
        prediction_time_availability="Always available before departure",
        leakage_risk="None (deterministic calendar property)",
        description="Indicates whether the departure occurs on a weekend."
    ),
    "is_monday": FeatureDefinition(
        name="is_monday",
        category="calendar",
        data_source="Event / Planned departure date",
        calculation="1 if day_of_week == 'Monday' else 0",
        prediction_time_availability="Always available before departure",
        leakage_risk="None (deterministic calendar property)",
        description="Captures Monday start-of-week lateness bottlenecks."
    ),
    "is_first_event_of_day": FeatureDefinition(
        name="is_first_event_of_day",
        category="calendar",
        data_source="User daily schedule",
        calculation="1 if event is the earliest scheduled event on that calendar date else 0",
        prediction_time_availability="Available from daily schedule",
        leakage_risk="None (derived from planned start times)",
        description="Indicates whether this routine is the first departure of the day."
    ),
    "event_time_of_day": FeatureDefinition(
        name="event_time_of_day",
        category="calendar",
        data_source="planned_departure",
        calculation="hour + minute / 60.0 of planned departure",
        prediction_time_availability="Available from scheduled routine",
        leakage_risk="None (based strictly on planned departure time)",
        description="Continuous decimal hour of planned departure (e.g., 8.5 for 08:30)."
    ),
    "month": FeatureDefinition(
        name="month",
        category="calendar",
        data_source="date",
        calculation="Calendar month (1-12)",
        prediction_time_availability="Always available",
        leakage_risk="None (derived from date)",
        description="Seasonal month indicator."
    ),
    "week_of_year": FeatureDefinition(
        name="week_of_year",
        category="calendar",
        data_source="date",
        calculation="ISO calendar week number (1-52)",
        prediction_time_availability="Always available",
        leakage_risk="None (derived from date)",
        description="Captures term, semester, or seasonal work cycles."
    ),

    # Preparation features
    "estimated_prep_duration": FeatureDefinition(
        name="estimated_prep_duration",
        category="preparation",
        data_source="User routine / event template",
        calculation="Direct user estimated preparation minutes",
        prediction_time_availability="Available before prep starts",
        leakage_risk="None (user plan)",
        description="Planned prep time in minutes."
    ),
    "bag_prepared_indicator": FeatureDefinition(
        name="bag_prepared_indicator",
        category="preparation",
        data_source="User toggle / voice confirmation",
        calculation="1 if bag/materials packed previous night else 0",
        prediction_time_availability="Recorded before starting morning preparation",
        leakage_risk="None (pre-departure state variable)",
        description="Indicates whether routines materials were packed the previous evening."
    ),
    "historical_prep_duration_deviation": FeatureDefinition(
        name="historical_prep_duration_deviation",
        category="preparation",
        data_source="estimated_prep_duration - user_average_prep_duration",
        calculation="estimated_prep - historical_avg_prep",
        prediction_time_availability="Available before departure",
        leakage_risk="Safe (historical average uses strictly past observations)",
        description="Measures how ambitious or conservative the current prep plan is relative to history."
    ),

    # Travel features
    "estimated_travel_duration": FeatureDefinition(
        name="estimated_travel_duration",
        category="travel",
        data_source="Event template",
        calculation="Planned travel duration in minutes",
        prediction_time_availability="Available before departure",
        leakage_risk="None (planned travel duration)",
        description="Expected transit time in minutes."
    ),
    "historical_travel_error": FeatureDefinition(
        name="historical_travel_error",
        category="travel",
        data_source="Historical departure logs",
        calculation="Historical mean of (actual_travel - estimated_travel) strictly prior to this event",
        prediction_time_availability="Available before departure",
        leakage_risk="Safe (computed on shifted past departures)",
        description="Average past transit underestimation error."
    ),
    "departure_time_bucket": FeatureDefinition(
        name="departure_time_bucket",
        category="travel",
        data_source="planned_departure",
        calculation="Binned into early_morning (<8), morning_peak (8-10), midday (10-14), afternoon (14-17), evening (>=17)",
        prediction_time_availability="Available before departure",
        leakage_risk="None (based strictly on planned departure)",
        description="Traffic and routine context category based on departure hour."
    ),

    # Historical behavioral features
    "user_average_prep_duration": FeatureDefinition(
        name="user_average_prep_duration",
        category="behavioral",
        data_source="Historical actual_prep_duration_min",
        calculation="Cumulative mean of actual prep duration on strictly previous events",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="User's true historical average prep duration."
    ),
    "user_average_prep_bias": FeatureDefinition(
        name="user_average_prep_bias",
        category="behavioral",
        data_source="Historical prep bias",
        calculation="Cumulative mean of (actual_prep - estimated_prep) on strictly previous events",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="Personal planning fallacy index: how many minutes the user tends to underestimate prep."
    ),
    "user_historical_delay_rate": FeatureDefinition(
        name="user_historical_delay_rate",
        category="behavioral",
        data_source="Historical total_delay_min",
        calculation="Fraction of strictly previous departures that were late (> 0 min)",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="Historical probability of late departure."
    ),
    "event_specific_average_delay": FeatureDefinition(
        name="event_specific_average_delay",
        category="behavioral",
        data_source="Historical total_delay_min by event_name",
        calculation="Historical average delay for this specific event strictly prior to this observation",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="Event-specific lateness tendency (e.g. higher delay for Labs than Seminars)."
    ),
    "day_specific_average_delay": FeatureDefinition(
        name="day_specific_average_delay",
        category="behavioral",
        data_source="Historical total_delay_min by day_of_week",
        calculation="Historical average delay for this day of week strictly prior to this observation",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="Day-of-week lateness tendency."
    ),
    "historical_phone_distraction_frequency": FeatureDefinition(
        name="historical_phone_distraction_frequency",
        category="behavioral",
        data_source="Historical tag_phone_usage",
        calculation="Fraction of strictly previous departures where phone distraction was tagged",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="User's historical phone distraction rate."
    ),
    "historical_item_search_frequency": FeatureDefinition(
        name="historical_item_search_frequency",
        category="behavioral",
        data_source="Historical tag_finding_items",
        calculation="Fraction of strictly previous departures where item searching was tagged",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="User's historical item search overhead rate."
    ),
    "historical_last_task_frequency": FeatureDefinition(
        name="historical_last_task_frequency",
        category="behavioral",
        data_source="Historical tag_one_last_task",
        calculation="Fraction of strictly previous departures where one last task was tagged",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="Frequency of last-minute chore/commit bottlenecks."
    ),

    # Behavioral trend features
    "recent_average_delay": FeatureDefinition(
        name="recent_average_delay",
        category="trend",
        data_source="Historical total_delay_min",
        calculation="Rolling mean of total_delay_min over the last 5 strictly prior departures",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="Recent departure delay momentum over the last 5 events."
    ),
    "recent_lateness_rate": FeatureDefinition(
        name="recent_lateness_rate",
        category="trend",
        data_source="Historical total_delay_min",
        calculation="Fraction of last 5 strictly prior departures that were late",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="Recent lateness frequency."
    ),
    "recent_prep_bias": FeatureDefinition(
        name="recent_prep_bias",
        category="trend",
        data_source="Historical prep bias",
        calculation="Rolling mean of (actual_prep - estimated_prep) over the last 5 strictly prior departures",
        prediction_time_availability="Available from past logs",
        leakage_risk="Strictly prevented via shift(1)",
        description="Recent planning fallacy trend."
    ),
    "change_in_recent_behavior": FeatureDefinition(
        name="change_in_recent_behavior",
        category="trend",
        data_source="recent_average_delay - user_average_delay",
        calculation="recent_average_delay - user_average_prep_duration",
        prediction_time_availability="Available from past logs",
        leakage_risk="Safe (both operands use past data only)",
        description="Behavioral delta indicating whether delays are worsening or improving."
    )
}


class HistoricalFeatureExtractor:
    """Extracts leakage-safe temporal and historical behavioral features.
    
    Guarantees:
      - Training rows are chronologically sorted.
      - Past statistics strictly use shift(1) so current row actuals/targets are never visible.
      - Inference rows apply user's historical summary up to the inference timestamp.
    """

    def __init__(self, rolling_window_size: int = 5):
        self.rolling_window_size = rolling_window_size

        # Population default fallbacks for cold-start (zero past history)
        self.default_prep_duration = 20.0
        self.default_prep_bias = 2.5
        self.default_delay_rate = 0.35
        self.default_average_delay = 3.0
        self.default_travel_error = 0.5
        self.default_phone_freq = 0.25
        self.default_finding_freq = 0.20
        self.default_last_task_freq = 0.30

    def _extract_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract deterministic calendar features from date and planned_departure."""
        res = df.copy()

        # Day of week helpers
        res["is_weekend"] = res["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
        res["is_monday"] = (res["day_of_week"] == "Monday").astype(int)

        # Date-based features
        if "date" in res.columns:
            date_series = pd.to_datetime(res["date"], errors="coerce")
            res["month"] = date_series.dt.month.fillna(datetime.date.today().month).astype(int)
            res["week_of_year"] = date_series.dt.isocalendar().week.fillna(datetime.date.today().isocalendar()[1]).astype(int)
        else:
            today = datetime.date.today()
            res["month"] = today.month
            res["week_of_year"] = today.isocalendar()[1]

        # Planned departure hour / time bucket
        if "planned_departure" in res.columns:
            def parse_hour(val):
                if not val or pd.isna(val):
                    return 8.5
                s = str(val).strip()
                try:
                    parts = s.split(":")
                    return int(parts[0]) + int(parts[1]) / 60.0
                except Exception:
                    return 8.5

            res["event_time_of_day"] = res["planned_departure"].apply(parse_hour)
        else:
            res["event_time_of_day"] = 8.5

        def get_time_bucket(hour: float) -> str:
            if hour < 8.0:
                return "early_morning"
            elif hour < 10.0:
                return "morning_peak"
            elif hour < 14.0:
                return "midday"
            elif hour < 17.0:
                return "afternoon"
            else:
                return "evening"

        res["departure_time_bucket"] = res["event_time_of_day"].apply(get_time_bucket)

        # First event of day
        if "date" in res.columns and "event_time_of_day" in res.columns:
            min_times = res.groupby("date")["event_time_of_day"].transform("min")
            res["is_first_event_of_day"] = (res["event_time_of_day"] == min_times).astype(int)
        else:
            res["is_first_event_of_day"] = 1

        return res

    def transform_training_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract leakage-safe historical and calendar features from historical DataFrame.
        
        CRITICAL: df must be sorted chronologically. All expanding/rolling operations
        use .shift(1) so an observation NEVER consumes its own actual values.
        """
        if df.empty:
            return pd.DataFrame()

        # 1. Sort chronologically by date and planned_departure
        sort_cols = [c for c in ["date", "planned_departure", "id"] if c in df.columns]
        if sort_cols:
            work_df = df.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            work_df = df.copy().reset_index(drop=True)

        # 2. Extract calendar features
        work_df = self._extract_calendar_features(work_df)

        # 3. Preparation baseline features
        work_df["estimated_prep_duration"] = work_df["estimated_prep_duration_min"].astype(float)
        work_df["estimated_travel_duration"] = work_df["estimated_travel_duration_min"].astype(float)
        work_df["bag_prepared_indicator"] = work_df["bag_prepared"].astype(int)

        # 4. Historical behavioral features (STRICTLY SHIFTED BY 1 TO PREVENT LEAKAGE)
        # Target / actual series shifted:
        past_actual_prep = work_df["actual_prep_duration_min"].shift(1) if "actual_prep_duration_min" in work_df.columns else pd.Series(index=work_df.index, dtype=float)
        past_est_prep = work_df["estimated_prep_duration_min"].shift(1)
        past_delay = work_df["total_delay_min"].shift(1) if "total_delay_min" in work_df.columns else pd.Series(index=work_df.index, dtype=float)
        past_is_late = pd.Series(
            np.where(past_delay.isna(), np.nan, (past_delay > 0).astype(float)),
            index=work_df.index
        )
        past_prep_bias = past_actual_prep - past_est_prep

        past_act_travel = work_df["actual_travel_duration_min"].shift(1) if "actual_travel_duration_min" in work_df.columns else pd.Series(index=work_df.index, dtype=float)
        past_est_travel = work_df["estimated_travel_duration_min"].shift(1)
        past_travel_error = past_act_travel - past_est_travel

        past_phone = work_df["tag_phone_usage"].shift(1).astype(float) if "tag_phone_usage" in work_df.columns else pd.Series(index=work_df.index, dtype=float)
        past_finding = work_df["tag_finding_items"].shift(1).astype(float) if "tag_finding_items" in work_df.columns else pd.Series(index=work_df.index, dtype=float)
        past_last_task = work_df["tag_one_last_task"].shift(1).astype(float) if "tag_one_last_task" in work_df.columns else pd.Series(index=work_df.index, dtype=float)

        # Cumulative expanding statistics (excluding current row)
        work_df["user_average_prep_duration"] = past_actual_prep.expanding(min_periods=1).mean().fillna(self.default_prep_duration)
        work_df["user_average_prep_bias"] = past_prep_bias.expanding(min_periods=1).mean().fillna(self.default_prep_bias)
        work_df["user_historical_delay_rate"] = past_is_late.expanding(min_periods=1).mean().fillna(self.default_delay_rate)
        work_df["historical_prep_duration_deviation"] = work_df["estimated_prep_duration"] - work_df["user_average_prep_duration"]
        work_df["historical_travel_error"] = past_travel_error.expanding(min_periods=1).mean().fillna(self.default_travel_error)

        work_df["historical_phone_distraction_frequency"] = past_phone.expanding(min_periods=1).mean().fillna(self.default_phone_freq)
        work_df["historical_item_search_frequency"] = past_finding.expanding(min_periods=1).mean().fillna(self.default_finding_freq)
        work_df["historical_last_task_frequency"] = past_last_task.expanding(min_periods=1).mean().fillna(self.default_last_task_freq)

        # Event-specific and Day-specific average delays (shifted)
        # Compute expanding mean per group using shifted values
        work_df["event_specific_average_delay"] = work_df.groupby("event_name")["total_delay_min"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        ).fillna(self.default_average_delay)

        work_df["day_specific_average_delay"] = work_df.groupby("day_of_week")["total_delay_min"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        ).fillna(self.default_average_delay)

        # 5. Behavioral trend features (Rolling window of last K events)
        work_df["recent_average_delay"] = past_delay.rolling(
            window=self.rolling_window_size, min_periods=1
        ).mean().fillna(self.default_average_delay)

        work_df["recent_lateness_rate"] = past_is_late.rolling(
            window=self.rolling_window_size, min_periods=1
        ).mean().fillna(self.default_delay_rate)

        work_df["recent_prep_bias"] = past_prep_bias.rolling(
            window=self.rolling_window_size, min_periods=1
        ).mean().fillna(self.default_prep_bias)

        # Change in recent behavior: recent delay trend vs long-term delay trend
        user_long_term_delay = past_delay.expanding(min_periods=1).mean().fillna(self.default_average_delay)
        work_df["change_in_recent_behavior"] = work_df["recent_average_delay"] - user_long_term_delay

        return work_df

    def compute_user_historical_state(self, history_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute the cumulative historical summary state for a user up to the current moment.
        
        Used to enrich live inference requests (/api/predict) with exact historical behavior.
        """
        if history_df.empty or len(history_df) == 0:
            return {
                "user_average_prep_duration": self.default_prep_duration,
                "user_average_prep_bias": self.default_prep_bias,
                "user_historical_delay_rate": self.default_delay_rate,
                "user_average_delay": self.default_average_delay,
                "historical_travel_error": self.default_travel_error,
                "historical_phone_distraction_frequency": self.default_phone_freq,
                "historical_item_search_frequency": self.default_finding_freq,
                "historical_last_task_frequency": self.default_last_task_freq,
                "recent_average_delay": self.default_average_delay,
                "recent_lateness_rate": self.default_delay_rate,
                "recent_prep_bias": self.default_prep_bias,
                "change_in_recent_behavior": 0.0,
                "event_means": {},
                "day_means": {}
            }

        sort_cols = [c for c in ["date", "planned_departure", "id"] if c in history_df.columns]
        h_df = history_df.sort_values(by=sort_cols).reset_index(drop=True) if sort_cols else history_df.copy()

        # Cumulative metrics
        avg_prep = float(h_df["actual_prep_duration_min"].mean()) if "actual_prep_duration_min" in h_df.columns and not h_df["actual_prep_duration_min"].isna().all() else self.default_prep_duration
        
        if "actual_prep_duration_min" in h_df.columns and "estimated_prep_duration_min" in h_df.columns:
            prep_bias = float((h_df["actual_prep_duration_min"] - h_df["estimated_prep_duration_min"]).mean())
        else:
            prep_bias = self.default_prep_bias

        if "total_delay_min" in h_df.columns:
            delays = h_df["total_delay_min"].dropna()
            delay_rate = float((delays > 0).mean()) if len(delays) > 0 else self.default_delay_rate
            avg_delay = float(delays.mean()) if len(delays) > 0 else self.default_average_delay
            event_means = h_df.groupby("event_name")["total_delay_min"].mean().to_dict()
            day_means = h_df.groupby("day_of_week")["total_delay_min"].mean().to_dict()

            recent_slice = delays.tail(self.rolling_window_size)
            recent_delay = float(recent_slice.mean()) if len(recent_slice) > 0 else avg_delay
            recent_late_rate = float((recent_slice > 0).mean()) if len(recent_slice) > 0 else delay_rate
        else:
            delay_rate = self.default_delay_rate
            avg_delay = self.default_average_delay
            event_means = {}
            day_means = {}
            recent_delay = self.default_average_delay
            recent_late_rate = self.default_delay_rate

        if "actual_travel_duration_min" in h_df.columns and "estimated_travel_duration_min" in h_df.columns:
            travel_error = float((h_df["actual_travel_duration_min"] - h_df["estimated_travel_duration_min"]).mean())
        else:
            travel_error = self.default_travel_error

        phone_freq = float(h_df["tag_phone_usage"].mean()) if "tag_phone_usage" in h_df.columns else self.default_phone_freq
        finding_freq = float(h_df["tag_finding_items"].mean()) if "tag_finding_items" in h_df.columns else self.default_finding_freq
        last_task_freq = float(h_df["tag_one_last_task"].mean()) if "tag_one_last_task" in h_df.columns else self.default_last_task_freq

        if "actual_prep_duration_min" in h_df.columns and "estimated_prep_duration_min" in h_df.columns:
            recent_bias_slice = (h_df["actual_prep_duration_min"] - h_df["estimated_prep_duration_min"]).tail(self.rolling_window_size)
            recent_bias = float(recent_bias_slice.mean()) if len(recent_bias_slice) > 0 else prep_bias
        else:
            recent_bias = self.default_prep_bias

        change_in_behavior = recent_delay - avg_delay

        return {
            "user_average_prep_duration": avg_prep,
            "user_average_prep_bias": prep_bias,
            "user_historical_delay_rate": delay_rate,
            "user_average_delay": avg_delay,
            "historical_travel_error": travel_error,
            "historical_phone_distraction_frequency": phone_freq,
            "historical_item_search_frequency": finding_freq,
            "historical_last_task_frequency": last_task_freq,
            "recent_average_delay": recent_delay,
            "recent_lateness_rate": recent_late_rate,
            "recent_prep_bias": recent_bias,
            "change_in_recent_behavior": change_in_behavior,
            "event_means": event_means,
            "day_means": day_means
        }

    def enrich_inference_record(
        self,
        query: Dict[str, Any],
        user_history_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Enrich a live inference request with matching calendar and historical behavioral features."""
        enriched = dict(query)
        today = datetime.date.today()

        day_name = str(query.get("day_of_week", today.strftime("%A"))).capitalize()
        event_name = str(query.get("event_name", "General Routine"))
        est_prep = float(query.get("estimated_prep_duration_min", query.get("estimated_prep_duration", 20.0)))
        est_travel = float(query.get("estimated_travel_duration_min", query.get("estimated_travel_duration", 20.0)))
        bag_prep = 1 if bool(query.get("bag_prepared", False)) else 0

        # Calendar features
        enriched["day_of_week"] = day_name
        enriched["is_weekend"] = 1 if day_name in ["Saturday", "Sunday"] else 0
        enriched["is_monday"] = 1 if day_name == "Monday" else 0
        enriched["month"] = int(query.get("month", today.month))
        enriched["week_of_year"] = int(query.get("week_of_year", today.isocalendar()[1]))

        # Time of day
        planned_dep = query.get("planned_departure")
        if planned_dep:
            try:
                parts = str(planned_dep).split(":")
                hour_float = int(parts[0]) + int(parts[1]) / 60.0
            except Exception:
                hour_float = 8.5
        else:
            hour_float = float(query.get("event_time_of_day", 8.5))

        enriched["event_time_of_day"] = hour_float
        if hour_float < 8.0:
            enriched["departure_time_bucket"] = "early_morning"
        elif hour_float < 10.0:
            enriched["departure_time_bucket"] = "morning_peak"
        elif hour_float < 14.0:
            enriched["departure_time_bucket"] = "midday"
        elif hour_float < 17.0:
            enriched["departure_time_bucket"] = "afternoon"
        else:
            enriched["departure_time_bucket"] = "evening"

        enriched["is_first_event_of_day"] = int(query.get("is_first_event_of_day", 1))

        # Preparation & Travel inputs
        enriched["estimated_prep_duration"] = est_prep
        enriched["estimated_prep_duration_min"] = est_prep
        enriched["estimated_travel_duration"] = est_travel
        enriched["estimated_travel_duration_min"] = est_travel
        enriched["bag_prepared_indicator"] = bag_prep
        enriched["bag_prepared"] = bag_prep

        # Historical stats from user history
        state = self.compute_user_historical_state(
            user_history_df if user_history_df is not None else pd.DataFrame()
        )

        enriched["user_average_prep_duration"] = state["user_average_prep_duration"]
        enriched["user_average_prep_bias"] = state["user_average_prep_bias"]
        enriched["user_historical_delay_rate"] = state["user_historical_delay_rate"]
        enriched["historical_prep_duration_deviation"] = est_prep - state["user_average_prep_duration"]
        enriched["historical_travel_error"] = state["historical_travel_error"]

        enriched["historical_phone_distraction_frequency"] = state["historical_phone_distraction_frequency"]
        enriched["historical_item_search_frequency"] = state["historical_item_search_frequency"]
        enriched["historical_last_task_frequency"] = state["historical_last_task_frequency"]

        enriched["event_specific_average_delay"] = state["event_means"].get(event_name, state["user_average_delay"])
        enriched["day_specific_average_delay"] = state["day_means"].get(day_name, state["user_average_delay"])

        enriched["recent_average_delay"] = state["recent_average_delay"]
        enriched["recent_lateness_rate"] = state["recent_lateness_rate"]
        enriched["recent_prep_bias"] = state["recent_prep_bias"]
        enriched["change_in_recent_behavior"] = state["change_in_recent_behavior"]

        return enriched


# Explicit feature sets for comparison and benchmarking
ORIGINAL_FEATURE_COLS = [
    "importance",
    "day_of_week",
    "event_name",
    "bag_prepared",
    "estimated_prep_duration_min",
    "estimated_travel_duration_min"
]

ENGINEERED_NUMERICAL_COLS = [
    "estimated_prep_duration",
    "estimated_travel_duration",
    "bag_prepared_indicator",
    "is_weekend",
    "is_monday",
    "is_first_event_of_day",
    "event_time_of_day",
    "month",
    "week_of_year",
    "user_average_prep_duration",
    "user_average_prep_bias",
    "user_historical_delay_rate",
    "event_specific_average_delay",
    "day_specific_average_delay",
    "historical_phone_distraction_frequency",
    "historical_item_search_frequency",
    "historical_last_task_frequency",
    "historical_prep_duration_deviation",
    "historical_travel_error",
    "recent_average_delay",
    "recent_lateness_rate",
    "recent_prep_bias",
    "change_in_recent_behavior"
]

ENGINEERED_CATEGORICAL_COLS = [
    "event_name",
    "day_of_week",
    "importance",
    "departure_time_bucket"
]

ALL_ENGINEERED_FEATURE_COLS = ENGINEERED_NUMERICAL_COLS + ENGINEERED_CATEGORICAL_COLS


def compare_feature_sets(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Compare predictive accuracy of the original feature set vs. engineered feature set.
    
    Uses identical chronological holdout partitioning to guarantee a rigorous,
    unbiased evaluation.
    """
    if len(df) < 10:
        return {
            "eligible": False,
            "message": "At least 10 observations required for temporal comparison."
        }

    # Chronological sort
    sort_cols = [c for c in ["date", "planned_departure", "id"] if c in df.columns]
    sorted_df = df.sort_values(by=sort_cols).reset_index(drop=True) if sort_cols else df.copy()

    split_idx = int(len(sorted_df) * (1.0 - test_size))
    train_raw = sorted_df.iloc[:split_idx].copy()
    test_raw = sorted_df.iloc[split_idx:].copy()

    # 1. Original Feature Set Baseline
    try:
        from ml.preprocessing import MicroDelayPreprocessor
    except ModuleNotFoundError:
        from backend.ml.preprocessing import MicroDelayPreprocessor
    orig_preprocessor = MicroDelayPreprocessor()
    X_train_orig, y_train_orig, _ = orig_preprocessor.fit_transform(train_raw)
    X_test_orig = orig_preprocessor.transform(test_raw)
    y_test = test_raw["total_delay_min"].values.astype(float)

    rf_orig = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf_orig.fit(X_train_orig, y_train_orig)
    preds_orig = rf_orig.predict(X_test_orig)
    mae_orig = float(mean_absolute_error(y_test, preds_orig))
    rmse_orig = float(root_mean_squared_error(y_test, preds_orig))

    # 2. Engineered Feature Set Pipeline
    extractor = HistoricalFeatureExtractor()
    full_enriched = extractor.transform_training_dataframe(sorted_df)
    train_enriched = full_enriched.iloc[:split_idx].copy()
    test_enriched = full_enriched.iloc[split_idx:].copy()

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    eng_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), ENGINEERED_NUMERICAL_COLS),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), ENGINEERED_CATEGORICAL_COLS)
        ],
        remainder="drop"
    )

    X_train_eng = eng_preprocessor.fit_transform(train_enriched[ALL_ENGINEERED_FEATURE_COLS])
    y_train_eng = train_enriched["total_delay_min"].values.astype(float)
    X_test_eng = eng_preprocessor.transform(test_enriched[ALL_ENGINEERED_FEATURE_COLS])

    rf_eng = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf_eng.fit(X_train_eng, y_train_eng)
    preds_eng = rf_eng.predict(X_test_eng)
    mae_eng = float(mean_absolute_error(y_test, preds_eng))
    rmse_eng = float(root_mean_squared_error(y_test, preds_eng))

    improvement_pct = ((mae_orig - mae_eng) / mae_orig * 100.0) if mae_orig > 0 else 0.0

    return {
        "eligible": True,
        "train_samples": len(train_raw),
        "test_samples": len(test_raw),
        "original_features": {
            "num_features": len(ORIGINAL_FEATURE_COLS),
            "mae": round(mae_orig, 3),
            "rmse": round(rmse_orig, 3)
        },
        "engineered_features": {
            "num_features": len(ALL_ENGINEERED_FEATURE_COLS),
            "mae": round(mae_eng, 3),
            "rmse": round(rmse_eng, 3)
        },
        "mae_reduction_pct": round(improvement_pct, 2),
        "recommendation": "Retain engineered features" if mae_eng <= mae_orig else "Retain original features"
    }
