"""Automated tests for leakage-safe feature engineering pipeline in MicroDelay."""

import datetime
import pytest
import pandas as pd
import numpy as np

from ml.features import (
    HistoricalFeatureExtractor,
    FEATURE_CATALOG,
    ORIGINAL_FEATURE_COLS,
    ALL_ENGINEERED_FEATURE_COLS,
    ENGINEERED_NUMERICAL_COLS,
    ENGINEERED_CATEGORICAL_COLS,
    compare_feature_sets,
)


@pytest.fixture
def sample_chronological_df():
    """Create a controlled 10-observation sequence for testing temporal features."""
    dates = [
        "2026-08-01", "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04",
        "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08", "2026-08-09"
    ]
    # 2026-08-01 was Saturday, 08-02 Sunday, 08-03 Monday, etc.
    days = [
        "Saturday", "Saturday", "Sunday", "Monday", "Tuesday",
        "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    times = [
        "08:00", "14:00", "09:30", "08:15", "10:00",
        "07:30", "12:00", "16:45", "11:00", "18:00"
    ]
    events = [
        "Weekend Kickoff", "Library Study", "Library Study", "ML Lab", "Theory Lecture",
        "Morning Assembly", "Seminar", "Design Sync", "Library Study", "Weekend Kickoff"
    ]
    # Actual values
    actual_preps = [25.0, 30.0, 15.0, 40.0, 20.0, 22.0, 35.0, 18.0, 25.0, 28.0]
    estimated_preps = [20.0, 20.0, 20.0, 25.0, 20.0, 20.0, 30.0, 20.0, 20.0, 20.0]
    actual_travels = [22.0, 20.0, 15.0, 30.0, 25.0, 15.0, 20.0, 25.0, 18.0, 22.0]
    estimated_travels = [20.0, 20.0, 15.0, 20.0, 25.0, 15.0, 20.0, 20.0, 20.0, 20.0]
    delays = [5.0, 10.0, 0.0, 15.0, 0.0, 2.0, 5.0, 0.0, 5.0, 8.0]
    tags_phone = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
    tags_finding = [0, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    tags_last_task = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]

    rows = []
    for i in range(10):
        rows.append({
            "id": i + 1,
            "user_id": 1,
            "event_name": events[i],
            "day_of_week": days[i],
            "date": dates[i],
            "planned_departure": times[i],
            "importance": "high" if i % 2 == 0 else "medium",
            "bag_prepared": 1 if i % 3 == 0 else 0,
            "estimated_prep_duration_min": estimated_preps[i],
            "estimated_travel_duration_min": estimated_travels[i],
            "actual_prep_duration_min": actual_preps[i],
            "actual_travel_duration_min": actual_travels[i],
            "total_delay_min": delays[i],
            "tag_phone_usage": tags_phone[i],
            "tag_finding_items": tags_finding[i],
            "tag_one_last_task": tags_last_task[i],
            "tag_waiting": 0,
            "provenance": "synthetic",
            "is_synthetic": 1
        })
    return pd.DataFrame(rows)


def test_feature_catalog_completeness():
    """Verify that all engineered features are comprehensively cataloged."""
    assert len(FEATURE_CATALOG) >= 20
    for name, fdef in FEATURE_CATALOG.items():
        assert fdef.name == name
        assert fdef.category in ["calendar", "behavioral", "preparation", "travel", "trend"]
        assert len(fdef.data_source) > 0
        assert len(fdef.calculation) > 0
        assert len(fdef.prediction_time_availability) > 0
        assert len(fdef.leakage_risk) > 0


def test_calendar_features(sample_chronological_df):
    """Verify calendar feature calculation: weekend, monday, month, hour, time bucket."""
    extractor = HistoricalFeatureExtractor()
    enriched = extractor.transform_training_dataframe(sample_chronological_df)

    # 2026-08-01 is Saturday (weekend)
    assert enriched.loc[0, "is_weekend"] == 1
    assert enriched.loc[0, "is_monday"] == 0
    # 2026-08-03 is Monday
    assert enriched.loc[3, "is_weekend"] == 0
    assert enriched.loc[3, "is_monday"] == 1

    # Month and week
    assert (enriched["month"] == 8).all()
    assert (enriched["week_of_year"] > 0).all()

    # Time of day: 08:00 -> 8.0, 14:00 -> 14.0
    assert enriched.loc[0, "event_time_of_day"] == 8.0
    assert enriched.loc[1, "event_time_of_day"] == 14.0

    # Buckets: 07:30 -> early_morning, 08:00 -> morning_peak, 12:00 -> midday, 16:45 -> afternoon, 18:00 -> evening
    assert enriched.loc[5, "departure_time_bucket"] == "early_morning"
    assert enriched.loc[0, "departure_time_bucket"] == "morning_peak"
    assert enriched.loc[6, "departure_time_bucket"] == "midday"
    assert enriched.loc[7, "departure_time_bucket"] == "afternoon"
    assert enriched.loc[9, "departure_time_bucket"] == "evening"

    # First event of day: on 2026-08-01, 08:00 is first, 14:00 is not
    assert enriched.loc[0, "is_first_event_of_day"] == 1
    assert enriched.loc[1, "is_first_event_of_day"] == 0


def test_no_lookahead_data_leakage(sample_chronological_df):
    """Strictly verify that future or current-row target values do NOT leak into features.
    
    If we change row 4's actual delay, actual prep, or tags, row 4's engineered input
    features MUST remain 100% IDENTICAL because historical statistics are shifted by 1.
    """
    extractor = HistoricalFeatureExtractor()
    
    # Run on original data
    df1 = sample_chronological_df.copy()
    enriched1 = extractor.transform_training_dataframe(df1)
    
    # Run on data where row 4's actual outcomes are drastically changed
    df2 = sample_chronological_df.copy()
    df2.loc[4, "total_delay_min"] = 999.0
    df2.loc[4, "actual_prep_duration_min"] = 350.0
    df2.loc[4, "actual_travel_duration_min"] = 400.0
    df2.loc[4, "tag_phone_usage"] = 1
    df2.loc[4, "tag_finding_items"] = 1
    enriched2 = extractor.transform_training_dataframe(df2)

    # For row 4, every single historical behavioral and trend feature MUST match identically
    leakage_sensitive_cols = [
        "user_average_prep_duration",
        "user_average_prep_bias",
        "user_historical_delay_rate",
        "historical_prep_duration_deviation",
        "historical_travel_error",
        "historical_phone_distraction_frequency",
        "historical_item_search_frequency",
        "historical_last_task_frequency",
        "recent_average_delay",
        "recent_lateness_rate",
        "recent_prep_bias",
        "change_in_recent_behavior"
    ]
    for col in leakage_sensitive_cols:
        val1 = enriched1.loc[4, col]
        val2 = enriched2.loc[4, col]
        assert np.isclose(val1, val2), f"Leakage detected in {col}: {val1} != {val2}"

    # Row 5 (the subsequent row) SHOULD reflect the change from row 4
    assert enriched2.loc[5, "user_average_prep_duration"] != enriched1.loc[5, "user_average_prep_duration"]


def test_cold_start_fallback_defaults(sample_chronological_df):
    """Verify that row 0 (cold start, zero past observations) assumes default fallbacks."""
    extractor = HistoricalFeatureExtractor()
    enriched = extractor.transform_training_dataframe(sample_chronological_df)

    # Row 0 has no preceding rows, so it should equal the default fallbacks
    assert enriched.loc[0, "user_average_prep_duration"] == extractor.default_prep_duration
    assert enriched.loc[0, "user_average_prep_bias"] == extractor.default_prep_bias
    assert enriched.loc[0, "user_historical_delay_rate"] == extractor.default_delay_rate
    assert enriched.loc[0, "recent_average_delay"] == extractor.default_average_delay
    assert enriched.loc[0, "recent_lateness_rate"] == extractor.default_delay_rate


def test_enrich_inference_record():
    """Verify that live inference enrichment computes identical features matching training schema."""
    extractor = HistoricalFeatureExtractor()
    
    # 1. Cold-start query without history
    query = {
        "event_name": "ML Lab",
        "day_of_week": "Monday",
        "importance": "high",
        "bag_prepared": True,
        "estimated_prep_duration_min": 25.0,
        "estimated_travel_duration_min": 20.0,
        "planned_departure": "08:30"
    }
    enriched = extractor.enrich_inference_record(query, user_history_df=None)
    
    assert enriched["is_weekend"] == 0
    assert enriched["is_monday"] == 1
    assert enriched["event_time_of_day"] == 8.5
    assert enriched["departure_time_bucket"] == "morning_peak"
    assert enriched["user_average_prep_duration"] == extractor.default_prep_duration
    assert enriched["bag_prepared_indicator"] == 1

    # 2. Query with user history
    history_rows = [
        {"total_delay_min": 10.0, "actual_prep_duration_min": 30.0, "estimated_prep_duration_min": 20.0,
         "actual_travel_duration_min": 20.0, "estimated_travel_duration_min": 20.0,
         "tag_phone_usage": 1, "tag_finding_items": 0, "tag_one_last_task": 0,
         "event_name": "ML Lab", "day_of_week": "Monday"},
        {"total_delay_min": 0.0, "actual_prep_duration_min": 20.0, "estimated_prep_duration_min": 20.0,
         "actual_travel_duration_min": 20.0, "estimated_travel_duration_min": 20.0,
         "tag_phone_usage": 0, "tag_finding_items": 0, "tag_one_last_task": 0,
         "event_name": "ML Lab", "day_of_week": "Monday"}
    ]
    history_df = pd.DataFrame(history_rows)
    enriched_with_hist = extractor.enrich_inference_record(query, user_history_df=history_df)
    
    # Average prep = (30 + 20) / 2 = 25.0
    assert enriched_with_hist["user_average_prep_duration"] == 25.0
    # Average delay = (10 + 0) / 2 = 5.0
    assert enriched_with_hist["recent_average_delay"] == 5.0
    # Phone usage = 1/2 = 0.5
    assert enriched_with_hist["historical_phone_distraction_frequency"] == 0.5


def test_compare_feature_sets(sample_chronological_df):
    """Verify that feature set comparison executes without errors and evaluates MAE/RMSE."""
    res = compare_feature_sets(sample_chronological_df, test_size=0.2)
    assert res["eligible"] is True
    assert "original_features" in res
    assert "engineered_features" in res
    assert "mae" in res["original_features"]
    assert "mae" in res["engineered_features"]
    assert "mae_reduction_pct" in res
    assert res["original_features"]["num_features"] == len(ORIGINAL_FEATURE_COLS)
    assert res["engineered_features"]["num_features"] == len(ALL_ENGINEERED_FEATURE_COLS)
