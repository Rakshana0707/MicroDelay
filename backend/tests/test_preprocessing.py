import os
import sys
import numpy as np
import pandas as pd
import pytest
import joblib

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from ml.preprocessing import (
    ValidationConfig,
    validate_single_observation,
    validate_observations_dataframe,
    MicroDelayPreprocessor,
    create_feature_preprocessor,
    ALL_INPUT_FEATURE_COLS
)


def test_missing_values_imputation():
    """Test that missing values in imputable fields are cleanly imputed without crashing."""
    raw_record = {
        "user_id": 1,
        "event_name": "Math Lecture",
        "day_of_week": "Monday",
        "importance": None,                     # Missing categorical -> should default to 'medium'
        "estimated_prep_duration_min": None,   # Missing numerical -> imputed
        "estimated_travel_duration_min": 15.0,
        "bag_prepared": None                    # Missing bool -> default 0
    }
    res = validate_single_observation(raw_record)
    assert res.is_valid is True
    cleaned = res.cleaned_record
    assert cleaned["importance"] == "medium"
    assert cleaned["estimated_prep_duration_min"] == 20.0
    assert cleaned["bag_prepared"] == 0


def test_invalid_durations_rejection():
    """Test that negative or physically impossible durations are categorized as invalid and rejected."""
    # Negative prep duration
    neg_prep = {
        "user_id": 1,
        "event_name": "Lab",
        "day_of_week": "Tuesday",
        "estimated_prep_duration_min": -15.0,
        "estimated_travel_duration_min": 20.0
    }
    res_neg = validate_single_observation(neg_prep)
    assert res_neg.is_valid is False
    assert res_neg.status == "invalid"
    assert any("Negative estimated_prep_duration_min" in issue for issue in res_neg.issues)

    # Implausible prep duration (> 720 mins)
    huge_prep = {
        "user_id": 1,
        "event_name": "Lab",
        "day_of_week": "Tuesday",
        "estimated_prep_duration_min": 5000.0,
        "estimated_travel_duration_min": 20.0
    }
    res_huge = validate_single_observation(huge_prep)
    assert res_huge.is_valid is False
    assert res_huge.status == "invalid"


def test_unknown_events_handling():
    """Test that custom user-defined events (not in initial vocabulary) are handled gracefully."""
    train_data = pd.DataFrame([
        {"event_name": "ML Lab", "day_of_week": "Monday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 1, "total_delay_min": 5.0},
        {"event_name": "Seminar", "day_of_week": "Tuesday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 15.0, "bag_prepared": 0, "total_delay_min": 2.0},
        {"event_name": "ML Lab", "day_of_week": "Wednesday", "importance": "high", "estimated_prep_duration_min": 25.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 1, "total_delay_min": 8.0},
        {"event_name": "Seminar", "day_of_week": "Thursday", "importance": "low", "estimated_prep_duration_min": 10.0, "estimated_travel_duration_min": 10.0, "bag_prepared": 1, "total_delay_min": 0.0},
        {"event_name": "Math", "day_of_week": "Friday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 0, "total_delay_min": 3.0},
        {"event_name": "Math", "day_of_week": "Monday", "importance": "medium", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 1, "total_delay_min": 4.0},
        {"event_name": "Seminar", "day_of_week": "Wednesday", "importance": "high", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 15.0, "bag_prepared": 1, "total_delay_min": 1.0},
        {"event_name": "ML Lab", "day_of_week": "Friday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 0, "total_delay_min": 12.0}
    ])

    preprocessor = MicroDelayPreprocessor()
    preprocessor.fit(train_data)

    # Transform inference row with novel/unseen event name
    novel_row = {
        "event_name": "Custom Guitar Lesson",  # Never seen in training!
        "day_of_week": "Sunday",               # Unseen day
        "importance": "high",
        "estimated_prep_duration_min": 25.0,
        "estimated_travel_duration_min": 30.0,
        "bag_prepared": 1
    }
    X_novel = preprocessor.transform_single_inference(novel_row)
    assert X_novel.shape[0] == 1
    assert X_novel.shape[1] == len(preprocessor.feature_names_)
    assert not np.isnan(X_novel).any()


def test_duplicate_observations_removal():
    """Test that duplicate departures are properly identified and deduplicated."""
    df_with_dups = pd.DataFrame([
        {"user_id": 1, "date": "2026-11-16", "planned_departure": "08:30:00", "event_name": "ML Lab", "day_of_week": "Monday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 1, "total_delay_min": 5.0},
        {"user_id": 1, "date": "2026-11-16", "planned_departure": "08:30:00", "event_name": "ML Lab", "day_of_week": "Monday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 1, "total_delay_min": 6.0}, # Duplicate
        {"user_id": 1, "date": "2026-11-17", "planned_departure": "09:00:00", "event_name": "Seminar", "day_of_week": "Tuesday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 15.0, "bag_prepared": 0, "total_delay_min": 2.0}
    ])

    clean_df, report = validate_observations_dataframe(df_with_dups, drop_duplicates=True)
    assert report.duplicate_records == 1
    assert len(clean_df) == 2


def test_extreme_but_valid_values_retained():
    """Ensure that legitimate large delays (e.g. 75 min prep, 90 min delay) are retained as potential outliers rather than dropped."""
    extreme_record = {
        "user_id": 1,
        "event_name": "Special Review",
        "day_of_week": "Thursday",
        "importance": "high",
        "estimated_prep_duration_min": 140.0,  # Legitimate but long prep
        "estimated_travel_duration_min": 25.0,
        "bag_prepared": 0,
        "total_delay_min": 85.0                # Legitimate major delay
    }

    res = validate_single_observation(extreme_record)
    assert res.is_valid is True
    assert res.status == "potential_outlier"
    assert any("Extreme estimated_prep_duration_min" in issue for issue in res.issues)
    assert any("Substantial delay observation" in issue for issue in res.issues)

    df = pd.DataFrame([extreme_record])
    clean_df, report = validate_observations_dataframe(df, drop_invalid=True)
    assert len(clean_df) == 1, "Valid unusual observations must NOT be removed"
    assert report.potential_outliers == 1
    assert report.invalid_records == 0


def test_training_inference_feature_consistency():
    """Verify that train-fitted preprocessor transforms inference inputs into identical feature structures."""
    train_df = pd.DataFrame([
        {"event_name": "Standup", "day_of_week": "Monday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 1, "total_delay_min": 4.0},
        {"event_name": "Code Review", "day_of_week": "Tuesday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 0, "total_delay_min": 2.0},
        {"event_name": "Deploy", "day_of_week": "Friday", "importance": "low", "estimated_prep_duration_min": 10.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 1, "total_delay_min": 0.0},
        {"event_name": "Standup", "day_of_week": "Wednesday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 1, "total_delay_min": 6.0},
        {"event_name": "Code Review", "day_of_week": "Thursday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 1, "total_delay_min": 1.0},
        {"event_name": "Standup", "day_of_week": "Friday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 0, "total_delay_min": 8.0},
        {"event_name": "Design Sync", "day_of_week": "Wednesday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 1, "total_delay_min": 3.0},
        {"event_name": "Deploy", "day_of_week": "Monday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 1, "total_delay_min": 5.0}
    ])

    preprocessor = MicroDelayPreprocessor()
    X_train, y_train, _ = preprocessor.fit_transform(train_df)
    assert X_train.shape[0] == 8
    num_features = X_train.shape[1]

    # Test inference transforms on single rows with different combinations
    test_cases = [
        {"event_name": "Standup", "day_of_week": "Monday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 25.0, "bag_prepared": True},
        {"event_name": "Code Review", "day_of_week": "Tuesday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 25.0, "bag_prepared": False},
        {"event_name": "Unknown Meeting", "day_of_week": "Sunday", "importance": "low", "estimated_prep_duration_min": 10.0, "estimated_travel_duration_min": 10.0, "bag_prepared": True}
    ]

    for tc in test_cases:
        X_inf = preprocessor.transform_single_inference(tc)
        assert X_inf.shape == (1, num_features), f"Inconsistent inference feature dimension for {tc['event_name']}"
        assert not np.isnan(X_inf).any()


def test_preprocessor_serialization(tmp_path):
    """Test that preprocessor can be saved with joblib and produces identical predictions upon reloading."""
    train_df = pd.DataFrame([
        {"event_name": "Standup", "day_of_week": "Monday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 1, "total_delay_min": 4.0},
        {"event_name": "Code Review", "day_of_week": "Tuesday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 0, "total_delay_min": 2.0},
        {"event_name": "Deploy", "day_of_week": "Friday", "importance": "low", "estimated_prep_duration_min": 10.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 1, "total_delay_min": 0.0},
        {"event_name": "Standup", "day_of_week": "Wednesday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 1, "total_delay_min": 6.0},
        {"event_name": "Code Review", "day_of_week": "Thursday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 1, "total_delay_min": 1.0},
        {"event_name": "Standup", "day_of_week": "Friday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 0, "total_delay_min": 8.0},
        {"event_name": "Design Sync", "day_of_week": "Wednesday", "importance": "medium", "estimated_prep_duration_min": 15.0, "estimated_travel_duration_min": 25.0, "bag_prepared": 1, "total_delay_min": 3.0},
        {"event_name": "Deploy", "day_of_week": "Monday", "importance": "high", "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 20.0, "bag_prepared": 1, "total_delay_min": 5.0}
    ])

    preprocessor = MicroDelayPreprocessor()
    preprocessor.fit(train_df)

    sample_query = {
        "event_name": "Standup",
        "day_of_week": "Monday",
        "importance": "high",
        "estimated_prep_duration_min": 20.0,
        "estimated_travel_duration_min": 25.0,
        "bag_prepared": True
    }
    orig_output = preprocessor.transform_single_inference(sample_query)

    # Save to disk
    artifact_path = tmp_path / "test_preprocessor.pkl"
    joblib.dump(preprocessor, artifact_path)

    # Load from disk
    loaded_preprocessor = joblib.load(artifact_path)
    loaded_output = loaded_preprocessor.transform_single_inference(sample_query)

    np.testing.assert_allclose(orig_output, loaded_output)
