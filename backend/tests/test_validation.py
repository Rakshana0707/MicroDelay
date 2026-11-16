"""Automated tests for chronological training and evaluation pipeline in MicroDelay."""

import datetime
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ml.validation import (
    ChronologicalSplitter,
    ExpandingWindowCV,
    ChronologicalEvaluator,
    EvaluationReport,
    MIN_CHRONOLOGICAL_SAMPLES,
)
from ml.preprocessing import MicroDelayPreprocessor


@pytest.fixture
def chronological_sequence_20():
    """Create a 20-observation chronological sequence over 20 consecutive days."""
    rows = []
    base_date = datetime.date(2026, 8, 1)
    events = ["ML Lab", "Seminar", "Library Study", "Math Lecture", "Weekend Kickoff"]

    for i in range(20):
        current_date = base_date + datetime.timedelta(days=i)
        day_name = current_date.strftime("%A")
        time_str = f"{8 + (i % 3):02d}:30:00"

        rows.append({
            "id": i + 1,
            "user_id": 1,
            "event_name": events[i % len(events)],
            "day_of_week": day_name,
            "date": current_date.isoformat(),
            "planned_departure": time_str,
            "importance": "high" if i % 2 == 0 else "medium",
            "bag_prepared": 1 if i % 3 == 0 else 0,
            "estimated_prep_duration_min": 20.0,
            "estimated_travel_duration_min": 20.0,
            "actual_prep_duration_min": 25.0 + (i % 5),
            "actual_travel_duration_min": 20.0,
            "total_delay_min": 5.0 + (i % 4),
            "provenance": "real",
            "is_synthetic": 0
        })
    return pd.DataFrame(rows)


def test_chronological_ordering_train_precedes_val_precedes_test(chronological_sequence_20):
    """Test that training observations strictly precede validation, which strictly precedes testing."""
    splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = splitter.split(chronological_sequence_20)

    # Verify counts: 70% of 20 = 14 train, 15% = 3 val, 15% = 3 test
    assert len(train_df) == 14
    assert len(val_df) == 3
    assert len(test_df) == 3

    # 1. Training strictly precedes validation in calendar date
    train_last_date = train_df["date"].max()
    val_first_date = val_df["date"].min()
    assert train_last_date <= val_first_date
    assert train_df["id"].max() < val_df["id"].min()

    # 2. Validation strictly precedes testing in calendar date
    val_last_date = val_df["date"].max()
    test_first_date = test_df["date"].min()
    assert val_last_date <= test_first_date
    assert val_df["id"].max() < test_df["id"].min()


def test_no_test_observation_leaks_into_training(chronological_sequence_20):
    """Test that zero test observation records or IDs appear in the training or validation sets."""
    splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = splitter.split(chronological_sequence_20)

    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    # Disjoint partition check
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


def test_expanding_window_cv_temporal_integrity(chronological_sequence_20):
    """Test that ExpandingWindowCV strictly maintains past-to-future temporal direction in all folds."""
    dev_df = chronological_sequence_20.iloc[:17].copy()  # Train + Val portion
    cv = ExpandingWindowCV(n_splits=3)

    folds = list(cv.split(dev_df))
    assert len(folds) >= 2

    previous_train_len = 0
    for fold_idx, cv_train, cv_val in folds:
        # 1. Fold training window expands over time
        assert len(cv_train) > previous_train_len
        previous_train_len = len(cv_train)

        # 2. Fold training observations strictly precede validation fold
        assert cv_train["date"].max() <= cv_val["date"].min()
        assert cv_train["id"].max() < cv_val["id"].min()

        # 3. No overlap
        assert len(set(cv_train["id"]).intersection(set(cv_val["id"]))) == 0


def test_preprocessing_fitted_only_on_training_data():
    """Verify that feature preprocessing transformations are fitted EXCLUSIVELY on training data."""
    # Construct training data where estimated_prep_duration_min median is 20.0
    train_data = pd.DataFrame([
        {"id": 1, "date": "2026-08-01", "planned_departure": "08:00", "event_name": "ML Lab", "day_of_week": "Monday", "importance": "high", "bag_prepared": 1, "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 15.0, "total_delay_min": 5.0},
        {"id": 2, "date": "2026-08-02", "planned_departure": "08:00", "event_name": "ML Lab", "day_of_week": "Tuesday", "importance": "high", "bag_prepared": 1, "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 15.0, "total_delay_min": 5.0},
        {"id": 3, "date": "2026-08-03", "planned_departure": "08:00", "event_name": "ML Lab", "day_of_week": "Wednesday", "importance": "high", "bag_prepared": 1, "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 15.0, "total_delay_min": 5.0},
        {"id": 4, "date": "2026-08-04", "planned_departure": "08:00", "event_name": "ML Lab", "day_of_week": "Thursday", "importance": "high", "bag_prepared": 1, "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 15.0, "total_delay_min": 5.0},
    ])

    # Test data has a completely missing prep duration and an unseen event
    test_data = pd.DataFrame([
        {"id": 5, "date": "2026-08-05", "planned_departure": "08:00", "event_name": "Brand New Unseen Event", "day_of_week": "Friday", "importance": "low", "bag_prepared": 0, "estimated_prep_duration_min": np.nan, "estimated_travel_duration_min": 30.0, "total_delay_min": 10.0}
    ])

    preprocessor = MicroDelayPreprocessor()
    preprocessor.fit(train_data)

    # Imputer learned median 20.0 strictly from train_data
    numeric_imputer = preprocessor.pipeline.named_transformers_["num"].named_steps["imputer"]
    assert numeric_imputer.statistics_[0] == 20.0

    # Transforming test_data uses train-fitted imputer (fills 20.0) and ignores unknown category without error
    X_test = preprocessor.transform(test_data)
    assert X_test.shape[0] == 1
    # Check that imputed prep duration is 20.0
    assert np.isclose(X_test[0, 0], 20.0)


def test_insufficient_datasets_handled_safely():
    """Verify that datasets below minimum threshold produce an informative report without fabricated metrics."""
    small_df = pd.DataFrame([
        {"id": 1, "date": "2026-08-01", "planned_departure": "08:00", "event_name": "ML Lab", "day_of_week": "Monday", "importance": "high", "bag_prepared": 1, "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 15.0, "total_delay_min": 5.0},
        {"id": 2, "date": "2026-08-02", "planned_departure": "08:00", "event_name": "ML Lab", "day_of_week": "Tuesday", "importance": "high", "bag_prepared": 1, "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 15.0, "total_delay_min": 5.0},
        {"id": 3, "date": "2026-08-03", "planned_departure": "08:00", "event_name": "ML Lab", "day_of_week": "Wednesday", "importance": "high", "bag_prepared": 1, "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 15.0, "total_delay_min": 5.0}
    ])

    evaluator = ChronologicalEvaluator(min_samples=MIN_CHRONOLOGICAL_SAMPLES)
    report, model, prep = evaluator.evaluate(small_df, provenance="real")

    assert report.is_validated is False
    assert report.evaluation_status == "insufficient_data"
    assert "Insufficient observations" in report.status_reason
    assert report.test_mae is None
    assert report.test_rmse is None
    assert model is None
    assert prep is None


def test_evaluation_report_schema_and_date_ranges(chronological_sequence_20):
    """Verify the completeness of the EvaluationReport schema, date ranges, and untouched test metrics."""
    evaluator = ChronologicalEvaluator(min_samples=MIN_CHRONOLOGICAL_SAMPLES)
    report, model, preprocessor = evaluator.evaluate(chronological_sequence_20, provenance="real")

    assert report.is_validated is True
    assert report.evaluation_status == "validated"
    assert report.data_provenance == "real"
    assert report.total_records == 20

    # Check partition metadata
    assert report.train_metadata["name"] == "training"
    assert report.train_metadata["count"] == 14
    assert report.train_metadata["date_range"]["start"] == "2026-08-01"
    assert report.train_metadata["date_range"]["end"] == "2026-08-14"

    assert report.val_metadata["name"] == "validation"
    assert report.val_metadata["count"] == 3
    assert report.val_metadata["date_range"]["start"] == "2026-08-15"
    assert report.val_metadata["date_range"]["end"] == "2026-08-17"

    assert report.test_metadata["name"] == "test"
    assert report.test_metadata["count"] == 3
    assert report.test_metadata["date_range"]["start"] == "2026-08-18"
    assert report.test_metadata["date_range"]["end"] == "2026-08-20"

    # Metrics
    assert report.num_cv_folds >= 2
    assert len(report.cv_fold_maes) >= 2
    assert report.test_mae is not None and report.test_mae >= 0.0
    assert report.test_rmse is not None and report.test_rmse >= 0.0
    assert report.baseline_test_mae is not None and report.baseline_test_mae >= 0.0

    # Model and preprocessor are successfully fitted
    assert model is not None
    assert preprocessor is not None
    assert preprocessor.is_fitted is True


def test_separate_evaluation_modes(chronological_sequence_20):
    """Test that synthetic demonstration evaluation and real data evaluation maintain provenance labels."""
    evaluator = ChronologicalEvaluator(min_samples=MIN_CHRONOLOGICAL_SAMPLES)

    # 1. Real data evaluation
    real_report, _, _ = evaluator.evaluate(chronological_sequence_20, provenance="real")
    assert real_report.data_provenance == "real"
    assert real_report.is_validated is True

    # 2. Synthetic demonstration evaluation
    synth_df = chronological_sequence_20.copy()
    synth_df["provenance"] = "synthetic"
    synth_df["is_synthetic"] = 1
    synth_report, _, _ = evaluator.evaluate(synth_df, provenance="synthetic")
    assert synth_report.data_provenance == "synthetic"
    assert synth_report.is_validated is True
