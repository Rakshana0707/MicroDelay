"""Automated tests for multi-algorithm regression benchmarking pipeline in MicroDelay."""

import datetime
import pytest
import pandas as pd
import numpy as np

from ml.benchmark import (
    ModelBenchmarkSuite,
    ModelBenchmarkResult,
    BenchmarkSuiteResult,
    calculate_prediction_tolerance,
)


@pytest.fixture
def benchmark_sample_25():
    """Create a 25-observation chronological dataset with realistic delay patterns."""
    rows = []
    base_date = datetime.date(2026, 8, 1)
    events = ["ML Lab", "Seminar", "Library Study", "Math Lecture", "Weekend Kickoff"]

    for i in range(25):
        current_date = base_date + datetime.timedelta(days=i)
        day_name = current_date.strftime("%A")
        time_str = f"{8 + (i % 3):02d}:30:00"

        # Introduce patterned delay: higher delay when bag is unprepared and importance is high
        bag_prep = 1 if i % 3 == 0 else 0
        importance = "high" if i % 2 == 0 else "medium"
        base_delay = 8.0 if bag_prep == 0 else 2.0
        if importance == "high":
            base_delay += 3.0

        rows.append({
            "id": i + 1,
            "user_id": 1,
            "event_name": events[i % len(events)],
            "day_of_week": day_name,
            "date": current_date.isoformat(),
            "planned_departure": time_str,
            "importance": importance,
            "bag_prepared": bag_prep,
            "estimated_prep_duration_min": 20.0,
            "estimated_travel_duration_min": 20.0,
            "actual_prep_duration_min": 25.0 + (i % 4),
            "actual_travel_duration_min": 22.0,
            "total_delay_min": float(base_delay + (i % 3)),
            "provenance": "synthetic",
            "is_synthetic": 1
        })
    return pd.DataFrame(rows)


def test_prediction_tolerance_calculation():
    """Test prediction tolerance metric (+/- 5.0 minutes threshold)."""
    # 1. Exact matches: 100%
    y_true = np.array([5.0, 10.0, 15.0, 20.0])
    y_pred = np.array([5.0, 10.0, 15.0, 20.0])
    assert calculate_prediction_tolerance(y_true, y_pred, tolerance_min=5.0) == 100.0

    # 2. Borderline and outside:
    # Errors: |0.0|=0, |4.9|=4.9 (in), |5.0|=5.0 (in), |5.1|=5.1 (out) -> 3 out of 4 = 75.0%
    y_pred_mixed = np.array([5.0, 14.9, 20.0, 25.1])
    assert calculate_prediction_tolerance(y_true, y_pred_mixed, tolerance_min=5.0) == 75.0

    # 3. Empty input
    assert calculate_prediction_tolerance(np.array([]), np.array([])) == 0.0


def test_benchmark_suite_models_execution(benchmark_sample_25):
    """Test that the benchmark suite executes all candidate models on identical partitions."""
    suite = ModelBenchmarkSuite(min_samples=8)
    res = suite.run_benchmark(benchmark_sample_25, provenance="synthetic")

    assert res.benchmark_status == "completed"
    assert res.is_demonstration_only is True
    assert res.total_observations == 25
    assert res.train_samples > 0
    assert res.val_samples > 0
    assert res.test_samples > 0

    # Model names checked
    model_names = [m.model_name for m in res.models]
    assert "Historical Statistical Baseline" in model_names
    assert "Linear Regression" in model_names
    assert "Random Forest Regressor" in model_names
    assert "HistGradientBoosting Regressor" in model_names
    assert "XGBoost Regressor" in model_names

    # Verify metrics recorded for evaluated models
    for m in res.models:
        if m.status == "evaluated":
            assert m.val_mae is not None
            assert m.val_rmse is not None
            assert m.val_medae is not None
            assert m.val_tolerance_5m_pct is not None
            assert m.training_duration_ms >= 0.0


def test_model_selection_and_untouched_test_evaluation(benchmark_sample_25):
    """Test that the best model is selected by validation performance and evaluated on the untouched test set."""
    suite = ModelBenchmarkSuite(min_samples=8)
    res = suite.run_benchmark(benchmark_sample_25, provenance="synthetic")

    assert res.selected_best_model is not None
    assert "Selected" in res.selection_rationale

    # Find winning model result
    winning_model = next(m for m in res.models if m.model_name == res.selected_best_model)
    assert winning_model.status == "evaluated"
    assert winning_model.test_mae is not None
    assert winning_model.test_rmse is not None
    assert winning_model.test_medae is not None
    assert winning_model.test_tolerance_5m_pct is not None

    # Verify that unselected models DID NOT evaluate on the test set (holdout integrity)
    for m in res.models:
        if m.model_name != res.selected_best_model:
            assert m.test_mae is None


def test_insufficient_data_benchmark_safety():
    """Verify that datasets below minimum threshold return insufficient_data safely."""
    small_df = pd.DataFrame([
        {"id": 1, "date": "2026-08-01", "planned_departure": "08:00", "event_name": "ML Lab", "day_of_week": "Monday", "importance": "high", "bag_prepared": 1, "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 15.0, "total_delay_min": 5.0},
        {"id": 2, "date": "2026-08-02", "planned_departure": "08:00", "event_name": "ML Lab", "day_of_week": "Tuesday", "importance": "high", "bag_prepared": 1, "estimated_prep_duration_min": 20.0, "estimated_travel_duration_min": 15.0, "total_delay_min": 5.0}
    ])

    suite = ModelBenchmarkSuite(min_samples=8)
    res = suite.run_benchmark(small_df, provenance="real")

    assert res.benchmark_status == "insufficient_data"
    assert res.selected_best_model is None
    assert "Insufficient data" in res.status_reason


def test_api_benchmarks_endpoint(client, auth_headers, test_user, db_session):
    """Test GET /api/benchmarks endpoint returns structured multi-model comparisons."""
    from database import DepartureLog, DataProvenance, DataSource

    # Seed 15 demo departure logs
    base_date = datetime.date(2026, 8, 1)
    for i in range(15):
        curr_date = base_date + datetime.timedelta(days=i)
        log = DepartureLog(
            user_id=test_user.id,
            event_name="ML Lab" if i % 2 == 0 else "Seminar",
            date=curr_date.isoformat(),
            day_of_week=curr_date.strftime("%A"),
            planned_prep_start="08:00:00",
            planned_departure="08:20:00",
            estimated_prep_duration_min=20.0,
            estimated_travel_duration_min=20.0,
            actual_prep_start="08:00:00",
            actual_departure="08:25:00",
            actual_arrival="08:45:00",
            actual_prep_duration_min=25.0,
            actual_travel_duration_min=20.0,
            total_delay_min=5.0,
            bag_prepared=True,
            provenance=DataProvenance.SYNTHETIC.value,
            data_source=DataSource.GENERATOR.value,
            is_synthetic=True
        )
        db_session.add(log)
    db_session.commit()

    resp = client.get("/api/benchmarks?training_mode=synthetic_only", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["benchmark_status"] == "completed"
    assert data["data_provenance"] == "synthetic"
    assert data["is_demonstration_only"] is True
    assert len(data["models"]) >= 4
    assert data["selected_best_model"] is not None
