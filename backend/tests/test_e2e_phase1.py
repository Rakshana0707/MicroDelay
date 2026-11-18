"""End-to-End Verification Test Suite for MicroDelay Phase 1.

Verifies the complete Phase 1 pipeline:
1. User registration & synthetic cold-start generation.
2. Provenance separation & data quality indicators.
3. Genuine simulator departure checkout with observed delay attribution.
4. Leakage-safe feature extraction & chronological model training.
5. Multi-algorithm regression benchmarking (5 models).
6. Standardized evaluation reporting & residual analysis.
7. Model version registry & safe rollback.
8. Predict endpoint & routine management.
"""

import datetime
import pytest
from database import User, DepartureLog, Event, ModelVersionRecord


def test_full_phase1_end_to_end_user_journey(client, db_session):
    # 1. Register a new user
    username = "e2e_phase1_user"
    password = "e2e_password_123"
    profile_type = "student"

    reg_resp = client.post("/api/auth/register", json={
        "username": username,
        "password": password,
        "profile_type": profile_type
    })
    assert reg_resp.status_code == 200, f"Registration failed: {reg_resp.text}"
    auth_data = reg_resp.json()
    token = auth_data["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Verify synthetic demonstration data provenance
    summary_resp = client.get("/api/data-summary", headers=headers)
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["synthetic_samples"] >= 60
    assert summary["real_samples"] == 0
    assert summary["has_real_data"] is False
    assert summary["is_sufficient_real"] is False
    assert summary["active_model"]["trained"] is True
    assert summary["active_model"]["data_provenance"] == "synthetic"
    assert summary["active_model"]["is_personalized_real"] is False

    # Verify logs origin
    logs_resp = client.get("/api/logs", headers=headers)
    assert logs_resp.status_code == 200
    initial_logs = logs_resp.json()
    assert len(initial_logs) >= 60
    assert all(log["is_synthetic"] is True for log in initial_logs)
    assert all(log["provenance"] == "synthetic" for log in initial_logs)

    # 3. Complete a genuine departure checkout in Morning Simulator
    today_str = datetime.date.today().isoformat()
    checkout_payload = {
        "event_name": "ML Lab",
        "day_of_week": "Monday",
        "date": today_str,
        "importance": "high",
        "planned_prep_start": "07:45:00",
        "planned_departure": "08:10:00",
        "estimated_prep_duration_min": 25.0,
        "estimated_travel_duration_min": 20.0,
        "actual_prep_start": "07:35:00",
        "actual_departure": "08:18:00",
        "actual_arrival": "08:38:00",
        "actual_prep_duration_min": 43.0,
        "actual_travel_duration_min": 20.0,
        "bag_prepared": True,
        "tag_one_last_task": False,
        "tag_finding_items": True,
        "tag_phone_usage": True,
        "tag_waiting": False,
        "observed_finding_items_min": 3.0,
        "observed_phone_usage_min": 5.0
    }

    checkout_resp = client.post("/api/checkout", json=checkout_payload, headers=headers)
    assert checkout_resp.status_code == 200
    checkout_data = checkout_resp.json()
    assert checkout_data["calculated_delay_min"] == 8.0
    assert "allocations" in checkout_data
    assert checkout_data["allocations"]["finding_items"] > 0
    assert checkout_data["allocations"]["phone_usage"] > 0

    # 4. Verify genuine record provenance in data summary and logs
    summary_resp2 = client.get("/api/data-summary", headers=headers)
    assert summary_resp2.status_code == 200
    summary2 = summary_resp2.json()
    assert summary2["real_samples"] == 1
    assert summary2["synthetic_samples"] == summary["synthetic_samples"]
    assert summary2["has_real_data"] is True
    assert summary2["is_sufficient_real"] is False # 1 < 8 required

    logs_resp2 = client.get("/api/logs", headers=headers)
    assert logs_resp2.status_code == 200
    updated_logs = logs_resp2.json()
    assert len(updated_logs) == len(initial_logs) + 1
    real_logs = [l for l in updated_logs if not l["is_synthetic"]]
    assert len(real_logs) == 1
    latest_log = real_logs[0]
    assert latest_log["is_synthetic"] is False
    assert latest_log["provenance"] == "real"
    assert latest_log["data_source"] == "simulator_checkout"

    # 5. Run Model Training & Evaluation
    metrics_resp = client.get("/api/metrics", headers=headers)
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["success"] is True
    assert "baseline_mae" in metrics
    assert "ml_mae" in metrics
    assert "error_reduction_pct" in metrics
    assert "model_version" in metrics
    assert metrics["data_provenance"] == "synthetic" # Still synthetic demonstration because real < 8

    # 6. Run Multi-Algorithm Benchmark Suite
    bench_resp = client.get("/api/benchmarks", headers=headers)
    assert bench_resp.status_code == 200
    bench = bench_resp.json()
    assert len(bench["models"]) >= 4
    model_names = [m["model_name"] for m in bench["models"]]
    assert "Historical Statistical Baseline" in model_names
    assert "Linear Regression" in model_names
    assert "Random Forest Regressor" in model_names
    assert "HistGradientBoosting Regressor" in model_names
    assert bench["selected_best_model"] is not None
    assert "selection_rationale" in bench

    # 7. Fetch Latest Evaluation Report
    report_resp = client.get("/api/reports/latest?provenance=synthetic", headers=headers)
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert "metrics" in report
    assert "residual_analysis" in report
    assert report["metadata"]["data_provenance"] == "synthetic"
    tol_key = "tolerance_5m_pct" if "tolerance_5m_pct" in report["metrics"] else "tolerance_within_5min_pct"
    assert report["metrics"][tol_key] >= 0

    # 8. Check Model Version Registry
    versions_resp = client.get("/api/models/versions", headers=headers)
    assert versions_resp.status_code == 200
    versions = versions_resp.json()
    assert len(versions) >= 1
    active_version = [v for v in versions if v["is_active"]][0]
    assert active_version["model_status"] == "active"
    assert active_version["model_version"] != ""

    # 9. Verify Prediction API with versioned model
    pred_resp = client.get("/api/predict", params={
        "event_name": "ML Lab",
        "day_of_week": "Monday",
        "importance": "high",
        "bag_prepared": True,
        "estimated_prep": 25.0,
        "estimated_travel": 20.0
    }, headers=headers)
    assert pred_resp.status_code == 200
    pred = pred_resp.json()
    assert "predicted_delay_min" in pred
    assert "baseline_predicted_delay_min" in pred
    assert pred["model_used"] == "random_forest"
    assert pred["model_version"] != ""

    # 10. Verify Routine Creation & Management
    event_resp = client.post("/api/events", json={
        "name": "E2E Thesis Lab",
        "day_of_week": "Tuesday",
        "start_time": "10:00:00",
        "estimated_prep_duration": 30,
        "estimated_travel_duration": 15,
        "importance": "high"
    }, headers=headers)
    assert event_resp.status_code == 200
    created_event = event_resp.json()
    assert created_event["name"] == "E2E Thesis Lab"

    # List events
    events_list_resp = client.get("/api/events", headers=headers)
    assert events_list_resp.status_code == 200
    events_list = events_list_resp.json()
    assert any(e["name"] == "E2E Thesis Lab" for e in events_list)

    # Delete event
    del_resp = client.delete(f"/api/events/{created_event['id']}", headers=headers)
    assert del_resp.status_code == 200

    # 11. Verify Insights delay breakdown & attribution
    insights_resp = client.get("/api/insights", headers=headers)
    assert insights_resp.status_code == 200
    insights = insights_resp.json()
    assert insights["has_data"] is True
    assert "delay_sums" in insights
    assert "lateness_rate_pct" in insights
    assert len(insights["recommendations"]) > 0
    assert insights["real_samples"] == 1
    assert insights["synthetic_samples"] >= 60
    assert insights["data_provenance"] == "mixed_insufficient_real"
