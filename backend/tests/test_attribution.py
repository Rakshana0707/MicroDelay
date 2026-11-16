"""Automated tests for observed vs. estimated delay attribution pipeline in MicroDelay."""

import pytest
import datetime
import numpy as np

from ml.attribution import (
    DelayAttributionEngine,
    DepartureAttributionResult,
    CategoryAttribution,
    AttributionType,
    AttributionConfidence,
    AttributionSource,
)


def test_zero_delay_attribution():
    """Verify that zero delay assigns zero departure lateness impact."""
    tags = {"phone_usage": True, "finding_items": False, "one_last_task": False, "waiting": False}
    observed = {"phone_usage": 3.0}

    res = DelayAttributionEngine.attribute_departure(
        total_delay_min=0.0,
        actual_prep_duration_min=20.0,
        estimated_prep_duration_min=20.0,
        tags=tags,
        observed_durations=observed
    )

    assert res.is_early_or_on_time is True
    assert res.departure_delay_budget_min == 0.0
    assert res.early_departure_buffer_min == 0.0
    assert res.total_delay_min == 0.0

    # Behavioral actions during prep must not be blamed for departure lateness
    for cat in res.categories:
        assert cat.departure_impact_min == 0.0
    assert "absorbed by prep buffer" in (res.overlap_explanation or "")


def test_negative_delay_early_departure():
    """Verify early departure (negative delay) handles buffer correctly with zero departure lateness."""
    tags = {"phone_usage": True}
    observed = {"phone_usage": 2.0}

    res = DelayAttributionEngine.attribute_departure(
        total_delay_min=-6.5,
        actual_prep_duration_min=18.0,
        estimated_prep_duration_min=25.0,
        tags=tags,
        observed_durations=observed
    )

    assert res.is_early_or_on_time is True
    assert res.early_departure_buffer_min == 6.5
    assert res.departure_delay_budget_min == 0.0
    for cat in res.categories:
        assert cat.departure_impact_min == 0.0


def test_multiple_observed_categories():
    """Verify directly timed behaviors are marked OBSERVED and MEASURED."""
    tags = {
        "phone_usage": True,
        "finding_items": True,
        "one_last_task": False,
        "waiting": False
    }
    observed = {
        "phone_usage": 4.0,
        "finding_items": 3.0
    }

    res = DelayAttributionEngine.attribute_departure(
        total_delay_min=10.0,
        actual_prep_duration_min=30.0,
        estimated_prep_duration_min=20.0,
        tags=tags,
        observed_durations=observed
    )

    assert res.is_early_or_on_time is False
    assert res.departure_delay_budget_min == 10.0
    assert res.total_observed_min == 7.0
    assert res.has_overlapping_durations is False

    # Check observed categories
    phone_cat = next(c for c in res.categories if c.category == "phone_usage")
    assert phone_cat.attribution_type == AttributionType.OBSERVED.value
    assert phone_cat.confidence == AttributionConfidence.MEASURED.value
    assert phone_cat.departure_impact_min == 4.0

    finding_cat = next(c for c in res.categories if c.category == "finding_items")
    assert finding_cat.attribution_type == AttributionType.OBSERVED.value
    assert finding_cat.confidence == AttributionConfidence.MEASURED.value
    assert finding_cat.departure_impact_min == 3.0

    # Remaining 3.0m should be attributed to prep_bias as estimated
    bias_cat = next(c for c in res.categories if c.category == "prep_bias")
    assert bias_cat.attribution_type == AttributionType.ESTIMATED.value
    assert bias_cat.departure_impact_min == 3.0


def test_partial_attribution_mix_of_observed_and_unobserved():
    """Verify mix where one category has measured minutes and another has only an unmeasured checkbox."""
    tags = {
        "phone_usage": True,
        "one_last_task": True,
        "finding_items": False,
        "waiting": False
    }
    # User timed phone, but only checked one_last_task checkbox without a timer
    observed = {"phone_usage": 3.0}

    res = DelayAttributionEngine.attribute_departure(
        total_delay_min=8.0,
        actual_prep_duration_min=25.0,
        estimated_prep_duration_min=20.0,
        tags=tags,
        observed_durations=observed
    )

    phone_cat = next(c for c in res.categories if c.category == "phone_usage")
    assert phone_cat.attribution_type == AttributionType.OBSERVED.value
    assert phone_cat.confidence == AttributionConfidence.MEASURED.value
    assert phone_cat.departure_impact_min == 3.0

    task_cat = next(c for c in res.categories if c.category == "one_last_task")
    assert task_cat.attribution_type == AttributionType.ESTIMATED.value
    assert task_cat.confidence == AttributionConfidence.HEURISTIC.value
    assert task_cat.departure_impact_min > 0.0


def test_overlapping_durations_exceeding_budget():
    """Verify that when observed durations exceed total departure delay, overlapping intervals are documented."""
    tags = {"phone_usage": True, "finding_items": True}
    # Sum = 6.0 + 7.0 = 13.0m, but departure delay was only 5.0m
    observed = {"phone_usage": 6.0, "finding_items": 7.0}

    res = DelayAttributionEngine.attribute_departure(
        total_delay_min=5.0,
        actual_prep_duration_min=28.0,
        estimated_prep_duration_min=20.0,
        tags=tags,
        observed_durations=observed
    )

    assert res.has_overlapping_durations is True
    assert res.overlapping_duration_min == 8.0  # 13.0 - 5.0
    assert res.overlap_explanation is not None
    assert "exceeds the total departure delay" in res.overlap_explanation

    # Total departure impact should strictly not exceed departure delay budget (5.0m)
    total_impact = sum(c.departure_impact_min for c in res.categories)
    assert np.isclose(total_impact, 5.0, atol=0.1)


def test_missing_attribution_information():
    """Verify that delays with no active tags or measurements are handled safely."""
    tags = {"phone_usage": False, "finding_items": False, "one_last_task": False, "waiting": False}

    # Delay 7m, prep took planned 20m (so prep_bias is 0) -> unexplained residual
    res = DelayAttributionEngine.attribute_departure(
        total_delay_min=7.0,
        actual_prep_duration_min=20.0,
        estimated_prep_duration_min=20.0,
        tags=tags,
        observed_durations=None
    )

    residual_cat = next(c for c in res.categories if c.category == "unexplained_residual")
    assert residual_cat.attribution_type == AttributionType.RESIDUAL.value
    assert residual_cat.confidence == AttributionConfidence.UNEXPLAINED.value
    assert residual_cat.departure_impact_min == 7.0


def test_historical_records_previous_algorithm():
    """Verify backward compatibility when parsing legacy records created by previous algorithm."""
    class MockLegacyDepartureLog:
        total_delay_min = 10.0
        delay_one_last_task_min = 4.0
        delay_finding_items_min = 3.0
        delay_phone_usage_min = 2.0
        delay_waiting_min = 0.0
        delay_prep_bias_min = 1.0
        attribution_metadata = None

    legacy_log = MockLegacyDepartureLog()
    res = DelayAttributionEngine.parse_legacy_record(legacy_log)

    assert res.total_delay_min == 10.0
    assert res.total_observed_min == 0.0
    assert res.total_estimated_min == 10.0

    for cat in res.categories:
        assert cat.attribution_type == AttributionType.LEGACY.value
        assert cat.confidence == AttributionConfidence.HEURISTIC.value
        assert "Historical record" in (cat.notes or "")


def test_checkout_api_with_observed_durations(client, auth_headers, test_user, db_session):
    """Test POST /api/checkout accepts optional observed minutes and records structured attribution."""
    payload = {
        "event_name": "Math Lecture",
        "date": "2026-11-16",
        "day_of_week": "Monday",
        "planned_prep_start": "08:00:00",
        "planned_departure": "08:30:00",
        "estimated_prep_duration_min": 30.0,
        "estimated_travel_duration_min": 20.0,
        "actual_prep_start": "08:00:00",
        "actual_departure": "08:40:00",  # 10m late
        "actual_arrival": "09:00:00",
        "actual_prep_duration_min": 40.0,
        "actual_travel_duration_min": 20.0,
        "bag_prepared": True,
        "tag_one_last_task": False,
        "tag_finding_items": True,
        "tag_phone_usage": True,
        "tag_waiting": False,
        "observed_phone_usage_min": 4.5,
        "observed_finding_items_min": 3.0
    }

    resp = client.post("/api/checkout", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["success"] is True
    assert data["calculated_delay_min"] == 10.0

    # Query insights to verify structured attribution appears in the engine
    insights_resp = client.get("/api/insights", headers=auth_headers)
    assert insights_resp.status_code == 200
    insights_data = insights_resp.json()

    assert "attribution_summary" in insights_data
    attr = insights_data["attribution_summary"]
    assert "observed_delay_contributions" in attr
    assert "estimated_contributions" in attr
    assert "total_observed_min" in attr
    assert "attribution_completeness_pct" in attr
    assert "scientific_disclaimer" in attr
