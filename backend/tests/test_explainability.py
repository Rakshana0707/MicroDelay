"""Comprehensive Automated Test Suite for Prediction Explainability & Behavioral Reasoning (Prompt 2.6).

Verifies:
1. Model-specific explanation generation (Random Forest vs. Logistic Regression).
2. Local prediction explanations (positive vs. mitigating factors).
3. Historical behavioral comparisons (context vs. baseline profile).
4. Strict distinction between statistical association and causation.
5. SHAP dependency failure handling & graceful tree decomposition fallback.
6. Robustness against missing feature values and unseen categories.
7. Model version alignment and explanation audit trail persistence.
8. User data isolation (preventing unauthorized cross-user inspection).
9. FastAPI REST API endpoints integration (/api/intelligence/explain, history, importance).
"""

import pytest
import datetime
import json
import numpy as np
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from database import Base, User, DepartureLog, PredictionExplanationRecord
from main import app, get_db
import security
from ml.adaptive import PredictionContext
from ml.explainability import (
    ContributingFactor,
    PredictionExplanation,
    HistoricalBehavioralComparator,
    LocalPredictionExplainer,
    GlobalFeatureImportanceCalculator,
    ExplainabilityService,
    CAUSATION_DISCLAIMER
)


@pytest.fixture
def test_db():
    """In-memory SQLite database isolated per test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()


def create_test_user(db, username: str = "explain_user"):
    """Helper to create a test user."""
    user = User(
        username=username,
        hashed_password="hashed_pw",
        salt="test_salt",
        profile_type="it_professional",
        consent_to_global_training=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_local_prediction_explainer_logistic_regression():
    """Verify model-specific coefficient attribution for Logistic Regression."""
    # Synthetic dataset with known coefficients:
    # Feature 0: positive impact on lateness (e.g. Monday)
    # Feature 1: negative impact on lateness (e.g. bag_prepared)
    X = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    y = np.array([1, 0, 0, 0, 1, 0])

    model = LogisticRegression()
    model.fit(X, y)

    feature_names = ["day_of_week_Monday", "bag_prepared"]
    x_test = np.array([1.0, 0.0])  # Monday with bag unprepared -> should be high risk

    ctx = PredictionContext(
        user_id=1,
        event_name="Work Standup",
        day_of_week="Monday",
        event_importance="medium",
        estimated_prep_duration=20.0,
        estimated_travel_duration=20.0,
        bag_prepared=False
    )

    explanation = LocalPredictionExplainer.explain_prediction(
        model=model,
        feature_names=feature_names,
        x_vector=x_test,
        context=ctx,
        predicted_delay_min=6.0,
        late_probability=0.82,
        model_version="lr_test_v1"
    )

    assert explanation.explanation_method == "linear_coefficients"
    assert len(explanation.top_positive_contributors) > 0
    top_pos = explanation.top_positive_contributors[0]
    assert top_pos["direction"] == "increases_delay"
    assert top_pos["evidence_type"] == "model_contribution"
    assert top_pos["raw_weight"] is not None


def test_local_prediction_explainer_random_forest():
    """Verify tree-based feature contribution decomposition for Random Forest."""
    X = np.array([
        [0, 20.0, 15.0],
        [1, 35.0, 20.0],
        [0, 15.0, 10.0],
        [1, 40.0, 25.0],
        [0, 10.0, 10.0],
        [1, 30.0, 20.0]
    ])
    y = np.array([1, 0, 1, 0, 1, 0])

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    feature_names = ["bag_prepared", "estimated_prep_duration_min", "estimated_travel_duration_min"]
    x_test = np.array([0, 15.0, 10.0])

    ctx = PredictionContext(
        user_id=1,
        event_name="Work",
        day_of_week="Monday",
        event_importance="high",
        estimated_prep_duration=15.0,
        estimated_travel_duration=10.0,
        bag_prepared=False
    )

    explanation = LocalPredictionExplainer.explain_prediction(
        model=rf,
        feature_names=feature_names,
        x_vector=x_test,
        context=ctx,
        predicted_delay_min=5.5,
        late_probability=0.75,
        model_version="rf_test_v1"
    )

    assert explanation.explanation_method in ("tree_shap", "tree_feature_contribution")
    assert explanation.model_version == "rf_test_v1"
    assert len(explanation.top_positive_contributors) > 0
    assert explanation.causation_disclaimer == CAUSATION_DISCLAIMER


def test_shap_dependency_failure_graceful_fallback():
    """Verify inference does NOT crash when SHAP raises an error or is uninstalled."""
    X = np.array([[0, 20.0], [1, 30.0], [0, 15.0], [1, 40.0]])
    y = np.array([1, 0, 1, 0])
    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X, y)

    feature_names = ["bag_prepared", "estimated_prep_duration_min"]
    x_test = np.array([0, 15.0])

    ctx = PredictionContext(
        user_id=1,
        event_name="Office",
        day_of_week="Tuesday",
        event_importance="medium",
        estimated_prep_duration=15.0,
        estimated_travel_duration=15.0,
        bag_prepared=False
    )

    # Force any shap import or TreeExplainer call to raise an exception
    with patch.dict("sys.modules", {"shap": None}):
        explanation = LocalPredictionExplainer.explain_prediction(
            model=rf,
            feature_names=feature_names,
            x_vector=x_test,
            context=ctx,
            predicted_delay_min=4.0,
            late_probability=0.6,
            model_version="rf_no_shap_v1"
        )

        assert explanation.explanation_method == "tree_feature_contribution"
        assert len(explanation.top_positive_contributors) > 0
        # Limitations should document the fallback
        assert any("SHAP" in lim for lim in explanation.limitations)


def test_missing_feature_values_and_unseen_categories():
    """Verify robustness when feature values are None, empty, or novel categories."""
    ctx = PredictionContext(
        user_id=1,
        event_name="Novel Unseen Symposium Event",
        day_of_week="UnknownDay",  # Will sanitize
        event_importance="critical_unknown",
        estimated_prep_duration=0.0,
        estimated_travel_duration=0.0,
        bag_prepared=False
    )

    # Empty feature names and None model -> should safely trigger rule-based reasoning
    explanation = LocalPredictionExplainer.explain_prediction(
        model=None,
        feature_names=[],
        x_vector=np.array([]),
        context=ctx,
        predicted_delay_min=4.5,
        late_probability=0.55,
        model_version="default_fallback"
    )

    assert explanation.explanation_method == "behavioral_rule_reasoning"
    assert len(explanation.top_positive_contributors) > 0
    # Must contain causation disclaimer
    assert len(explanation.causation_disclaimer) > 0


def test_historical_behavioral_comparison():
    """Verify factual comparisons against user's historical profile."""
    ctx = PredictionContext(
        user_id=1,
        event_name="Morning Office Commute",
        day_of_week="Monday",
        event_importance="high",
        estimated_prep_duration=15.0,
        estimated_travel_duration=20.0,
        bag_prepared=False
    )

    mock_profile = {
        "dimensions": {
            "all_time": {
                "preparation": {"avg_actual_prep_min": 28.0},
                "organization": {"bag_prepared_consistency": 0.75},
                "departure": {
                    "day_specific_patterns": {
                        "Monday": {"avg_delay_min": 5.4, "lateness_rate": 0.65, "sample_count": 8}
                    }
                }
            }
        }
    }

    comparisons = HistoricalBehavioralComparator.compare_context_with_history(ctx, mock_profile)
    assert len(comparisons) >= 2

    comp_statements = " ".join([c["statement"] for c in comparisons])
    # Checks preparation underestimation
    assert "lower than your historical average" in comp_statements
    # Checks Monday elevated lateness
    assert "Mondays historically average" in comp_statements
    # Checks bag comparison
    assert "bag is not marked as prepared" in comp_statements or "recurring" in comp_statements


def test_causation_vs_association_distinction():
    """Verify that every explanation factor strictly carries its epistemic evidence type."""
    ctx = PredictionContext(
        user_id=1,
        event_name="Standup",
        day_of_week="Monday",
        event_importance="medium",
        estimated_prep_duration=15.0,
        estimated_travel_duration=15.0,
        bag_prepared=True
    )

    explanation = LocalPredictionExplainer.explain_prediction(
        model=None,
        feature_names=[],
        x_vector=np.array([]),
        context=ctx,
        predicted_delay_min=3.0,
        late_probability=0.4
    )

    valid_sources = {
        "model_contribution",
        "statistical_association",
        "directly_observed_behavior",
        "estimated_behavioral_influence"
    }

    all_factors = explanation.top_positive_contributors + explanation.top_mitigating_factors
    assert len(all_factors) > 0
    for f in all_factors:
        assert f["evidence_type"] in valid_sources
        assert "direction" in f
        assert f["direction"] in ("increases_delay", "decreases_delay")


def test_global_feature_importance_calculator():
    """Verify global feature ranking with permutation importance and MDI."""
    X = np.array([
        [0, 20.0, 15.0],
        [1, 35.0, 20.0],
        [0, 15.0, 10.0],
        [1, 40.0, 25.0],
        [0, 10.0, 10.0],
        [1, 30.0, 20.0]
    ])
    y = np.array([1, 0, 1, 0, 1, 0])

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    feature_names = ["bag_prepared", "estimated_prep_duration_min", "estimated_travel_duration_min"]

    # Test MDI
    res_mdi = GlobalFeatureImportanceCalculator.calculate_global_importance(rf, feature_names)
    assert res_mdi["method"] == "gini_impurity_mdi"
    assert len(res_mdi["features"]) == 3
    assert res_mdi["features"][0]["importance"] >= res_mdi["features"][1]["importance"]

    # Test Permutation Importance with validation data
    res_perm = GlobalFeatureImportanceCalculator.calculate_global_importance(rf, feature_names, X_val=X, y_val=y)
    assert res_perm["method"] in ("permutation_importance", "gini_impurity_mdi")
    assert len(res_perm["features"]) == 3


def test_explanation_persistence_and_user_data_isolation(test_db):
    """Verify explanation persistence and strict cross-user data isolation."""
    user_a = create_test_user(test_db, "user_a")
    user_b = create_test_user(test_db, "user_b")

    ctx_a = PredictionContext(
        user_id=user_a.id,
        event_name="Secret Meeting A",
        day_of_week="Friday",
        event_importance="high",
        estimated_prep_duration=25.0,
        estimated_travel_duration=20.0,
        bag_prepared=True
    )

    # 1. Record explanation for user_a
    res_a = ExplainabilityService.explain_and_record(
        db=test_db,
        user_id=user_a.id,
        context=ctx_a,
        predicted_delay_min=2.0,
        late_probability=0.25,
        model_version="test_v1_a",
        persist=True
    )

    pred_id = res_a["prediction_id"]
    assert pred_id.startswith("pred_")

    # 2. User A can retrieve it
    retrieved_a = ExplainabilityService.get_explanation_by_id(test_db, user_a.id, pred_id)
    assert retrieved_a is not None
    assert retrieved_a["model_version"] == "test_v1_a"
    assert retrieved_a["feature_audit_values"]["event_name"] == "Secret Meeting A"

    # 3. User B cannot retrieve User A's explanation (Isolation Security Guarantee)
    retrieved_b = ExplainabilityService.get_explanation_by_id(test_db, user_b.id, pred_id)
    assert retrieved_b is None

    # 4. User A's history contains the explanation
    history_a = ExplainabilityService.get_explanation_history(test_db, user_a.id, limit=5)
    assert len(history_a) == 1
    assert history_a[0]["prediction_id"] == pred_id

    # 5. User B's history is empty
    history_b = ExplainabilityService.get_explanation_history(test_db, user_b.id, limit=5)
    assert len(history_b) == 0


def test_fastapi_explainability_endpoints_integration(test_db):
    """Verify /api/intelligence/explain and related REST endpoints."""
    user = create_test_user(test_db, "api_explain_user")

    app.dependency_overrides[get_db] = lambda: test_db
    client = TestClient(app)

    token = security.create_access_token(data={"sub": user.username})
    headers = {"Authorization": f"Bearer {token}"}

    # 1. GET /api/intelligence/explain (on-demand context)
    res = client.get(
        "/api/intelligence/explain?event_name=Daily+Standup&day_of_week=Monday&bag_prepared=false&estimated_prep=20&estimated_travel=15",
        headers=headers
    )
    assert res.status_code == 200
    data = res.json()
    assert "prediction_id" in data
    assert "top_positive_contributors" in data
    assert "causation_disclaimer" in data

    pred_id = data["prediction_id"]

    # 2. GET /api/intelligence/explain by prediction_id
    res_id = client.get(f"/api/intelligence/explain?prediction_id={pred_id}", headers=headers)
    assert res_id.status_code == 200
    assert res_id.json()["prediction_id"] == pred_id

    # 3. GET /api/intelligence/explanations/history
    res_hist = client.get("/api/intelligence/explanations/history?limit=5", headers=headers)
    assert res_hist.status_code == 200
    hist = res_hist.json()
    assert isinstance(hist, list)
    assert len(hist) >= 1

    # 4. GET /api/intelligence/features/importance
    res_imp = client.get("/api/intelligence/features/importance", headers=headers)
    assert res_imp.status_code == 200
    imp_data = res_imp.json()
    assert "features" in imp_data
    assert len(imp_data["features"]) > 0

    # 5. Cross-user access blocked
    other_user = create_test_user(test_db, "other_explain_user")
    other_token = security.create_access_token(data={"sub": other_user.username})
    other_headers = {"Authorization": f"Bearer {other_token}"}

    res_cross = client.get(f"/api/intelligence/explain?prediction_id={pred_id}", headers=other_headers)
    assert res_cross.status_code == 404

    app.dependency_overrides.clear()
