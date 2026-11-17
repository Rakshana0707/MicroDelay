"""Personalized Lateness Risk Classification Engine for MicroDelay.

Extends MicroDelay beyond delay duration prediction by providing a dedicated,
calibrated binary classifier that estimates the probability of departing late.

Features:
- Configurable operational lateness tolerance (default: total_delay_min > 0.0)
- Leakage-safe feature extraction (prior to departure only)
- Multi-algorithm benchmarking: Logistic Regression, Random Forest, HistGradientBoosting, XGBoost
- Class-imbalance handling (class_weight='balanced', scale_pos_weight)
- Chronological train / validation / test evaluation
- Probability calibration (Platt scaling / Sigmoid & Isotonic)
- Comprehensive metrics: Accuracy, Precision, Recall, F1, Macro F1, ROC-AUC, PR-AUC, Confusion Matrix, Brier Score
- Configurable operational risk categories (Low < 30%, Moderate 30-60%, High > 60%)
- Independent operation alongside the regression model
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger("microdelay.ml.classifier")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@dataclass
class LatenessClassificationConfig:
    """Configuration for binary departure lateness classification."""
    tolerance_min: float = 0.0  # delay > tolerance_min is classified as late (1)
    low_risk_threshold_pct: float = 30.0   # < 30% -> Low Risk
    high_risk_threshold_pct: float = 60.0  # > 60% -> High Risk
    extreme_risk_threshold_pct: float = 85.0 # > 85% -> Extreme Risk
    calibration_method: str = "sigmoid"    # "sigmoid" (Platt scaling) or "isotonic"
    min_samples_for_training: int = 8
    random_state: int = 42

    def categorize_probability(self, late_probability: float) -> str:
        """Map calibrated probability [0.0, 1.0] to operational risk category."""
        pct = round(late_probability * 100.0, 2)
        if pct < self.low_risk_threshold_pct:
            return "low"
        elif pct <= self.high_risk_threshold_pct:
            return "moderate"
        elif pct <= self.extreme_risk_threshold_pct:
            return "high"
        else:
            return "extreme"


@dataclass
class ClassificationEvaluationMetrics:
    """Standardized classification performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    brier_score: float
    confusion_matrix: Dict[str, int]
    positive_prevalence_pct: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "macro_f1": round(self.macro_f1, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "pr_auc": round(self.pr_auc, 4) if self.pr_auc is not None else None,
            "brier_score": round(self.brier_score, 4),
            "confusion_matrix": self.confusion_matrix,
            "positive_prevalence_pct": round(self.positive_prevalence_pct, 1),
            "sample_count": self.sample_count
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> ClassificationEvaluationMetrics:
    """Calculate comprehensive evaluation metrics handling class skew and edge cases."""
    n = len(y_true)
    if n == 0:
        return ClassificationEvaluationMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, macro_f1=0.0,
            roc_auc=None, pr_auc=None, brier_score=1.0,
            confusion_matrix={"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            positive_prevalence_pct=0.0, sample_count=0
        )

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC-AUC requires at least 2 distinct classes in y_true
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = None
    else:
        roc_auc = None
        pr_auc = None

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = 0.5

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_dict = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }

    pos_prev = (float(np.sum(y_true == 1)) / n) * 100.0

    return ClassificationEvaluationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        macro_f1=macro_f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        confusion_matrix=cm_dict,
        positive_prevalence_pct=pos_prev,
        sample_count=n
    )


class LatenessClassifierSuite:
    """Benchmarks, calibrates, and selects optimal binary classifiers for departure lateness."""

    def __init__(self, config: Optional[LatenessClassificationConfig] = None):
        self.config = config or LatenessClassificationConfig()

    def train_and_benchmark(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform chronological train-val-test split, benchmark classifiers, and calibrate."""
        if len(df) < self.config.min_samples_for_training:
            return {
                "success": False,
                "reason": f"Insufficient records ({len(df)} < {self.config.min_samples_for_training}) for classifier training.",
                "models": [],
                "selected_model": None
            }

        # 1. Ensure Chronological Ordering
        df_sorted = df.copy()
        if "date" in df_sorted.columns:
            sort_cols = ["date"]
            if "planned_departure" in df_sorted.columns:
                sort_cols.append("planned_departure")
            df_sorted = df_sorted.sort_values(sort_cols).reset_index(drop=True)

        # 2. Define Binary Target (Operational Lateness Definition)
        # 0 = on-time/early, 1 = late (> tolerance_min)
        y_all = (df_sorted["total_delay_min"] > self.config.tolerance_min).astype(int).values

        # 3. Preprocess Features
        try:
            try:
                from ml.preprocessing import MicroDelayPreprocessor
            except ModuleNotFoundError:
                from backend.ml.preprocessing import MicroDelayPreprocessor
            preprocessor = MicroDelayPreprocessor()
        except Exception as e:
            logger.warning("Could not instantiate MicroDelayPreprocessor: %s", e)
            preprocessor = None

        n = len(df_sorted)
        train_end = max(4, int(n * 0.70))
        val_end = max(train_end + 2, int(n * 0.85))

        df_train = df_sorted.iloc[:train_end]
        df_val = df_sorted.iloc[train_end:val_end]
        df_test = df_sorted.iloc[val_end:]

        y_train = y_all[:train_end]
        y_val = y_all[train_end:val_end]
        y_test = y_all[val_end:]

        if preprocessor is not None:
            preprocessor.fit(df_train)
            X_train = preprocessor.transform(df_train)
            X_val = preprocessor.transform(df_val)
            X_test = preprocessor.transform(df_test)
            fitted_cols = preprocessor.feature_names_
        else:
            try:
                try:
                    from ml.features import preprocess_features
                except ModuleNotFoundError:
                    from backend.ml.features import preprocess_features
                X_train, fitted_cols = preprocess_features(df_train)
                X_val, _ = preprocess_features(df_val, fit_columns=fitted_cols)
                X_test, _ = preprocess_features(df_test, fit_columns=fitted_cols)
            except Exception:
                num_cols = ["estimated_prep_duration_min", "estimated_travel_duration_min", "bag_prepared"]
                X_train = df_train[num_cols].fillna(0).values
                X_val = df_val[num_cols].fillna(0).values
                X_test = df_test[num_cols].fillna(0).values
                fitted_cols = num_cols

        # Check training class diversity
        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        pos_weight = (neg_count / max(1, pos_count)) if pos_count > 0 else 1.0

        # 4. Instantiate Candidate Classifiers
        candidates: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=self.config.random_state
            ),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                class_weight="balanced",
                random_state=self.config.random_state
            ),
            "HistGradientBoosting Classifier": HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                class_weight="balanced",
                random_state=self.config.random_state
            )
        }

        if XGBOOST_AVAILABLE:
            candidates["XGBoost Classifier"] = XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.08,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=self.config.random_state
            )

        model_results = []
        best_val_f1 = -1.0
        best_model_name = None
        best_fitted_model = None

        # 5. Fit & Benchmark Candidates on Chronological Partitions
        for name, clf in candidates.items():
            try:
                # Fit candidate on training partition only
                clf.fit(X_train, y_train)

                # Validation predictions
                y_val_pred = clf.predict(X_val)
                if hasattr(clf, "predict_proba"):
                    y_val_prob = clf.predict_proba(X_val)[:, 1]
                else:
                    y_val_prob = y_val_pred.astype(float)

                val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_prob)

                # Raw Brier score
                brier_raw = val_metrics.brier_score

                # Probability Calibration via Platt scaling on validation fold
                calibrated_clf = clf
                brier_calibrated = brier_raw
                is_calibrated = False

                if len(np.unique(y_train)) > 1 and len(np.unique(y_val)) > 1 and len(y_val) >= 4:
                    try:
                        cal_cv = CalibratedClassifierCV(
                            estimator=clf,
                            method=self.config.calibration_method,
                            cv="prefit"
                        )
                        cal_cv.fit(X_val, y_val)
                        y_val_prob_cal = cal_cv.predict_proba(X_val)[:, 1]
                        brier_cal = float(brier_score_loss(y_val, y_val_prob_cal))
                        # Adopt calibration if Brier score improves or remains comparable
                        if brier_cal <= (brier_raw + 0.02):
                            calibrated_clf = cal_cv
                            brier_calibrated = brier_cal
                            is_calibrated = True
                    except Exception as cal_err:
                        logger.debug("Calibration skipped for %s: %s", name, cal_err)

                model_info = {
                    "model_name": name,
                    "validation_metrics": val_metrics.to_dict(),
                    "brier_score_raw": round(brier_raw, 4),
                    "brier_score_calibrated": round(brier_calibrated, 4),
                    "is_calibrated": is_calibrated,
                    "model_object": calibrated_clf
                }
                model_results.append(model_info)

                # Parsimonious model selection: Macro F1 on validation
                if val_metrics.macro_f1 > best_val_f1:
                    best_val_f1 = val_metrics.macro_f1
                    best_model_name = name
                    best_fitted_model = calibrated_clf

            except Exception as fit_err:
                logger.error("Failed to train candidate classifier '%s': %s", name, fit_err)

        if best_fitted_model is None:
            # Fallback to simple Logistic Regression
            best_fitted_model = LogisticRegression(class_weight="balanced")
            best_fitted_model.fit(X_train, y_train)
            best_model_name = "Logistic Regression (Fallback)"

        # 6. Evaluate Untouched Test Fold with Best Model
        y_test_pred = best_fitted_model.predict(X_test)
        if hasattr(best_fitted_model, "predict_proba"):
            y_test_prob = best_fitted_model.predict_proba(X_test)[:, 1]
        else:
            y_test_prob = y_test_pred.astype(float)

        test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_prob)

        return {
            "success": True,
            "selected_best_model": best_model_name,
            "test_metrics": test_metrics.to_dict(),
            "models": [
                {k: v for k, v in m.items() if k != "model_object"}
                for m in model_results
            ],
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "fitted_model": best_fitted_model,
            "fitted_columns": fitted_cols,
            "preprocessor": preprocessor,
            "config": {
                "tolerance_min": self.config.tolerance_min,
                "calibration_method": self.config.calibration_method,
                "low_risk_threshold_pct": self.config.low_risk_threshold_pct,
                "high_risk_threshold_pct": self.config.high_risk_threshold_pct
            }
        }


class LatenessClassifierManager:
    """Manages persistence, loading, and inference for user classification models."""

    @staticmethod
    def get_artifact_paths(user_id: int) -> Dict[str, str]:
        os.makedirs(MODELS_DIR, exist_ok=True)
        return {
            "model": os.path.join(MODELS_DIR, f"clf_model_user_{user_id}.pkl"),
            "preprocessor": os.path.join(MODELS_DIR, f"clf_preprocessor_user_{user_id}.pkl"),
            "features": os.path.join(MODELS_DIR, f"clf_feats_user_{user_id}.pkl"),
            "metadata": os.path.join(MODELS_DIR, f"clf_meta_user_{user_id}.pkl")
        }

    @classmethod
    def save_classifier_bundle(
        cls,
        user_id: int,
        model: Any,
        preprocessor: Any,
        feature_columns: List[str],
        metadata: Dict[str, Any]
    ) -> None:
        """Persist classification bundle to disk."""
        paths = cls.get_artifact_paths(user_id)
        joblib.dump(model, paths["model"])
        joblib.dump(preprocessor, paths["preprocessor"])
        joblib.dump(feature_columns, paths["features"])
        joblib.dump(metadata, paths["metadata"])
        logger.info("Saved classification artifacts for user_id=%d", user_id)

    @classmethod
    def load_classifier_bundle(cls, user_id: int) -> Optional[Dict[str, Any]]:
        """Load classification artifacts if present on disk."""
        paths = cls.get_artifact_paths(user_id)
        if not (os.path.exists(paths["model"]) and os.path.exists(paths["metadata"])):
            return None
        try:
            model = joblib.load(paths["model"])
            metadata = joblib.load(paths["metadata"])
            preprocessor = joblib.load(paths["preprocessor"]) if os.path.exists(paths["preprocessor"]) else None
            features = joblib.load(paths["features"]) if os.path.exists(paths["features"]) else None
            return {
                "model": model,
                "metadata": metadata,
                "preprocessor": preprocessor,
                "features": features
            }
        except Exception as e:
            logger.error("Failed to load classification bundle for user_id=%d: %s", user_id, e)
            return None

    @classmethod
    def predict_lateness_risk(
        cls,
        user_id: int,
        event_name: str,
        day_of_week: str,
        importance: str,
        bag_prepared: bool,
        estimated_prep_duration_min: float,
        estimated_travel_duration_min: float,
        config: Optional[LatenessClassificationConfig] = None
    ) -> Dict[str, Any]:
        """Predict departure lateness probability and risk category independently of regression."""
        cfg = config or LatenessClassificationConfig()
        bundle = cls.load_classifier_bundle(user_id)

        # Fallback if no personalized classifier is trained yet
        if bundle is None:
            # Empirical prior based on bag preparation and estimated duration
            base_prob = 0.35
            if not bag_prepared:
                base_prob += 0.20
            if day_of_week == "Monday":
                base_prob += 0.10
            if importance in ("high", "critical"):
                base_prob -= 0.05

            late_prob = max(0.05, min(0.95, base_prob))
            risk_cat = cfg.categorize_probability(late_prob)

            return {
                "late_probability": round(late_prob, 2),
                "risk_category": risk_cat,
                "model_version": "empirical_prior_fallback",
                "model_type": "empirical_prior",
                "is_calibrated": False,
                "confidence_status": {
                    "score": 0.45,
                    "level": "low",
                    "is_reliable": False,
                    "reasons": ["Classification model not yet trained; using empirical heuristic prior."]
                }
            }

        model = bundle["model"]
        preprocessor = bundle.get("preprocessor")
        meta = bundle.get("metadata", {})

        try:
            # Format single inference payload
            if preprocessor is not None and hasattr(preprocessor, "transform_single_inference"):
                X_pred = preprocessor.transform_single_inference({
                    "event_name": event_name,
                    "day_of_week": day_of_week,
                    "importance": importance,
                    "bag_prepared": 1 if bag_prepared else 0,
                    "estimated_prep_duration_min": estimated_prep_duration_min,
                    "estimated_travel_duration_min": estimated_travel_duration_min
                })
            else:
                features = bundle.get("features", [])
                try:
                    from ml.features import preprocess_features
                except ModuleNotFoundError:
                    from backend.ml.features import preprocess_features
                single_row = pd.DataFrame([{
                    "event_name": event_name,
                    "day_of_week": day_of_week,
                    "importance": importance,
                    "bag_prepared": 1 if bag_prepared else 0,
                    "estimated_prep_duration_min": estimated_prep_duration_min,
                    "estimated_travel_duration_min": estimated_travel_duration_min
                }])
                X_pred = preprocess_features(single_row, fit_columns=features)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_pred)[0]
                # Probability of class 1 (late)
                late_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            else:
                pred_label = model.predict(X_pred)[0]
                late_prob = 0.80 if pred_label == 1 else 0.20

            late_prob = max(0.01, min(0.99, late_prob))
            risk_cat = cfg.categorize_probability(late_prob)

            return {
                "late_probability": round(late_prob, 2),
                "risk_category": risk_cat,
                "model_version": meta.get("model_version", "clf_v1"),
                "model_type": meta.get("selected_best_model", "calibrated_classifier"),
                "is_calibrated": meta.get("is_calibrated", True),
                "test_macro_f1": meta.get("test_macro_f1"),
                "confidence_status": {
                    "score": 0.85,
                    "level": "high",
                    "is_reliable": True,
                    "reasons": ["Trained on chronological historical departure observations with probability calibration."]
                }
            }
        except Exception as e:
            logger.error("Classifier inference failed for user_id=%d: %s", user_id, e)
            return {
                "late_probability": 0.50,
                "risk_category": "moderate",
                "model_version": "error_fallback",
                "model_type": "fallback",
                "is_calibrated": False,
                "confidence_status": {
                    "score": 0.20,
                    "level": "low",
                    "is_reliable": False,
                    "reasons": [f"Classification inference error: {e}"]
                }
            }
