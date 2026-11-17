# MicroDelay 2.0 — Adaptive Behavioral Intelligence Architecture
## Foundation & Service Layer Design (Prompt 2.1)

**Package:** `backend/ml/adaptive/`  
**Status:** Foundation Established & Verified  
**Automated Tests:** 13/13 Module & Integration Tests Passing (67/67 Backend Pytest Total)

---

## 1. Architectural Overview

MicroDelay's **Adaptive Behavioral Intelligence Engine** builds directly upon the scientific foundations established in Phase 1 (data provenance, chronological evaluation, leakage-free feature engineering, and versioned model management). The engine transitions MicroDelay from purely descriptive historical models into an autonomous, proactive behavioral assistant that adapts to the user's habit lifecycle.

```mermaid
flowchart TD
    subgraph Client & API Layer
        A[Prediction Request: Event / Departure Context] --> B[FastAPI: /api/intelligence/predict & /api/predict]
    end

    subgraph Context & Validation
        B --> C[PredictionContext: Strict Temporal Causal State]
        C --> D[Context Validation & Sanitization]
    end

    subgraph Central Intelligence Orchestration
        D --> E[AdaptiveIntelligenceOrchestrator]
        E --> F1[Regression Core: Active ML Bundle / Preprocessor]
        
        subgraph Isolated Failure-Protected Components
            E --> F2[LatenessRiskClassifier: Calibrated Probabilities & Risk Tiers]
            E --> F3[PersonalizationEngine: Global -> Blended -> Autonomous]
            E --> F4[BehavioralClusteringEngine: Context Archetypes]
            E --> F5[BehavioralProfileEngine: Biases & Bottlenecks]
            E --> F6[PredictionExplainer: Contextual Drivers]
            E --> F7[AdaptiveRecommendationEngine: Targeted Actions]
            E --> F8[ConfidenceEstimator: Reliability Scoring]
            E --> F9[BehavioralDriftDetector: Habit Shift Monitoring]
        end
    end

    subgraph Standardized Output Contract
        F1 & F2 & F3 & F4 & F5 & F6 & F7 & F8 & F9 --> G[AdaptiveIntelligenceOutput]
        G --> H1[Standardized Intelligence JSON]
        G --> H2[Legacy Backward-Compatibility Adapter]
    end
```

---

## 2. Logical Module Responsibilities

The adaptive intelligence layer is organized within `backend/ml/adaptive/`. Each module is completely decoupled, testable in isolation, and fault-tolerant.

| Module | File | Core Responsibilities |
| :--- | :--- | :--- |
| **Unified Context** | `context.py` | Encapsulates the situational state available strictly at the prediction timestamp (`PredictionContext`). Enforces bounds validation on durations, days, and importance. |
| **Output Contract & Enums** | `schemas.py` | Defines `AdaptiveIntelligenceOutput`, `ModelAvailability`, `RiskCategory`, `ConfidenceLevel`, `ContributingFactor`, and `RecommendedIntervention`. Provides `to_legacy_dict()` for 100% backward compatibility. |
| **Central Orchestrator** | `orchestrator.py` | `AdaptiveIntelligenceOrchestrator` coordinates regression prediction and all analytical components with isolated `try/except` execution boundaries. |
| **Lateness Risk Classifier** | `risk_classifier.py` | `LatenessRiskClassifier` maps continuous predicted delay into discrete risk levels (`low`, `moderate`, `high`, `extreme`) and a calibrated logistic probability $P(\text{delay} > 0)$. |
| **Personalization Engine** | `personalization.py` | `PersonalizationEngine` evaluates data maturity across 4 tiers: `global_cold_start`, `persona_prior`, `bayesian_blended` (1-7 real samples), and `personalized_autonomous` ($\ge 8$ real samples). |
| **Behavioral Clustering** | `clustering.py` | `BehavioralClusteringEngine` matches departure contexts to routine archetypes (`early_morning_rush`, `high_stakes_departure`, `unprepared_commute`, `deliberate_preparation`). |
| **Behavioral Profile** | `behavioral_profile.py` | `BehavioralProfileEngine` aggregates persistent user traits without lookahead leakage: estimation bias, item search frequency, phone distraction frequency, and dominant bottleneck. |
| **Prediction Explainer** | `explainability.py` | `PredictionExplainer` identifies top contextual and behavioral drivers (e.g. unprepared bag impact, Monday inertia, model delta) contributing to departure delay. |
| **Adaptive Recommendations** | `adaptive_recommendations.py` | `AdaptiveRecommendationEngine` generates targeted, actionable habit interventions prioritized by estimated minutes saved. |
| **Behavioral Drift Detector** | `drift_detection.py` | `BehavioralDriftDetector` compares recent departure windows against baseline distributions to detect improving, degrading, or stable behavioral shifts. |
| **Confidence Estimator** | `confidence.py` | `ConfidenceEstimator` evaluates prediction trustworthiness score $[0.0, 1.0]$ based on real sample counts, extrapolation distance, and historical model validation MAE. |

---

## 3. Unified Contracts

### 3.1 Prediction Context (`PredictionContext`)
Contains **only** information available at the prediction timestamp:
```python
@dataclass
class PredictionContext:
    user_id: int
    event_name: str
    day_of_week: str
    event_importance: str  # "high", "medium", "low", "critical"
    estimated_prep_duration: float
    estimated_travel_duration: float
    bag_prepared: bool
    event_id: Optional[int] = None
    prediction_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    historical_behavior_features: Dict[str, Any] = field(default_factory=dict)
    active_model_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 3.2 Output Contract (`AdaptiveIntelligenceOutput`)
Standardized response contract for all consumers:
```json
{
  "predicted_delay_minutes": 5.2,
  "baseline_predicted_delay_minutes": 4.5,
  "late_probability": 0.72,
  "risk_category": "high",
  "contributing_factors": [
    {
      "factor": "bag_unprepared",
      "impact_minutes": 3.8,
      "direction": "increases_delay",
      "description": "Leaving bag packing to morning of departure increases item search and packing delays."
    }
  ],
  "recommended_interventions": [
    {
      "action": "Evening Bag Packing Checklist",
      "description": "Pack your bag, chargers, keys, and daily essentials the night before.",
      "expected_delay_reduction_min": 3.8,
      "category": "preparation",
      "priority": "high"
    }
  ],
  "confidence_status": {
    "score": 0.85,
    "level": "high",
    "is_reliable": true,
    "reasons": ["Calibrated personal data sample size (25 genuine departure observations)."]
  },
  "model_version": "rf_user_1_20261116_120000",
  "model_type": "random_forest",
  "model_availability": "available",
  "data_quality_status": {
    "provenance": "real",
    "sample_count": 25,
    "is_sufficient": true,
    "has_real_data": true
  },
  "fallback_applied": false,
  "component_status": {
    "regression": "success",
    "risk_classification": "success",
    "explainability": "success",
    "recommendations": "success",
    "confidence": "success",
    "drift_detection": "success"
  },
  "warnings": [],
  "execution_timestamp": "2026-11-16T23:00:00Z"
}
```

---

## 4. Operational Guarantees

### 4.1 Explicit Model Availability States
The engine tracks 5 distinct availability states (`ModelAvailability`):
1. `AVAILABLE`: Verified active model loaded and ready for personalized or calibrated inference.
2. `TRAINING`: Model retraining currently underway in background; fallback or baseline serving requests.
3. `INSUFFICIENT_DATA`: User has 0 real and 0 synthetic records; baseline heuristic serving requests.
4. `UNAVAILABLE`: Model artifacts missing from disk; safe fallback applied.
5. `FAILED`: Model inference encountered an unexpected numerical or pipeline error; fallback applied.

### 4.2 Component Failure Isolation
If any sub-component encounters an exception (e.g. explainability factor calculation or drift detector):
- The exception is trapped, logged as a warning, and recorded in `component_status[<component>] = "failed: <error>"`.
- The regression delay prediction and risk estimation continue unaffected.
- The user receives a valid response without experiencing HTTP 500 crashes.

### 4.3 Backward Compatibility
- `/api/predict` uses `AdaptiveIntelligenceOutput.to_legacy_dict()` to preserve existing response fields (`predicted_delay_min`, `baseline_predicted_delay_min`, `model_used`, `model_version`, `data_provenance`, `training_mode`, `is_personalized_real`) while appending adaptive fields (`late_probability`, `risk_category`, `model_availability`).
- Dedicated endpoints `/api/intelligence/predict` (GET & POST) provide the full schema for modern clients.

### 4.4 Privacy & Sensitive Data Protection
- Operational logs record only numerical and category identifiers (`user_id`, `predicted_delay`, `risk_category`, `confidence_score`, `latency_ms`).
- Personal notes, passwords, private metadata, and freeform text are never serialized into model artifacts or server logs.

---

## 5. Verification & Testing

The adaptive intelligence architecture is verified by `backend/tests/test_adaptive_intelligence.py` covering:
- Prediction context sanitization & out-of-bounds rejection.
- Standardized output serialization and backward compatibility mapping.
- Calibrated probability calculations across negative, moderate, and extreme delays.
- 4-tier data maturity transitions.
- Behavioral clustering and archetype assignments.
- Profile building from raw logs (bias, search frequency, bottlenecks).
- Explainer and recommendation engine synthesis.
- Behavioral drift detection on stable vs drifting histories.
- Confidence scoring with sample size, event familiarity, and MAE adjustments.
- Component failure isolation with injected mock runtime exceptions.
- Cold-start and non-existent user handling.
- Privacy scrubbing of user metadata.
- End-to-end FastAPI endpoint integration (`/api/predict` & `/api/intelligence/predict`).

---

## 6. Personalized Lateness Risk Classification Engine (Prompt 2.2)

### 6.1 Operational Lateness Definition & Target Formulation
MicroDelay separates the continuous delay duration ($y \in \mathbb{R}$) from the discrete probability of departing late ($P(Y > \tau)$):
$$\text{departure\_delay} = \text{actual\_departure} - \text{planned\_departure}$$
$$Y_{\text{binary}} = \begin{cases} 1, & \text{if } \text{departure\_delay} > \tau \\ 0, & \text{if } \text{departure\_delay} \le \tau \end{cases}$$
where $\tau = 0.0$ minutes is the default configurable tolerance threshold. Late departure is strictly distinguished from destination arrival delay (traffic or transit disruptions en route).

### 6.2 Classification Benchmark Suite & Parsimonious Selection
The classification engine evaluates 4 distinct algorithms over identical chronological holdout partitions:
1. **Logistic Regression:** Regularized linear probabilistic baseline (`class_weight='balanced'`).
2. **Random Forest Classifier:** Non-linear bagged ensemble (`n_estimators=50, max_depth=5, class_weight='balanced'`).
3. **HistGradientBoosting Classifier:** Fast tree booster with sample weighting (`class_weight='balanced'`).
4. **XGBoost Classifier:** Regularized gradient booster with positive class weighting (`scale_pos_weight`).

Models are ranked and selected based on **Validation Macro F1** to ensure equal penalization of false alarms and missed delays under class imbalance.

### 6.3 Strictly Proper Probability Calibration
Raw model outputs (e.g. tree fraction votes or uncalibrated margins) undergo probability calibration via **Platt Scaling (Sigmoid CalibratedClassifierCV)**. Platt calibration is only retained if the validation **Brier Score** ($\text{Brier} = \frac{1}{N}\sum (p_i - y_i)^2$) improves or preserves calibration quality ($\text{Brier}_{\text{cal}} \le \text{Brier}_{\text{raw}} + 0.02$).

### 6.4 Operational Risk Tiers
Calibrated probabilities are mapped to actionable risk thresholds:
- **Low Risk ($< 30\%$):** Normal departure; standard routine prompts.
- **Moderate Risk ($30\% - 60\%$):** Elevated vulnerability; proactive distraction warning.
- **High Risk ($60\% - 85\%$):** High likelihood of delay; targeted intervention prompts (e.g., bag checklist).
- **Extreme Risk ($> 85\%$):** Critical lateness expected; urgent departure acceleration.

### 6.5 Edge-Case Safeguards & Independence
- **Single-Class Protection:** If training data contains only on-time or only late records, the pipeline falls back gracefully to empirical historical rates without numerical singularity.
- **Independence:** Regression delay models and classification risk models are persisted independently (`clf_model_user_{id}.pkl`, `clf_preprocessor_user_{id}.pkl`, `clf_feats_user_{id}.pkl`, `clf_meta_user_{id}.pkl`), allowing isolated upgrades, retraining, and rollbacks.
- **API Endpoints:** Dedicated GET/POST `/api/intelligence/risk` and `/api/models/classifier/metrics`.

---

## 7. Global-to-Personalized Learning Architecture (Prompt 2.3)

### 7.1 Cold-Start Mitigation & Prediction Hierarchy
Newly registered accounts lack genuine personal observations ($N_{\text{real}} = 0$). Relying solely on per-user models either forces synthetic-only training or fails completely. MicroDelay solves this through a three-level learning architecture:
1. **Level A — Global Cohort Model:**
   - Learns cross-user behavioral patterns and departure heuristics across consenting users' genuine observations.
   - Serves initial predictions during cold start when individual historical records are insufficient.
2. **Level B — Personalized Model:**
   - Trains exclusively on an individual user's genuine records ($N_{\text{real}} \ge 8$).
   - Captures idiosyncratic morning routines, personal preparation bias, and specific event vulnerabilities.
3. **Level C — Dynamic Hybrid Blended Prediction:**
   - Dynamically blends global population patterns with emergent individual patterns:
     $$\hat{y}_{\text{hybrid}} = \alpha \cdot \hat{y}_{\text{global}} + (1 - \alpha) \cdot \hat{y}_{\text{personal}}$$
   - $\alpha \in [0.0, 1.0]$ is computed dynamically rather than assigned as a static heuristic.

### 7.2 Dynamic Blending Weight ($\alpha$) Formulation
The blending weight $\alpha$ dynamically transitions from global reliance ($\alpha \to 1.0$) to individual autonomy ($\alpha \to 0.0$):
$$\alpha = \text{clip}\left( \left(1.0 - \frac{N_{\text{real}}}{8}\right) \cdot \mu_{\text{perf}} \cdot \mu_{\text{stab}}, \; 0.05, \; 0.95 \right)$$
where:
- **Sample Maturity Base:** $1.0 - \frac{N_{\text{real}}}{8}$ smoothly steps down weight as genuine observations accumulate toward the promotion threshold ($N=8$).
- **Performance Multiplier ($\mu_{\text{perf}}$):** If personal model validation MAE exceeds global model validation MAE ($\text{MAE}_{\text{personal}} > \text{MAE}_{\text{global}}$), $\mu_{\text{perf}} > 1.0$ increases global reliance; when personal model outperforms global, $\mu_{\text{perf}} < 1.0$ accelerates personal adoption.
- **Stability Multiplier ($\mu_{\text{stab}}$):** Elevated behavioral variance ($\sigma_{\text{delay}} > 6.0$ minutes) dampens personal reliance and leverages global smoothing; low variance ($\sigma_{\text{delay}} < 2.5$ minutes) signals stable routines, accelerating personal weighting.

### 7.3 Lifecycle States & Cold-Start Progression
The personalization engine transitions users across 6 explicit lifecycle phases:
- `new_user`: 0 genuine samples, global model unavailable (serves fallback baseline 4.5m).
- `collecting_history`: User is actively logging checkouts, but genuine samples remain below cold-start activation threshold ($N_{\text{real}} < 3$) and no global model exists.
- `global_prediction`: $N_{\text{real}} < 3$ or personal model inactive, but consented global cohort model is active ($\alpha = 1.0$).
- `personalizing`: $3 \le N_{\text{real}} < 8$; hybrid dynamic blending is active, smoothly shifting $\alpha$ as genuine samples accumulate.
- `personalized`: $N_{\text{real}} \ge 8$; personal model is fully qualified, approved, and active ($\alpha = 0.0$).
- `insufficient_data`: Zero records and missing artifacts; graceful baseline fallback.

### 7.4 Strict Privacy & Security Safeguards
Cross-user learning is guarded by mathematical privacy boundaries:
1. **Explicit User Consent & Opt-Out:**
   - Controlled via `User.consent_to_global_training: bool` (default: True with 1-click opt-out toggle).
   - Opting out immediately excludes the user's records from all subsequent global training datasets.
2. **Direct Identifier Scrubbing:**
   - Identity columns (`user_id`, `username`, `email`, personal tags) are strictly excluded from training feature vectors.
   - Vectorization uses purely anonymized contextual features (`day_of_week`, `event_importance`, `bag_prepared`, `estimated_prep_duration_min`, `estimated_travel_duration_min`).
3. **Synthetic Quarantine:**
   - Synthetic demonstration observations (`is_synthetic == True`) are strictly filtered out (`is_synthetic == False` AND `provenance == 'real'`). Synthetic data never contaminates global cohort models.
4. **Minimum Cohort Threshold ($K$-Anonymity Guard):**
   - Global model training aborts safely unless at least 2 distinct consented users and at least 10 genuine observations are present, preventing single-user reverse-engineering.

### 7.5 Leave-Users-Out Cross-Validation & Scientific Generalization
To prove that global models generalize to unseen individuals without demographic stereotyping:
- **Evaluation Protocol:** For each consenting user $u \in U$, a model is trained on observations from $U \setminus \{u\}$ and tested strictly on $u$'s held-out records.
- **Metrics Tracked:** Individual holdout MAE and cohort-level aggregated transferability MAE.
- **Group Evaluation:** Leave-users-out performance is segmented across user profile groups (Student, Teacher, Professor, IT Professional) to empirically confirm cross-user transferability.

### 7.6 API & UI Integration
- `GET /api/users/consent` & `POST /api/users/consent`: Inspect and toggle global data sharing consent.
- `GET /api/models/global/status`: Inspect global model availability, version, cohort size, and validation metrics.
- `POST /api/models/global/train`: Train and safely evaluate a new candidate global cohort model.
- `GET /api/benchmarks/leave-users-out`: Retrieve out-of-sample transferability and group cross-validation results.
- `frontend/src/App.tsx`: Real-time lifecycle badge, maturity progress meter ($N_{\text{real}}/8$), dynamic hybrid blending breakdown bar, cohort benchmarking table, and privacy consent controls.


