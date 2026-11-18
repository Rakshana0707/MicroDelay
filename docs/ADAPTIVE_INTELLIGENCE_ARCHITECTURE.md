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

---

## 8. Behavioral Habit Discovery & Clustering Engine (Prompt 2.4)

### 8.1 Objective & Paradigm Shift
Rather than relying exclusively on predefined demographic personas (e.g. "student" or "software engineer"), MicroDelay 2.0 introduces an unsupervised habit discovery engine that clusters recurring behavioral signatures directly from genuine historical observations.

### 8.2 Standard 12-Dimensional Behavioral Feature Space
User-level behavioral feature vectors are extracted chronologically without lookahead leakage:
1. `average_prep_duration`: Mean actual preparation time in minutes.
2. `average_prep_bias`: Mean planning bias ($\text{actual\_prep} - \text{estimated\_prep}$).
3. `average_departure_delay`: Mean delay relative to scheduled leave time.
4. `lateness_rate`: Proportion of departures resulting in delay ($> 0.0$m).
5. `phone_distraction_frequency`: Proportion of departures interrupted by phone or notifications.
6. `item_search_frequency`: Proportion of departures delayed searching for keys, badges, or essentials.
7. `last_minute_task_frequency`: Proportion of departures delayed by "one last task" right at door.
8. `bag_prepared_percentage`: Proportion of mornings with materials prepared the night before.
9. `average_travel_error`: Mean estimation error for travel/commute buffer.
10. `weekday_delay_variation`: Standard deviation of delays across days of the week.
11. `preparation_duration_variability`: Standard deviation of actual preparation time.
12. `recent_behavior_change`: Delay difference between recent 5 checkouts versus older baseline.

### 8.3 Unsupervised Clustering Tournament & Selection
The engine evaluates and compares three clustering families over standardized feature vectors (`StandardScaler`):
- **K-Means:** $k \in [2, 5]$, minimizing inertia with orthogonal initialization (`n_init=10`).
- **Agglomerative Hierarchical Clustering:** Ward's minimum variance criterion over candidate $k$.
- **DBSCAN:** Density-based clustering with adaptive epsilon sweeps to identify non-spherical clusters and isolate noise.

Model selection is strictly evidence-based, ranking candidates using:
- **Silhouette Score ($S \in [-1, 1]$):** Cluster cohesion and separation.
- **Davies-Bouldin Index ($DB \ge 0$):** Ratio of within-cluster distance to between-cluster distance (lower is better).
- **Size Distribution Balance:** Penalizes degenerate singletons or extreme skew.

### 8.4 Evidence-Based Dynamic Interpretation Service
Cluster assignments are never hardcoded by cluster ID numbers. The `ClusterInterpretationService` dynamically projects centroids back into interpretable feature units, computes $z$-scores relative to the population mean ($\mathbf{\mu}_{\text{pop}}, \mathbf{\sigma}_{\text{pop}}$), and synthesizes evidence-based habit labels:
- **Consistently Prepared Routine:** High overnight bag preparedness ($\ge 70\%$), low departure delay ($\le 1.5$m), minimal distractions.
- **Phone-Interrupted Preparation:** Dominant phone distraction frequency ($\ge 30\%$).
- **Frequent Item-Search Delays:** Dominant item searching at departure door ($\ge 30\%$).
- **Last-Minute Task Driven:** High "one last task" delays at scheduled leave deadline ($\ge 30\%$).
- **Preparation Underestimation Bias:** Planning fallacy where actual prep exceeds planned by $\ge 3.0$m.
- **Travel-Time Optimism:** Commute buffer underestimated by $\ge 3.0$m.
- **Inconsistent Routine Timeline:** Volatile prep duration variability ($\ge 6.0$m).
- **Standard Variable Routine:** Balanced baseline variance.

### 8.5 Lifecycle Sufficiency & Historical Profile Evolution
- **Sufficiency States:**
  - `insufficient_data`: $< 3$ observations. Baseline guidance provided.
  - `gathering_history`: $3 \le N < 8$ observations. Provisional habit cluster assigned.
  - `sufficient`: $\ge 8$ observations. Fully mature behavioral habit profile.
- **Nearest Centroid Assignment:** Unseen users and algorithms without natural inference methods (Agglomerative) are mapped via Euclidean distance to the closest standardized centroid.
- **Historical Comparison & Trend Tracking:** Profiles are versioned in `user_cluster_profiles` with timestamps. Successive generations evaluate:
  - Habit shifts (e.g. "Shifted from Preparation Underestimation to Consistently Prepared").
  - Delay delta ($\Delta \text{delay}$) and trend classification (`improving`, `regressing`, `stable`).
  - Recent behavior change metrics.

### 8.6 API & Frontend Visualizations
- `GET /api/intelligence/clusters/profile`: Current user's behavioral profile, supporting stats, and trend comparison.
- `POST /api/intelligence/clusters/train`: Triggers clustering tournament across genuine records and persists artifacts.
- `GET /api/intelligence/clusters/benchmark`: Comparative benchmark metrics across K-Means, Agglomerative, and DBSCAN.
- `frontend/src/App.tsx`: Real-time Discovered Behavioral Habit card on Dashboard, metric comparison grid (User vs Population), trend badge, and Tournament Benchmark matrix on Model Intelligence tab.

---

## 9. Personalized Behavioral Profiling Engine (Prompt 2.5)

### 9.1 Objective & Philosophical Shift
MicroDelay 2.0 moves beyond static persona assignments by continuously maintaining an evidence-based **Personalized Behavioral Profile** for every user. Instead of relying solely on predefined labels (e.g. "student", "IT professional"), the profiling engine dynamically derives the user's preparation habits, distraction tendencies, organization discipline, and estimation accuracy directly from their genuine checkout history.

### 9.2 Six Foundational Behavioral Dimensions
The profiler calculates granular statistics across six behavioral dimensions:

1. **Dimension A: Preparation Behavior**
   - Average actual preparation duration ($\text{avg\_actual\_prep\_min}$).
   - Preparation time bias ($\text{prep\_bias\_min} = \text{actual\_prep} - \text{estimated\_prep}$).
   - Preparation duration variability ($\sigma_{\text{prep}}$ standard deviation).
   - Preparation consistency score ($1 - \min(1, \text{CV}_{\text{prep}})$).

2. **Dimension B: Distraction Patterns**
   - Phone usage frequency ($\%$ of checkouts with recorded phone distraction).
   - Average phone interruption overhead ($\text{avg\_phone\_interruption\_min}$).
   - Most common interruption window (e.g. *"Final 15 min of morning preparation"*).
   - Historical distraction trend.

3. **Dimension C: Organization & Readiness Behavior**
   - Doorway item-search frequency ($\%$ with misplaced keys, badges, transit cards).
   - Item search delay impact ($\text{avg\_item\_search\_delay\_min}$).
   - Overnight bag preparation consistency ($\%$ packed the night before).
   - Recurring missing item pattern.

4. **Dimension D: Time Estimation Accuracy**
   - Preparation estimation bias ($\text{actual\_prep} - \text{estimated\_prep}$).
   - Travel estimation bias ($\text{actual\_travel} - \text{estimated\_travel}$).
   - Event-specific prep bias breakdown across distinct routine categories.

5. **Dimension E: Departure Habits**
   - Average departure delay / lateness ($\text{avg\_lateness\_min}$).
   - Late departure frequency ($\% > 0\text{m}$).
   - Early / on-time departure frequency ($\% \le 0\text{m}$).
   - Day-of-week departure patterns (Monday through Sunday delay and lateness rate breakdown).

6. **Dimension F: Routine Consistency**
   - Schedule adherence rate ($\%$ departures matching planned departure time).
   - Weekday vs weekend delay variation ($\text{weekday\_delay\_std}$).
   - Recent trend shift ($\text{recent}_5 - \text{older}$ average delay difference).

### 9.3 Multi-Window Rolling Calculations & Temporal Safeguards
To avoid temporal bias and prevent transient disruptions from distorting long-term habits, the profiling engine calculates metrics across three rolling horizons relative to the user's latest recorded checkout:
- **All-Time Baseline:** Historical baseline across all valid records.
- **30-Day Window (`30d`):** Medium-term habits and recent trajectory. Requires $N \ge 3$ checkouts.
- **7-Day Window (`7d`):** Short-term operational focus. Requires $N \ge 3$ checkouts.

If sample size within a window is $< 3$, the system explicitly suppresses trend assertions and flags the window as `is_sufficient: false`.

### 9.4 Data Sufficiency Lifecycle
To prevent premature or misleading claims, the engine enforces three rigorous sufficiency states:
- `insufficient_data` ($N < 3$ observations): Profiles are withheld; UI instructs user to record additional checkouts.
- `gathering_history` ($3 \le N < 8$ observations): Early calibration profile active. Informs user that stable personalization requires 8 checkouts.
- `sufficient` ($N \ge 8$ observations): Mature, statistically defensible behavioral profile.

### 9.5 Evidence-Backed Insight Generation
The insights generator is strictly bounded by underlying data. It enforces the following scientific guardrails:
- **No unsupported causal leaps:** Does not assert that phone usage caused lateness unless phone delay was directly observed.
- **Strictly factual statements:** E.g., *"You underestimate your preparation time by an average of 4.5 minutes."*, *"Your departure delays are elevated on Mondays (averaging 6.2 minutes late across 4 checkouts)."*
- **Traceable supporting metrics:** Every generated finding carries the exact numeric metric and factual evidence string.

### 9.6 Versioning, Caching, and Delta Comparisons
- **Persistence:** Profiles are versioned and stored in `behavioral_profiles` (`BehavioralProfileSnapshot` model) with UUID-timestamped identifiers (`usr_{id}_prof_{YYYYMMDD_HHMMSS_ffffff}`).
- **Caching & Invalidation:** Active profiles are cached. When $\ge 3$ new valid checkouts accumulate, the engine automatically recalculates and persists a new snapshot. Users can also trigger explicit forced recalculation via `POST /api/intelligence/profile/refresh`.
- **Period Comparison:** `PersonalizedBehavioralProfiler.compare_profile_periods` computes numeric deltas ($\Delta \text{prep bias}$, $\Delta \text{lateness}$, $\Delta \text{phone distraction}$, $\Delta \text{bag readiness}$, $\Delta \text{adherence}$) and classifies the overall shift as `improving`, `regressing`, or `stable`.

### 9.7 API Endpoints & Frontend UI
- `GET /api/intelligence/profile`: Returns active profile with 6 dimensions, rolling windows, sufficiency status, and insights.
- `GET /api/intelligence/profile/history`: Returns audit trail of historical profile snapshots.
- `GET /api/intelligence/profile/compare`: Compares two snapshots or rolling windows (`window_a` vs `window_b`), with user access isolation.
- `POST /api/intelligence/profile/refresh`: Forces recalculation of profile snapshot.
- **Frontend UI (`activeTab === 'profile'`):** Dedicated **Behavioral Profile** tab in React featuring sufficiency badges, window selectors (`All-Time`, `30 Days`, `7 Days`), 6 dimension metric cards, evidence-backed insight list, delta comparison view, and snapshot audit history table.

---

## 10. Prediction Explainability & Behavioral Reasoning Engine (Prompt 2.6)

### 10.1 Objective & Architectural Foundation
To prevent users from receiving uninterpretable black-box outputs (e.g. an isolated "78% chance of being late" or "+6.2m delay"), MicroDelay 2.0 incorporates a dedicated **Prediction Explainability & Behavioral Reasoning Engine** (`backend/ml/explainability.py`).

The engine bridges statistical machine learning outputs with intuitive human reasoning by decomposing predictions into:
1. **Mathematical Feature Attribution:** Exact directional influence of situational variables on predicted minutes and probabilities.
2. **Behavioral Habit Comparison:** Empirical contrast between situational inputs and the user's validated historical baseline (established via Prompt 2.5).
3. **Epistemic Source Tagging:** Explicit labeling of the source and reliability tier for every claim.
4. **Scientific Causation Guardrails:** Rigorous distinctions between statistical predictive correlation and physical/biological causation.

---

### 10.2 Model-Specific Local Explanation Decompositions
Different algorithms require mathematically distinct explanation techniques rather than a single uniform heuristic:

1. **Tree-Based Models (Random Forest Regressor & Classifier):**
   - **Primary Method (`tree_shap`):** Attempts Shapley Additive Explanations via `shap.TreeExplainer` when compatible libraries are present.
   - **Graceful Fallback (`tree_feature_contribution`):** In environments where `shap` is absent or encounters version incompatibilities, the explainer decomposes predictions by calculating per-feature node contribution across decision tree decision paths ($w_i = \text{val}_i \cdot \text{feature\_importance}_i$), ensuring 100% inference resilience without crashing.

2. **Linear & Logistic Regression (`linear_coefficients`):**
   - Standardized coefficient attribution ($w_i \cdot x_i$), measuring exact linear pushes above or below intercept bias for both log-odds and continuous delays.

3. **Cold-Start & Heuristic Baselines (`behavioral_rule_reasoning`):**
   - Prioritized behavioral rules applied to situational inputs (e.g., bag unreadiness penalties, Monday departure lag, prep time compression).

---

### 10.3 Historical Behavioral Context Comparator
Local prediction explanations do not merely interpret model weights; they anchor predictions in personal behavioral truth:
- **Preparation Underestimation:** Contrasts planned prep duration with user's historical average actual prep time (e.g., "Planned prep duration of 20.0m is 12.4m shorter than your historical average of 32.4m").
- **Overnight Bag Preparation:** Contrasts bag state with historical pack rate (e.g., "Bag is not packed; historically you leave bags unprepared 65% of the time, causing doorway bottlenecks").
- **Monday Morning Commute Inertia:** Flags weekday friction when planning high-stakes departures on historically difficult days.

---

### 10.4 Global Feature Importance Sensitivity Analysis
- **Engine (`GlobalFeatureImportanceCalculator`):** Evaluates overall model sensitivity across all historical checkouts.
- **Algorithms:** Computes Permutation Importance on validation datasets where sample counts permit, falling back to Mean Decrease in Impurity (Gini MDI) for Random Forest or normalized coefficients for Logistic Regression.
- **Visual Presentation:** Exported via `GET /api/intelligence/features/importance` and visualized with Recharts horizontal sensitivity bar charts.

---

### 10.5 Epistemic Source Tagging System
Every contributing factor and comparison is tagged with a strict epistemic source badge:
- `directly_observed_behavior`: Explicit binary/continuous state directly provided by user checkout (e.g. `bag_prepared = False`).
- `model_contribution`: Pure mathematical ML coefficient or Tree SHAP weight contribution.
- `statistical_association`: Empirical correlation discovered from personal historical checkouts.
- `estimated_behavioral_influence`: Heuristic or domain-knowledge estimate applied under cold-start conditions.

---

### 10.6 Scientific Guardrails & Causation Disclaimer
All explanation outputs enforce a strict scientific disclaimer:
> *"Scientific Notice: Contributing factors reflect mathematical model weights and empirical associations observed across historical checkouts. They do not constitute deterministic or biological causation. High feature importance signifies predictive correlation, not guaranteed outcome."*

Furthermore, unmeasured environmental variables (sudden traffic bottlenecks, severe weather events, public transit cancellations) are explicitly listed under `limitations`.

---

### 10.7 Persistence, Schema, and Security Isolation
- **Database Table:** `prediction_explanations` (`PredictionExplanationRecord` in `backend/database.py`).
- **Audit Fields:** `prediction_id`, `user_id`, `model_version`, `feature_schema_version`, `explanation_method`, `predicted_delay_minutes`, `late_probability`, `top_positive_contributors`, `top_mitigating_factors`, `historical_comparisons`, `limitations`, `causation_disclaimer`, `feature_audit_values`, `created_at`.
- **User Isolation:** All explanation lookups (`models.get_user_prediction_explanation`, `models.get_user_explanation_history`) strictly filter by `user_id == current_user.id`. User A can never query or view User B's explanation records.

---

### 10.8 API Endpoints & React Frontend Integration
- **Backend Endpoints:**
  - `GET /api/intelligence/risk`: Now returns comprehensive `explanation` payload along with risk probability.
  - `POST /api/intelligence/risk` & `POST /api/intelligence/predict`: Return full explanation contract.
  - `GET /api/intelligence/explain`: Retrieves existing prediction explanation by `prediction_id` or generates on-demand explanation for given situational parameters.
  - `GET /api/intelligence/features/importance`: Global feature importance ranking for active model.
  - `GET /api/intelligence/explanations/history`: User's audit trail of past explanations.
- **Frontend UI Integration (`frontend/src/App.tsx`):**
  - **Morning Tracker / Simulator:** Expandable **"Why This Prediction?"** card with method badge, factor filter buttons (`All`, `Delay Increasers`, `Mitigating Factors`), epistemic source chips, personal baseline comparisons, and scientific disclaimer.
  - **Model Intelligence Tab:** Global Feature Importance ranking with Recharts bar chart, human-accessible feature sensitivity breakdown, and historical prediction explanations audit table.

---

## 11. Adaptive Personalized Recommendation & Habit Intervention Engine (Prompt 2.7)

### 11.1 Architecture & Problem Statement
Prior recommendation solutions in MicroDelay relied on static persona-based rules (e.g. generic tips for "students" or "professors"). MicroDelay 2.0 transforms this into an **evidence-driven, closed-loop behavioral intervention engine**. 
Instead of generic advice, recommendations are strictly diagnosed from an individual's actual historical departure checkouts, tracked through an active trial lifecycle, evaluated with pre/post statistical deltas, and adaptively re-prioritized based on empirical success, user feedback, and dismissal fatigue cooling.

```mermaid
flowchart TD
    Logs["User Departure Logs & Checkout History"] --> Diag["RecommendationEligibilityEvaluator"]
    Diag --> Catalog[("14-Item Structured Catalog")]
    Catalog --> Prioritizer["AdaptivePrioritizer"]
    Prioritizer -->|"Multi-Criteria Scoring"| RankedRecs["Ranked Personalized Interventions"]
    RankedRecs --> UserUI["Frontend Habits & Interventions UI"]
    UserUI -->|"Accept Habit"| ActiveTrial["Active Intervention Trial + Baseline Snapshot"]
    ActiveTrial --> PostCheckouts["Subsequent Departure Checkouts"]
    PostCheckouts --> Complete["Complete Intervention + Rating & Feedback"]
    Complete --> Evaluator["EffectivenessMeasurementService"]
    Evaluator -->|"Delta Delay, Delta Lateness Rate, Confounders"| Outcome[("Measured Outcome Record")]
    Outcome -->|"Effectiveness Boost"| Prioritizer
    UserUI -->|"Dismiss / Snooze"| FatigueCooldown["Dismissal Cooling & Snooze Filter"]
    FatigueCooldown --> Prioritizer
```

---

### 11.2 Structured 14-Item Recommendation Catalog Across 4 Domains
The catalog (`backend/ml/recommendations.py:RECOMMENDATION_CATALOG`) provides 14 structured, actionable interventions:

1. **Preparation Habits:**
   - `prep_bag_previous_night`: Pack bag and equipment the previous evening (target: unprepared bag doorway bottlenecks).
   - `prep_fixed_item_location`: Designate a permanent doorway drop tray for keys, badges, and chargers (target: frequent item search delays).
   - `prep_clothing_in_advance`: Select attire and outerwear before sleeping (target: morning wardrobe decision delays).
   - `prep_materials_digital_sync`: Sync essential devices, laptops, and chargers overnight (target: device charging/file search delays).

2. **Distraction Mitigation:**
   - `distract_prep_focus_mode`: Enable Preparation Do-Not-Disturb on mobile device (target: phone usage during prep).
   - `distract_delay_phone_check`: Defer social media, chat, and news feeds until after departure (target: social media rabbit holes).
   - `distract_reduce_notifications`: Mute non-critical app alerts during departure windows (target: notification interruptions).

3. **Time Estimation Calibration:**
   - `time_increase_prep_buffer`: Add a calibrated buffer to planned prep based on empirical prep bias (target: chronic prep underestimation).
   - `time_use_historical_prep_avg`: Substitute subjective estimates with empirical historical average prep duration (target: optimism bias).
   - `time_realistic_travel_buffer`: Expand transit margin by 15-20% (target: underestimated commute delays).
   - `time_monday_morning_rule`: Add specific 10-15 minute padding for high-stakes Monday departures (target: Monday morning inertia).

4. **Routine Consistency:**
   - `routine_start_recurring_earlier`: Shift alarm and prep start 10 minutes earlier for high-priority recurring events.
   - `routine_prepare_materials_before_sleep`: Stage next morning's materials before going to sleep.
   - `routine_fixed_departure_sequence`: Establish a standard 5-step doorway sequence (jacket -> bag -> keys -> check -> leave).

---

### 11.3 Behavioral Problem Diagnosis & Eligibility Linking
Every intervention is strictly conditioned on diagnosed empirical bottlenecks (`RecommendationEligibilityEvaluator`):
- **Item Searches:** Triggered when user has $\ge 2$ logs with `tag_finding_items = True` or `delay_finding_items_min > 2.0`.
- **Preparation Underestimation:** Triggered when mean prep bias ($\text{actual} - \text{planned prep}$) exceeds $+3.0$ minutes in $\ge 3$ checkouts.
- **Phone Distraction:** Triggered when user records $\ge 2$ logs with `tag_phone_usage = True` or `delay_phone_usage_min > 3.0`.
- **Doorway Friction:** Triggered when user leaves bag unprepared in $\ge 40\%$ of historical logs.
- **Monday Inertia:** Triggered when Monday departure delay is $\ge 4.0$ minutes higher than mid-week averages.

Each recommendation carries a detailed `supporting_evidence` audit payload:
```json
{
  "problem_frequency_pct": 80.0,
  "observed_overhead_min": 4.5,
  "sample_count": 5,
  "affected_sessions": 4,
  "metric_name": "delay_finding_items_min",
  "supporting_statement": "Triggered by frequent item searches (keys, wallet, badge) in 80% of your departure routines."
}
```

---

### 11.4 Adaptive Multi-Criteria Prioritization Formula
Eligible suggestions are prioritized dynamically using an adaptive scoring formula:

$$\text{Priority Score} = (\text{Evidence Weight} \times \text{Expected Reduction}) \times \text{Effort Multiplier} \times \text{Past Boost} \times \text{Fatigue Penalty}$$

Where:
- **Evidence Weight:** Problem frequency percentage scaled to range $[0.5, 1.0]$.
- **Expected Delay Reduction:** Base catalog reduction calibrated by personal observed overhead.
- **Effort Multiplier:** High effort = $0.70$, Medium = $0.85$, Low = $1.00$.
- **Past Effectiveness Boost:**
  - $+20\%$ boost if the user rated the intervention $\ge 4$ stars in past trials.
  - $+25\%$ boost if past empirical evaluation produced a `success` label ($\ge 3.0$ min delay reduction).
- **Fatigue Cooling Penalty:**
  - If dismissed $< 14$ days ago: score multiplied by $(1.0 - 0.70 \times (1 - \text{days\_since} / 14))$.
  - If currently snoozed ($\text{now} < \text{snoozed\_until}$): score set to $0.0$ (completely hidden).
  - If currently active in progress: excluded from eligible suggestions list.

---

### 11.5 Intervention Lifecycle & State Machine
Interventions are managed via persistent records (`UserInterventionRecord` mapped to table `user_interventions` in `backend/database.py`):
```
[suggested] ---> [accepted / started] ---> [completed (evaluated)]
     |                    |
     +---->[dismissed]    +---->[dismissed]
     |
     +---->[snoozed (7-14d)]
```

When an intervention is accepted:
1. `started_at` is timestamped.
2. An immutable snapshot of the user's `baseline_metrics` is captured:
   - `avg_departure_delay_min`
   - `lateness_rate`
   - `avg_prep_bias_min`
   - `sample_count`

---

### 11.6 Empirical Effectiveness Measurement & Confounder Guardrails
When a user completes an active trial, `EffectivenessMeasurementService` evaluates post-intervention departure logs recorded after `started_at`:
- **Statistical Deltas:**
  - $\Delta \text{Average Delay} = \text{post\_avg\_delay} - \text{baseline\_avg\_delay}$
  - $\Delta \text{Lateness Rate} = \text{post\_lateness\_rate} - \text{baseline\_lateness\_rate}$
  - $\Delta \text{Prep Bias} = \text{post\_prep\_bias} - \text{baseline\_prep\_bias}$
- **Outcome Labels:**
  - `success`: Delay reduced by $\ge 3.0$ min or lateness rate reduced by $\ge 20\%$.
  - `partial_success`: Delay reduced by $1.0$ to $3.0$ min.
  - `neutral`: Delta between $-1.0$ and $+1.5$ min.
  - `regressed`: Delay increased by $> 1.5$ min.
  - `insufficient_post_observations`: Fewer than 2 departure checkouts recorded during trial window.
- **Confounding Variables Detected:**
  - Small post sample size ($< 5$ checkouts).
  - High variance in planned prep duration.
  - Inconsistent routine start times across trial period.
- **Scientific Causation Notice:**
  > *"Scientific Notice: Observed improvements in departure punctuality correlate with the intervention period but do not prove sole causation. Unmeasured factors such as external weather, traffic, schedule shifts, or temporary motivation may also contribute."*

---

### 11.7 User Control & REST API Endpoints
- `GET /api/recommendations/personalized?limit=6`: Returns eligible recommendations adaptively ranked by evidence and benefit.
- `GET /api/interventions?status=...`: Lists user interventions filtered by status (`suggested`, `accepted`, `completed`, `dismissed`, `snoozed`).
- `POST /api/interventions/accept`: Accepts a suggestion, creates active trial record, and captures baseline snapshot.
- `POST /api/interventions/dismiss`: Dismisses an intervention, initiating a 14-day cooling penalty.
- `POST /api/interventions/snooze`: Temporarily conceals the recommendation for specified days (default 7 days).
- `POST /api/interventions/complete`: Completes trial, records 1-5 star user rating and notes, and triggers empirical before/after effectiveness evaluation.
- `POST /api/interventions/feedback`: Submits user rating and qualitative reflection notes.
- `GET /api/interventions/{intervention_id}/effectiveness`: Inspects granular empirical impact metrics and confounding variables.

---

### 11.8 Frontend Experience (`frontend/src/App.tsx` & `frontend/src/index.css`)
- **Dashboard Spotlight:** Evidence-Driven Intervention Spotlight card displaying the active trial in progress or top eligible recommendations with direct navigation to the interventions pane.
- **Dedicated Habits & Interventions Tab:**
  - Multi-status filter tabs: `All`, `Eligible Suggestions`, `Active In Progress`, `Completed & Measured`.
  - Eligible suggestions cards with priority rank, category pill, diagnosed bottleneck evidence box, estimated delay savings chip, and collapsible step-by-step instructions.
  - Interactive Action Controls: **Start Habit Trial**, **Snooze 7d**, **Dismiss**.
  - Active Trial Tracker with baseline metric snapshot, active status badge, and **Complete & Review** trigger.
  - Completed Trials Showcase displaying measured before/after impact chips (Delay $\Delta$, Lateness $\Delta$), 5-star user rating, qualitative notes, confounder tags, and scientific causation disclaimer.
  - **Completion Feedback Modal:** Interactive dialog enabling 1-5 star ratings, helpfulness toggle, and qualitative user reflection notes.

---

## 12. Behavioral Drift Detection & Habit Change Monitoring (Prompt 2.8)

### 12.1 Objective & Architecture
Human morning routines are not static; people change schedules, wake up later during examination periods, adapt to timetable alterations, adopt remote work patterns, or improve punctuality following interventions. 

MicroDelay 2.0 introduces an advanced **Behavioral Drift & Habit Change Monitoring Engine** (`backend/ml/drift.py`) designed to identify meaningful routine shifts while strictly separating input data evolution from predictive model degradation.

```mermaid
flowchart TD
    Logs[Chronological Departure Logs] --> BaselineEngine[BehavioralBaselineService]
    BaselineEngine --> BaselineRecord[(Versioned Behavioral Baseline)]
    Logs --> TemporalAnalyzer[TemporalDriftAnalyzer]
    BaselineRecord --> TemporalAnalyzer
    TemporalAnalyzer --> KS[Kolmogorov-Smirnov Test]
    TemporalAnalyzer --> PSI[Population Stability Index]
    TemporalAnalyzer --> Welch[Welch t-test]
    TemporalAnalyzer --> PH[Page-Hinkley Sequential Test]
    
    KS --> ValenceClass[Valence & Type Classifier]
    PSI --> ValenceClass
    Welch --> ValenceClass
    PH --> ValenceClass
    
    ValenceClass -->|Positive Shift| Improve[Behavioral Improvement]
    ValenceClass -->|Negative Shift| Deteriorate[Behavioral Deterioration]
    ValenceClass -->|Input Shift| DataDrift[Data Drift]
    
    Logs --> ModelMonitor[ModelPerformanceMonitor]
    ModelMonitor -->|Rolling MAE & Bias| ConceptDrift[Concept Drift Alert]
    
    Improve --> Adapt[AdaptiveDriftCoordinator]
    Deteriorate --> Adapt
    ConceptDrift --> RetrainSignal[Candidate Retrain Recommended]
    Adapt --> UI[Frontend Trends & Habit Change Section]
```

---

### 12.2 Critical Distinction Between Drift Categories & Behavioral Valence
A core design requirement is that **the system does NOT automatically interpret every statistical distribution change as a negative event**:

1. **Behavioral Improvement (`positive` valence):**
   - Sustained reduction in departure delay, lateness rate, preparation underestimation bias, or morning distraction tags.
   - Evaluated as positive habit formation; highlighted with green visual indicators and positive feedback.
2. **Behavioral Deterioration (`negative` valence):**
   - Sustained increase in departure delay, chronic prep underestimation, or spiking item-search delays.
   - Tagged as emerging routine friction; triggers proactive habit recommendations.
3. **Data Drift (`neutral` valence):**
   - Distributional shifts in operational inputs (e.g., changes in planned departure times, weekday distributions, or preparation durations where punctuality is maintained).
   - Reflects human routine adaptation without implying failure.
4. **Concept Drift (`negative` valence, model health):**
   - Changes in the relationship between input features and actual departure delays, diagnosed via **predictive performance degradation** (Rolling MAE rising by $> 30\%$ over baseline MAE and signed prediction bias shifting $> \pm 3$ min).

---

### 12.3 Statistical Detection Methods
To avoid algorithmic opacity, MicroDelay combines 4 mathematically distinct, validated detectors:

| Method | Type | Primary Use Case & Threshold |
| :--- | :--- | :--- |
| **Population Stability Index (PSI)** | Bin-based Distribution | Compares overall quantile bucket stability across comparison windows. $\text{PSI} < 0.10$ (stable), $0.10 \le \text{PSI} < 0.25$ (moderate), $\text{PSI} \ge 0.25$ (significant drift). |
| **Kolmogorov-Smirnov Test (KS)** | Continuous eCDF Distance | Compares maximum vertical distance $D$ between empirical cumulative distribution functions. Triggers when $p < 0.05$ and $D \ge 0.25$. |
| **Welch's t-test** | Parametric Hypothesis Test | Evaluates difference in mean values with unequal variances. Detects directional delta and magnitude of shift. |
| **Page-Hinkley Sequential Test** | Streaming Cumulative Sum | Tracks cumulative deviation $U_t - m_t > \lambda$ over incoming checkout streams. Specifically protects against false alarms from single isolated outliers while detecting persistent step changes. |

---

### 12.4 Multi-Window Temporal Analysis & Outlier Robustness
The engine analyzes user checkouts across multiple chronological windows:
- **Recent 7-Day Window ($W_{7d}$):** Captures acute shocks (exam periods, timetable adjustments, vacation return).
- **Recent 30-Day Window ($W_{30d}$):** Captures sustained medium-term habit shifts (remote work adoption, seasonal patterns).
- **Sequential Streams:** Evaluates continuous cumulative change over time via Page-Hinkley.

**Guardrails:**
- **Chronological Safety:** Analysis strictly partitions logs available prior to the evaluation timestamp.
- **Minimum Sample Requirements:** Requires $\ge 4$ recent checkouts and $\ge 8$ baseline checkouts ($12$ total minimum). Cold-start users return `status: "insufficient_data"` without raising false alerts.
- **Single Outlier Suppression:** If 1 checkout exhibits an extreme deviation (e.g., 45-minute delay due to a blown car tire) but the median of the recent window matches baseline, the engine suppresses statistical drift alerts.

---

### 12.5 Model Performance Monitoring vs Input Drift
The system strictly decouples input feature evolution from ML model predictive health (`ModelPerformanceMonitor`):
- **Rolling MAE:** Evaluates absolute delay prediction error over the last $N=8$ checkouts.
- **Signed Prediction Bias:** Measures whether the model systematically underestimates ($+ \text{bias}$) or overestimates ($- \text{bias}$) departure delays.
- **Degradation Detection:** If Rolling MAE increases by $\ge 30\%$ above the baseline model MAE and sample size criteria are met, the system flags `performance_degraded = True` and creates a `concept_drift` record.
- **Responsible Adaptation Policy:** Rather than blindly retraining on every minor distribution change, the active model is strictly preserved until candidate models pass chronological cross-validation.

---

### 12.6 Database Persistence & Schema
1. **`behavioral_drift_events` (`BehavioralDriftEventRecord`):**
   - Fields: `drift_event_id`, `user_id`, `feature_name`, `drift_type`, `detection_method`, `baseline_value`, `current_value`, `drift_score`, `p_value`, `severity`, `valence`, `temporal_window`, `sample_count_baseline`, `sample_count_current`, `detected_at`, `status`, `humanized_summary`, `is_statistically_significant`.
2. **`user_behavioral_baselines` (`UserBehavioralBaselineRecord`):**
   - Stores versioned baseline metrics (`mean`, `std`, `median`, `q25`, `q75`, `count`) across all 7 monitored behavioral dimensions.
3. **`model_monitoring_logs` (`ModelMonitoringLogRecord`):**
   - Tracks sequential predictive error evaluations, rolling MAE, signed bias, degradation percentage, and retrain recommendations.

---

### 12.7 REST API Endpoints & Frontend Experience
- **Backend Endpoints (`backend/main.py`):**
  - `GET /api/intelligence/drift/analysis`: Multi-window temporal drift diagnosis.
  - `GET /api/intelligence/drift/events`: List diagnosed drift events with optional status filtering.
  - `POST /api/intelligence/drift/events/{id}/acknowledge`: Acknowledge or dismiss an active drift event.
  - `GET /api/intelligence/drift/model-monitoring`: Inspect rolling MAE, signed bias, and degradation status.
  - `POST /api/intelligence/drift/recalibrate`: Force baseline recomputation and profile refresh.
- **Frontend UI (`frontend/src/App.tsx` & `frontend/src/index.css`):**
  - **Behavioral Drift Panel:** Embedded in the Behavioral Profile view with status indicator (Positive Progress / Routine Friction / Stable), Recalibrate Baseline button, and sub-tab filters.
  - **Event Cards:** Color-coded by valence (Green for Improvement, Red for Friction, Purple for Shift) with statistical rigor badges (*"Statistically Verified ($p < 0.05$ / PSI $\ge 0.25$)"* vs *"Descriptive Trend ($n < 10$)"*), detection method tag, and baseline-to-current metric comparison.
  - **Predictive Model Health Tab:** Dedicated card comparing Input Stability with Predictive Accuracy, displaying Rolling MAE, Signed Bias, and Model Status.
  - **Dashboard Spotlight:** Real-time habit trend banner alerting users to sustained improvements or emerging departure friction.

---

## 13. Prediction Confidence & Reliability Engine (Prompt 2.9)

### 13.1 Core Principle: Probability is Not Confidence
A central tenet of MicroDelay's responsible AI framework is that **model probability and model confidence are fundamentally distinct concepts**:
- **Lateness Probability:** An estimate of event risk ($P(\text{delay} > 0 \mid x)$). A model might estimate a 90% probability of departing late.
- **Prediction Confidence:** An assessment of model epistemic uncertainty—whether that 90% estimate is backed by sufficient genuine historical observations, empirical conformal interval coverage, and probability calibration.
- **Rule Against Invention:** MicroDelay strictly forbids creating arbitrary confidence scores or presenting ungrounded confidence percentages. All reliability metrics are mathematically derived from verified holdouts.

---

### 13.2 Four-Domain Reliability Framework
Reliability is evaluated independently across four behavioral intelligence domains:

1. **Regression Uncertainty:**
   - **Split Conformal Prediction:** Computes finite-sample, distribution-free prediction intervals $[\max(0.0, \hat{y} - \hat{q}), \hat{y} + \hat{q}]$ at nominal $80\%$ coverage ($\alpha = 0.20$), where $\hat{q} = \text{Quantile}_{\lceil (n+1)(1-\alpha) \rceil / n}(R)$ on non-conformity residuals $R_i = |y_i - \hat{y}_i|$.
   - **Historical Error Distribution:** Tracks validation MAE, RMSE, median absolute error, signed prediction bias, and 10th/50th/90th percentile error bounds.
   - **Temporal Drift Caveat:** Under active behavioral drift, exchangeability is violated; the engine expands intervals by $25\%$ and attaches an explicit exchangeability warning.

2. **Classification Calibration:**
   - **Brier Score Loss:** Measures mean squared error between probabilities and binary lateness: $\text{BS} = \frac{1}{N} \sum (p_i - y_i)^2$.
   - **Expected Calibration Error (ECE):** Discretizes probabilities into $B=5$ bins and computes weighted calibration discrepancy:
     $$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
   - **Calibration Status:** Categorized as `"calibrated"` (Platt scaling with $\text{ECE} \le 0.18$), `"uncalibrated"`, or `"insufficient_samples"` ($n < 8$).
   - **Qualitative Risk Guarantee:** When uncalibrated, probabilities are explicitly flagged as qualitative estimates rather than verified probabilities.

3. **Behavioral Profile Reliability:**
   - Tracks observation count $n$, attribute completeness percentage, recency days (flagging stale data $> 45$ days), and coefficient of variation ($\text{CV} = \sigma / \mu$) across prep duration and delays.

4. **Recommendation Evidence Tiers:**
   - Evaluates interventions by historical trials:
     - `Heuristic Rule`: $0$ prior user trials (triggered by observed behavioral gap).
     - `Preliminary`: $1-2$ trials.
     - `Empirically Verified`: $\ge 3$ trials with $\ge 60\%$ observed delay reduction.

---

### 13.3 Documented Rule-Based Reliability States
Confidence statuses are derived strictly from documented operational rules:

| Reliability State | Criteria & Trigger Conditions |
|---|---|
| `degraded_model` | Model monitoring detected predictive degradation (rolling MAE $\ge +30\%$ over baseline) or uncalibrated drift. |
| `insufficient_history` | Fewer than $8$ genuine checkout observations and no active global model prior. |
| `low_reliability` | Minimal observations ($8 \le n < 15$), OR stale observations ($> 45$ days), OR validation MAE $> 7.0$ min, OR uncalibrated classifier. |
| `moderate_reliability` | Adequate observations ($15 \le n < 30$), validation MAE $\le 5.5$ min, recency $\le 45$ days, reasonable interval width. |
| `high_reliability` | Robust observations ($n \ge 30$), validation MAE $\le 3.5$ min, calibrated classifier ($\text{ECE} \le 0.18$), fresh data ($\le 21$ days), and no active drift. |

---

### 13.4 Prediction Abstention Engine
MicroDelay responsibly **abstains from making predictions** rather than forcing an ungrounded guess when operating outside validated conditions:
- **Out-of-Range Contexts:** If estimated preparation or travel duration falls outside physical validation bounds ($[0, 300]$ minutes).
- **Extreme Cold Start:** If the user has $< 3$ observations and no global model is available.
- **Catastrophic Model Degradation:** When active models degrade without safe baseline fallbacks.
- **Abstention Contract:** Returns `is_abstained = True`, `predicted_delay_minutes = None`, `abstention_reason = "Prediction unavailable — insufficient reliable historical observations."`, and guides the user to log checkouts.

---

### 13.5 Database Schema & API Endpoints
1. **`model_reliability_records` (`ModelReliabilityRecord`):**
   - Fields: `user_id`, `model_version`, `evaluated_at`, `conformal_quantile_80`, `conformal_quantile_90`, `empirical_coverage_80`, `validation_mae`, `validation_rmse`, `brier_score`, `expected_calibration_error`, `calibration_status`, `data_sufficiency`, `reliability_status`, `sample_count_evaluated`, `is_abstained`, `details`.
2. **REST Endpoints (`backend/main.py`):**
   - `GET /api/intelligence/reliability/status`: Unified reliability assessment, prediction intervals, data sufficiency, profile reliability, and limitations.
   - `GET /api/intelligence/reliability/calibration`: Classification calibration diagram bins, Brier score, and ECE.
   - `GET/POST /api/intelligence/predict` & `/risk`: Extended with `prediction_interval_lower`, `prediction_interval_upper`, `coverage_level`, `reliability_status`, `calibration_status`, `data_sufficiency`, `limitations`, `is_abstained`, `abstention_reason`.



