# MicroDelay 2.0 — Feature Engineering Documentation

**Document Version:** 1.0.0  
**Module:** `backend/ml/features.py`  
**Purpose:** Comprehensive specification of calendar, historical behavioral, preparation, travel, and behavioral trend features for departure delay prediction.

---

## 1. Feature Philosophy & Information Leakage Prevention

MicroDelay's machine learning objective is to predict **departure delay** ($\text{total\_delay\_min} = \text{actual\_departure} - \text{planned\_departure}$) before the departure takes place.

### Core Data Leakage Principles:
1. **Strict Temporal Ordering:** Behavioral and rolling historical features are computed solely from observations that occurred **strictly prior** to the target departure time.
2. **Explicit Shift Operations:** In historical training sets, all actual durations (`actual_prep_duration_min`, `actual_travel_duration_min`, `total_delay_min`) and checkout tags are shifted by 1 position ($\text{shift}(1)$) before cumulative or rolling operations are performed. Observation $i$ never observes its own outcomes.
3. **Target & Post-Departure Value Exclusion:** The current departure's actual departure time, actual preparation duration, actual travel duration, and checkout reason tags are never used as model inputs.
4. **Identical Training and Inference Logic:** `HistoricalFeatureExtractor.enrich_inference_record()` extracts identical feature definitions during real-time prediction by summarizing past database logs up to the prediction timestamp.
5. **Cold-Start Fallbacks:** When a user has zero prior history, population neutral defaults (calibrated from baseline distributions) are used safely without crashing.

---

## 2. Feature Schema Catalog

The system provides 27 total features partitioned across five distinct categories:

### A. Calendar Features

| Feature Name | Data Source | Calculation | Prediction-Time Availability | Leakage Risk & Safeguards |
|---|---|---|---|---|
| `day_of_week` | Event template / schedule | Direct day name string | Available at schedule creation | **None.** Fixed calendar attribute. |
| `is_weekend` | `day_of_week` | `1` if day in `['Saturday', 'Sunday']` else `0` | Available before departure | **None.** Deterministic calendar transformation. |
| `is_monday` | `day_of_week` | `1` if day == `'Monday'` else `0` | Available before departure | **None.** Captures start-of-week delays. |
| `month` | Date | `date.month` (1 – 12) | Available before departure | **None.** Deterministic calendar extraction. |
| `week_of_year` | Date | `date.isocalendar().week` (1 – 52) | Available before departure | **None.** Captures semester/work cycle seasonality. |
| `event_time_of_day` | `planned_departure` | $\text{hour} + \frac{\text{minute}}{60.0}$ (decimal hour, e.g. 8.5 for 08:30) | Available from schedule template | **None.** Uses strictly planned time, not actual departure. |
| `is_first_event_of_day` | Daily schedule | `1` if scheduled departure is the earliest planned event on that date, else `0` | Available from daily schedule | **None.** Derived strictly from planned departure order. |

---

### B. Preparation Features

| Feature Name | Data Source | Calculation | Prediction-Time Availability | Leakage Risk & Safeguards |
|---|---|---|---|---|
| `importance` | Event template | Categorical: `'low'`, `'medium'`, `'high'` | Available in event template | **None.** Static routine priority. |
| `estimated_prep_duration` | Event template | User's planned prep time in minutes | Available before routine starts | **None.** Purely prospective user estimate. |
| `bag_prepared_indicator` | Routine start toggle | `1` if materials packed previous night, else `0` | Checked prior to prep start | **None.** Pre-prep state input. |
| `historical_prep_duration_deviation` | `estimated_prep` & user history | $\text{estimated\_prep} - \text{user\_average\_prep\_duration}$ | Available before routine starts | **Safe.** Historical average uses strictly shifted past events. |

---

### C. Travel Features

| Feature Name | Data Source | Calculation | Prediction-Time Availability | Leakage Risk & Safeguards |
|---|---|---|---|---|
| `estimated_travel_duration` | Event template | Planned transit duration in minutes | Available before departure | **None.** Planned static value. |
| `departure_time_bucket` | `planned_departure` | Binned: `'early_morning'`, `'morning_peak'`, `'midday'`, `'afternoon'`, `'evening'` | Available before departure | **None.** Based strictly on planned schedule. |
| `historical_travel_error` | Historical travel logs | $\text{mean}(\text{actual\_travel} - \text{estimated\_travel})$ over past events | Available before departure | **Safe.** Excludes current transit duration; shifted by 1. |

---

### D. Historical Behavioral Features (Shifted Cumulative)

All behavioral features below are computed over strictly past completed departures:

| Feature Name | Data Source | Calculation | Prediction-Time Availability | Leakage Risk & Safeguards |
|---|---|---|---|---|
| `user_average_prep_duration` | Historical `actual_prep` | $\frac{1}{k}\sum_{j=1}^{k} \text{actual\_prep}_j$ for $j < i$ | Computed from previous logs | **Guaranteed safe.** Past events only (via `shift(1)`). |
| `user_average_prep_bias` | Historical prep delta | $\frac{1}{k}\sum_{j=1}^{k} (\text{actual\_prep}_j - \text{est\_prep}_j)$ for $j < i$ | Computed from previous logs | **Guaranteed safe.** Measures user's personal planning fallacy. |
| `user_historical_delay_rate` | Historical `total_delay` | $\frac{\text{count}(\text{delay}_j > 0)}{k}$ for $j < i$ | Computed from previous logs | **Guaranteed safe.** Lateness probability. |
| `event_specific_average_delay` | Historical delays grouped by event | $\text{mean}(\text{delay}_j)$ for event $E$ where $j < i$ | Computed from previous logs | **Guaranteed safe.** Event-specific delay tendency. |
| `day_specific_average_delay` | Historical delays grouped by day | $\text{mean}(\text{delay}_j)$ for day $D$ where $j < i$ | Computed from previous logs | **Guaranteed safe.** Day-specific delay tendency. |
| `historical_phone_distraction_frequency` | Historical `tag_phone_usage` | $\frac{\text{count}(\text{phone}_j = \text{True})}{k}$ for $j < i$ | Computed from previous logs | **Guaranteed safe.** Historical phone distraction rate. |
| `historical_item_search_frequency` | Historical `tag_finding_items` | $\frac{\text{count}(\text{search}_j = \text{True})}{k}$ for $j < i$ | Computed from previous logs | **Guaranteed safe.** Historical item searching rate. |
| `historical_last_task_frequency` | Historical `tag_one_last_task` | $\frac{\text{count}(\text{last\_task}_j = \text{True})}{k}$ for $j < i$ | Computed from previous logs | **Guaranteed safe.** Last-minute chore frequency. |

---

### E. Behavioral Trend Features (Rolling Window $K=5$)

| Feature Name | Data Source | Calculation | Prediction-Time Availability | Leakage Risk & Safeguards |
|---|---|---|---|---|
| `recent_average_delay` | Recent 5 departures | Rolling 5-event mean of `total_delay_min` strictly prior to $i$ | Computed from recent 5 logs | **Guaranteed safe.** Rolling window uses `shift(1)`. |
| `recent_lateness_rate` | Recent 5 departures | Rolling 5-event fraction of late departures ($> 0$ min) | Computed from recent 5 logs | **Guaranteed safe.** Captures acute sleep/routine changes. |
| `recent_prep_bias` | Recent 5 departures | Rolling 5-event mean of prep underestimation | Computed from recent 5 logs | **Guaranteed safe.** Captures recent prep rush patterns. |
| `change_in_recent_behavior` | Trend comparison | $\text{recent\_average\_delay} - \text{user\_average\_delay}$ | Computed from past logs | **Guaranteed safe.** Behavioral momentum indicator. |

---

## 3. Empirical Feature Validation Methodology

To ensure that engineered features are retained only when supported by genuine evaluation:
1. Both original and engineered feature sets are evaluated using identical **chronological holdout splits** (80% training / 20% test).
2. The preprocessor is fitted strictly on the training partition.
3. The evaluation records MAE and RMSE improvements.
4. If the engineered features fail to reduce MAE compared to the original baseline, the pipeline recommends retaining the simpler feature set.
