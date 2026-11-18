// Phase 2 Adaptive Behavioral Intelligence TypeScript Interfaces

export interface RiskContributingFactor {
  factor: string;
  impact_minutes: number;
  direction: string;
  description: string;
}

export interface RecommendedIntervention {
  action: string;
  description: string;
  expected_delay_reduction_min: number;
  category: string;
  priority: string;
}

export interface ContributingFactorData {
  factor: string;
  impact_value: number;
  direction: 'increases_delay' | 'decreases_delay';
  description: string;
  evidence_type: 'model_contribution' | 'statistical_association' | 'directly_observed_behavior' | 'estimated_behavioral_influence';
  raw_weight?: number | null;
  feature_name?: string | null;
}

export interface HistoricalComparisonData {
  dimension: string;
  metric: string;
  current_value: any;
  historical_baseline: any;
  statement: string;
  evidence_type: string;
}

export interface PredictionExplanationData {
  prediction_id: string;
  model_version: string;
  feature_schema_version: string;
  explanation_method: 'tree_shap' | 'tree_feature_contribution' | 'linear_coefficients' | 'behavioral_rule_reasoning';
  predicted_delay_minutes: number;
  late_probability?: number | null;
  top_positive_contributors: ContributingFactorData[];
  top_mitigating_factors: ContributingFactorData[];
  historical_comparisons: HistoricalComparisonData[];
  limitations: string[];
  causation_disclaimer: string;
  feature_audit_values?: Record<string, any>;
  explanation_timestamp: string;
}

export interface RiskPredictionData {
  late_probability: number | null;
  risk_category: 'low' | 'moderate' | 'high' | 'extreme' | null;
  predicted_delay_minutes: number | null;
  baseline_predicted_delay_minutes: number;
  prediction_interval_lower?: number | null;
  prediction_interval_upper?: number | null;
  coverage_level?: number | null;
  reliability_status?: 'insufficient_history' | 'low_reliability' | 'moderate_reliability' | 'high_reliability' | 'degraded_model';
  calibration_status?: 'calibrated' | 'uncalibrated' | 'insufficient_samples';
  data_sufficiency?: 'insufficient' | 'minimal' | 'moderate' | 'robust';
  limitations?: string[];
  is_abstained?: boolean;
  abstention_reason?: string | null;
  reliability_assessment?: any;
  model_version: string;
  model_type: string;
  model_availability: string;
  training_data_provenance: string;
  prediction_timestamp?: string;
  confidence_status: {
    score: number;
    level: string;
    is_reliable: boolean;
    reasons: string[];
  };
  contributing_factors: RiskContributingFactor[];
  recommended_interventions: RecommendedIntervention[];
  personalization_info?: PersonalizationInfoData;
  explanation?: PredictionExplanationData;
}

export interface CalibrationBinData {
  bin_index: number;
  bin_lower: number;
  bin_upper: number;
  bin_center: number;
  mean_confidence: number;
  empirical_accuracy: number;
  sample_count: number;
}

export interface CalibrationReportData {
  has_classifier: boolean;
  model_name: string;
  brier_score: number;
  expected_calibration_error: number;
  max_calibration_error: number;
  calibration_status: 'calibrated' | 'uncalibrated' | 'insufficient_samples';
  is_calibrated: boolean;
  sample_count: number;
  positive_prevalence_pct: number;
  bins: CalibrationBinData[];
  risk_groups: Record<string, any>;
  notes: string[];
}

export interface UserReliabilityStatusData {
  reliability_status: 'insufficient_history' | 'low_reliability' | 'moderate_reliability' | 'high_reliability' | 'degraded_model';
  data_sufficiency: 'insufficient' | 'minimal' | 'moderate' | 'robust';
  calibration_status: 'calibrated' | 'uncalibrated' | 'insufficient_samples';
  is_abstained: boolean;
  abstention_reason: string | null;
  prediction_interval?: {
    lower_bound: number;
    upper_bound: number;
    coverage_level: number;
    interval_width: number;
    conformal_quantile: number;
    method: string;
    is_calibrated: boolean;
    is_drift_impacted: boolean;
    warning?: string;
  };
  error_distribution?: {
    mae: number;
    rmse: number;
    median_abs_error: number;
    signed_bias: number;
    residual_quantiles: { p10: number; p50: number; p90: number };
    sample_count: number;
  };
  classification_calibration?: CalibrationReportData;
  data_sufficiency_report: {
    level: string;
    genuine_count: number;
    synthetic_count: number;
    total_count: number;
    recency_days: number | null;
    completeness_score: number;
    notes: string[];
  };
  profile_reliability?: {
    observation_count: number;
    completeness_pct: number;
    recency_days: number | null;
    recency_status: string;
    metric_stability: string;
    is_reliable: boolean;
    notes: string[];
  };
  recommendation_reliabilities: Array<{
    action: string;
    behavioral_problem: string;
    evidence_level: string;
    trial_count: number;
    success_count: number;
    success_rate_pct: number;
    is_verified: boolean;
  }>;
  limitations: string[];
  model_version?: string;
  has_active_personal_model?: boolean;
  has_active_global_model?: boolean;
  is_model_degraded?: boolean;
  is_drift_detected?: boolean;
}

export interface PersonalizationInfoData {
  learning_mode: 'new_user' | 'collecting_history' | 'global_prediction' | 'personalizing' | 'personalized' | 'insufficient_data';
  alpha: number;
  genuine_samples: number;
  target_samples: number;
  is_global_used: boolean;
  is_personal_used: boolean;
  explanation: string;
  global_prediction?: number | null;
  personal_prediction?: number | null;
  hybrid_prediction?: number | null;
  global_model_version?: string | null;
  personal_model_version?: string | null;
}

export interface UserConsentData {
  user_id: number;
  username: string;
  consent_to_global_training: boolean;
}

export interface GlobalModelStatusData {
  is_available: boolean;
  global_model_version: string | null;
  cohort_users_count: number;
  total_consented_samples: number;
  validation_mae: number | null;
  test_mae: number | null;
  best_algorithm: string | null;
  trained_at: string | null;
  status_reason: string;
}

export interface ClusterDominantCharacteristic {
  feature: string;
  cluster_value: number;
  population_mean: number;
  relative_difference: string;
  z_score: number;
}

export interface PreviousProfileComparison {
  previous_cluster_name: string;
  previous_generated_at: string;
  delay_delta_minutes: number;
  trend: 'improving' | 'regressing' | 'stable';
  trend_message: string;
  cluster_shifted: boolean;
}

export interface UserClusterProfileData {
  user_id: number;
  cluster_id: number;
  cluster_name: string;
  cluster_description: string;
  dominant_characteristics: ClusterDominantCharacteristic[];
  recommended_action: string;
  user_features: Record<string, number>;
  cluster_centroid: Record<string, number>;
  population_mean: Record<string, number>;
  data_sufficiency_status: 'sufficient' | 'gathering_history' | 'insufficient_data';
  sample_count: number;
  cluster_model_version: string;
  algorithm: string;
  silhouette_score: number;
  davies_bouldin_index: number;
  profile_generation_timestamp: string;
  previous_profile_comparison?: PreviousProfileComparison | null;
}

export interface BehavioralProfileData {
  profile_id?: string;
  user_id?: number;
  temporal_window?: string;
  observation_count?: number;
  data_provenance?: string;
  data_sufficiency_status?: string;
  confidence_or_sufficiency_status?: string;
  profile_generation_timestamp?: string;
  generated_at?: string;
  narrative_summary?: string;
  recommendations?: any[];
  dimensions?: any;
}

export interface SupportingEvidenceData {
  problem_frequency_pct?: number;
  problem_frequency?: number;
  observed_overhead_min?: number;
  sample_count?: number;
  affected_sessions?: number;
  metric_name?: string;
  metric_value?: string;
  supporting_statement?: string;
  statement?: string;
}

export interface MeasuredOutcomeData {
  status?: string;
  outcome_label?: string;
  is_improved?: boolean;
  delay_delta_min?: number;
  lateness_rate_delta?: number;
  prep_bias_delta_min?: number;
  baseline_avg_delay?: number;
  post_avg_delay?: number;
  baseline_lateness_rate?: number;
  post_lateness_rate?: number;
  baseline_sample_count?: number;
  post_sample_count?: number;
  target_metric?: string;
  target_improvement?: boolean;
  message?: string;
  measured_at?: string;
}

export interface InterventionData {
  intervention_id: string;
  recommendation_id: string;
  category: 'preparation' | 'distraction' | 'timing' | 'routine_consistency';
  action: string;
  description: string;
  implementation_steps: string[];
  implementation_effort?: 'low' | 'medium' | 'high';
  behavioral_problem: string;
  supporting_evidence: SupportingEvidenceData;
  expected_delay_reduction_min: number;
  priority?: 'high' | 'medium' | 'low';
  priority_score?: number;
  status: 'suggested' | 'accepted' | 'started' | 'completed' | 'dismissed' | 'snoozed' | 'expired';
  expected_outcome?: Record<string, any>;
  measured_outcome?: MeasuredOutcomeData | null;
  baseline_metrics?: Record<string, any> | null;
  post_metrics?: Record<string, any> | null;
  confounding_factors?: string[];
  causation_disclaimer: string;
  generated_at: string;
  accepted_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  dismissed_at?: string | null;
  snoozed_until?: string | null;
  user_rating?: number | null;
  user_feedback_notes?: string | null;
  user_helpful?: boolean | null;
  completion_status?: string | null;
}

export interface DriftEventData {
  drift_event_id: string;
  feature_name: string;
  drift_type: 'data_drift' | 'concept_drift' | 'behavioral_improvement' | 'behavioral_deterioration';
  detection_method: string;
  baseline_value: number;
  current_value: number;
  drift_score: number;
  p_value?: number | null;
  severity: 'informational' | 'moderate' | 'significant';
  valence: 'positive' | 'negative' | 'neutral';
  temporal_window: string;
  sample_count_baseline: number;
  sample_count_current: number;
  detected_at: string;
  status: 'active' | 'acknowledged' | 'resolved' | 'dismissed';
  humanized_summary: string;
  is_statistically_significant: boolean;
}

export interface DriftAnalysisData {
  status: string;
  total_observations: number;
  events_detected: DriftEventData[];
  improvements_count: number;
  deteriorations_count: number;
  data_drifts_count: number;
  summary: string;
  evaluated_at: string;
  message?: string;
}

export interface ModelMonitoringData {
  status: string;
  model_version?: string;
  rolling_mae?: number | null;
  rolling_bias?: number | null;
  baseline_mae?: number;
  performance_degraded: boolean;
  degradation_pct?: number;
  error_distribution_shift?: boolean;
  retrain_recommended?: boolean;
  evaluated_at?: string;
  message?: string;
}

export interface DataSummaryData {
  total_samples: number;
  real_samples: number;
  synthetic_samples: number;
  has_real_data: boolean;
  is_sufficient_real: boolean;
}

export interface GlobalFeatureImportanceData {
  method: string;
  features: Array<{
    feature: string;
    importance: number;
    std?: number;
    normalized_weight?: number;
    human_name?: string;
  }>;
  top_feature?: string;
  evaluated_at: string;
  message?: string;
}
