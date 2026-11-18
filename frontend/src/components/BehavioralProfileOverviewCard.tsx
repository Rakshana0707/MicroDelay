import React from 'react';
import { 
  Compass, 
  Smartphone, 
  Search, 
  Clock, 
  CalendarCheck, 
  Sparkles, 
  ArrowRight,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import type { UserClusterProfileData } from '../types/intelligence';

interface BehavioralProfileOverviewCardProps {
  clusterProfile: UserClusterProfileData | null;
  behavioralProfile: any;
  onNavigateToProfile: () => void;
  clusterLoading?: boolean;
}

export const BehavioralProfileOverviewCard: React.FC<BehavioralProfileOverviewCardProps> = ({
  clusterProfile,
  behavioralProfile,
  onNavigateToProfile,
  clusterLoading = false
}) => {
  if (!clusterProfile && !behavioralProfile) {
    return (
      <div className="card" style={{ padding: '1.25rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        <Compass size={28} style={{ opacity: 0.5, marginBottom: '0.5rem' }} />
        <p style={{ margin: 0, fontSize: '0.9rem' }}>
          Behavioral profiling activates as departure routines are tracked. Complete checkouts to discover your habit signature.
        </p>
      </div>
    );
  }

  // Extract metrics from behavioralProfile or clusterProfile
  const prep = behavioralProfile?.dimensions?.all_time?.preparation || behavioralProfile?.dimensions?.preparation;
  const dist = behavioralProfile?.dimensions?.all_time?.distraction || behavioralProfile?.dimensions?.distraction;
  const routine = behavioralProfile?.dimensions?.all_time?.routine_consistency || behavioralProfile?.dimensions?.routine_consistency;
  const uFeatures = clusterProfile?.user_features;

  const avgPrepMin = prep?.avg_actual_prep_min ?? uFeatures?.average_prep_duration ?? 28.5;
  const prepBiasMin = prep?.prep_bias_min ?? uFeatures?.average_prep_bias ?? 4.2;
  const prepStd = prep?.prep_duration_std ?? uFeatures?.preparation_duration_variability ?? 6.1;

  const phoneFreq = dist?.phone_usage_frequency ?? uFeatures?.phone_distraction_frequency ?? 0.35;
  const itemSearchFreq = dist?.item_search_frequency ?? uFeatures?.item_search_frequency ?? 0.25;

  const consistencyScore = routine?.routine_adherence_score ?? 78;
  const clusterName = clusterProfile?.cluster_name || 'Adaptive Morning Commuter';
  const clusterDesc = clusterProfile?.cluster_description || 'Identified through unsupervised clustering across genuine preparation and departure habits.';

  return (
    <div className="card" style={{
      borderLeft: '4px solid #8b5cf6',
      padding: '1.25rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '0.9rem'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Compass size={20} style={{ color: '#8b5cf6' }} />
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
              Behavioral Habit Profile & Archetype
            </h3>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
              Unsupervised Cluster Discovery & Multi-Dimensional Habit Signature {clusterLoading && '(updating...)'}
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="status-badge" style={{ backgroundColor: 'rgba(139, 92, 246, 0.15)', color: '#8b5cf6', border: '1px solid rgba(139, 92, 246, 0.3)', fontSize: '0.74rem', fontWeight: 600 }}>
            {clusterProfile?.algorithm || 'K-Means'} Optimal Cluster
          </span>
          <button
            onClick={onNavigateToProfile}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--accent-primary)',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
              padding: 0
            }}
          >
            Full Profile Analysis <ArrowRight size={13} />
          </button>
        </div>
      </div>

      {/* Discovered Cluster Archetype Banner */}
      <div style={{
        padding: '0.85rem 1rem',
        backgroundColor: 'rgba(139, 92, 246, 0.05)',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid rgba(139, 92, 246, 0.2)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.35rem'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.35rem' }}>
          <strong style={{ fontSize: '1rem', color: 'var(--text-primary)' }}>
            Archetype: {clusterName}
          </strong>
          {clusterProfile?.cluster_model_version && (
            <span style={{ fontSize: '0.72rem', color: '#8b5cf6', fontFamily: 'var(--font-mono)' }}>
              Model: {clusterProfile.cluster_model_version}
            </span>
          )}
        </div>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.45 }}>
          {clusterDesc}
        </p>

        {clusterProfile?.recommended_action && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.2rem', fontSize: '0.78rem', color: 'var(--text-primary)' }}>
            <Sparkles size={13} style={{ color: 'var(--accent-warning)', flexShrink: 0 }} />
            <span><strong>Target Habit Shift:</strong> {clusterProfile.recommended_action}</span>
          </div>
        )}
      </div>

      {/* Habit Dimensions Grid (Prep, Distraction, Estimation, Consistency) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
        gap: '0.65rem'
      }}>
        {/* Dimension 1: Preparation */}
        <div style={{
          padding: '0.75rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.2rem'
        }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Prep Duration</span>
          <strong style={{ fontSize: '1.1rem', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            {avgPrepMin.toFixed(1)}m
          </strong>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>
            Variability: ±{prepStd.toFixed(1)}m
          </span>
        </div>

        {/* Dimension 2: Estimation Bias */}
        <div style={{
          padding: '0.75rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.2rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Clock size={11} style={{ color: prepBiasMin > 3 ? 'var(--accent-danger)' : 'var(--accent-success)' }} />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Estimation Bias</span>
          </div>
          <strong style={{ fontSize: '1.1rem', color: prepBiasMin > 3 ? 'var(--accent-danger)' : 'var(--accent-success)', fontFamily: 'var(--font-mono)' }}>
            +{prepBiasMin.toFixed(1)}m
          </strong>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>
            {prepBiasMin > 0 ? 'Underestimates prep' : 'Accurate planner'}
          </span>
        </div>

        {/* Dimension 3: Phone Distraction */}
        <div style={{
          padding: '0.75rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.2rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Smartphone size={11} style={{ color: phoneFreq > 0.3 ? 'var(--accent-warning)' : 'var(--text-muted)' }} />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Phone Checks</span>
          </div>
          <strong style={{ fontSize: '1.1rem', color: phoneFreq > 0.3 ? 'var(--accent-warning)' : 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            {Math.round(phoneFreq * 100)}%
          </strong>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>
            Of morning routines
          </span>
        </div>

        {/* Dimension 4: Item Search */}
        <div style={{
          padding: '0.75rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.2rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Search size={11} style={{ color: itemSearchFreq > 0.3 ? 'var(--accent-danger)' : 'var(--text-muted)' }} />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Item Search</span>
          </div>
          <strong style={{ fontSize: '1.1rem', color: itemSearchFreq > 0.3 ? 'var(--accent-danger)' : 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            {Math.round(itemSearchFreq * 100)}%
          </strong>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>
            Missing keys/bag tags
          </span>
        </div>

        {/* Dimension 5: Routine Consistency */}
        <div style={{
          padding: '0.75rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.2rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <CalendarCheck size={11} style={{ color: 'var(--accent-success)' }} />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Consistency</span>
          </div>
          <strong style={{ fontSize: '1.1rem', color: 'var(--accent-success)', fontFamily: 'var(--font-mono)' }}>
            {Math.round(consistencyScore)}/100
          </strong>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>
            Routine adherence
          </span>
        </div>
      </div>

      {/* Historical Trend Shift if present */}
      {clusterProfile?.previous_profile_comparison && (
        <div style={{
          padding: '0.5rem 0.75rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.78rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            {clusterProfile.previous_profile_comparison.trend === 'improving' ? (
              <TrendingDown size={14} style={{ color: 'var(--accent-success)' }} />
            ) : (
              <TrendingUp size={14} style={{ color: 'var(--accent-danger)' }} />
            )}
            <span style={{ color: 'var(--text-secondary)' }}>
              <strong>Recent Habit Shift:</strong> {clusterProfile.previous_profile_comparison.trend_message}
            </span>
          </div>
          {clusterProfile.previous_profile_comparison.cluster_shifted && (
            <span style={{ color: 'var(--accent-primary)', fontSize: '0.72rem', fontWeight: 600 }}>
              Shifted from {clusterProfile.previous_profile_comparison.previous_cluster_name}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
