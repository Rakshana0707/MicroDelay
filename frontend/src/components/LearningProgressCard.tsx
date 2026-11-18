import React from 'react';
import { 
  GitBranch, 
  RefreshCw, 
  Layers
} from 'lucide-react';
import type { PersonalizationInfoData, UserReliabilityStatusData } from '../types/intelligence';

interface LearningProgressCardProps {
  personalizationInfo: PersonalizationInfoData | null;
  reliabilityStatus: UserReliabilityStatusData | null;
  genuineSamples: number;
  syntheticSamples: number;
  lastTrainedAt?: string | null;
  lastProfileAt?: string | null;
  onRefresh?: () => void;
  refreshing?: boolean;
}

export const LearningProgressCard: React.FC<LearningProgressCardProps> = ({
  personalizationInfo,
  reliabilityStatus,
  genuineSamples,
  syntheticSamples,
  lastTrainedAt,
  lastProfileAt,
  onRefresh,
  refreshing = false
}) => {
  const targetSamples = 8;
  const progressPct = Math.min(100, Math.round((genuineSamples / targetSamples) * 100));

  // Determine current operational mode
  let modeLabel = 'Demonstration Mode';
  let modeBadgeClass = 'synthetic';
  let modeDescription = 'Using synthetic population baseline while awaiting your first genuine morning departure routines.';

  if (genuineSamples >= targetSamples) {
    modeLabel = 'Personalized Model (Level 3)';
    modeBadgeClass = 'real';
    modeDescription = `Trained directly on your ${genuineSamples} genuine departures. Predictions are fully tuned to your habit patterns.`;
  } else if (genuineSamples >= 5) {
    modeLabel = 'Hybrid Prior + Personal Residual (Level 2)';
    modeBadgeClass = 'hybrid';
    modeDescription = `Combining global consented population prior with your ${genuineSamples} genuine checkouts via residual correction (${progressPct}% personal weight).`;
  } else if (genuineSamples > 0) {
    modeLabel = 'Global Cold-Start (Level 1)';
    modeBadgeClass = 'warning';
    modeDescription = `Operating on privacy-consented global baseline while gathering your individual historical observations (${genuineSamples}/${targetSamples}).`;
  }

  const activeVersion = personalizationInfo?.personal_model_version || 
                        personalizationInfo?.global_model_version || 
                        reliabilityStatus?.model_version || 
                        'v2.4-rf-ensemble';

  return (
    <div className="card" style={{
      borderLeft: `4px solid ${genuineSamples >= targetSamples ? 'var(--accent-success)' : 'var(--accent-primary)'}`,
      padding: '1.25rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '0.85rem'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <GitBranch size={20} style={{ color: genuineSamples >= targetSamples ? 'var(--accent-success)' : 'var(--accent-primary)' }} />
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
              Continuous Learning & Model Governance
            </h3>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
              Real-time Global-to-Personalized Learning Progression
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className={`status-badge ${modeBadgeClass}`} style={{ fontSize: '0.74rem', fontWeight: 600 }}>
            {modeLabel}
          </span>
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={refreshing}
              title="Refresh learning & calibration status"
              style={{
                background: 'none',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.3rem 0.5rem',
                color: 'var(--text-secondary)',
                cursor: refreshing ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
                fontSize: '0.75rem'
              }}
            >
              <RefreshCw size={12} className={refreshing ? 'spinning' : ''} />
              {refreshing ? 'Updating...' : 'Refresh'}
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar toward Personal Threshold */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
          <span style={{ color: 'var(--text-secondary)' }}>
            <strong>Genuine Observations Progress:</strong> {genuineSamples} of {targetSamples} required for personal ML activation
          </span>
          <span style={{ fontWeight: 700, color: genuineSamples >= targetSamples ? 'var(--accent-success)' : 'var(--accent-primary)', fontFamily: 'var(--font-mono)' }}>
            {progressPct}%
          </span>
        </div>
        <div style={{
          height: '8px',
          backgroundColor: 'rgba(148, 163, 184, 0.2)',
          borderRadius: '4px',
          overflow: 'hidden'
        }}>
          <div style={{
            height: '100%',
            width: `${progressPct}%`,
            backgroundColor: genuineSamples >= targetSamples ? 'var(--accent-success)' : 'var(--accent-primary)',
            transition: 'width 0.4s ease'
          }} />
        </div>
      </div>

      {/* Mode Narrative */}
      <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.45 }}>
        {modeDescription}
      </p>

      {/* Learning Status Meta Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: '0.65rem',
        fontSize: '0.8rem',
        backgroundColor: 'var(--bg-primary)',
        padding: '0.75rem',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border-color)'
      }}>
        <div>
          <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.7rem' }}>Active Model Version</span>
          <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{activeVersion}</strong>
          {lastTrainedAt && (
            <span style={{ display: 'block', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              Trained: {new Date(lastTrainedAt).toLocaleDateString()}
            </span>
          )}
        </div>

        <div>
          <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.7rem' }}>Dataset Provenance</span>
          <strong style={{ color: 'var(--text-primary)' }}>
            {genuineSamples} Real / {syntheticSamples} Synthetic
          </strong>
        </div>

        <div>
          <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.7rem' }}>Model Reliability</span>
          <span className={`reliability-badge ${reliabilityStatus?.reliability_status || 'moderate_reliability'}`} style={{ fontSize: '0.68rem' }}>
            {reliabilityStatus?.reliability_status?.replace(/_/g, ' ') || 'Moderate'}
          </span>
        </div>

        <div>
          <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.7rem' }}>Last Profile Update</span>
          <strong style={{ color: 'var(--text-secondary)' }}>
            {lastProfileAt ? new Date(lastProfileAt).toLocaleDateString() : 'Active session'}
          </strong>
        </div>
      </div>

      {/* Explicit Learning What-Is-Learning Narrative */}
      <div style={{
        fontSize: '0.78rem',
        color: 'var(--text-secondary)',
        backgroundColor: 'rgba(99, 102, 241, 0.03)',
        padding: '0.5rem 0.75rem',
        borderRadius: 'var(--radius-sm)',
        border: '1px dashed var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem'
      }}>
        <Layers size={14} style={{ color: 'var(--accent-primary)', flexShrink: 0 }} />
        <span>
          <strong>Current Learning Focus:</strong>{' '}
          {genuineSamples < targetSamples 
            ? `Calibrating personal preparation duration bias and estimating day-of-week lateness variance.`
            : `Fine-tuning Random Forest tree splits on latest morning routines and monitoring for habit drift.`}
        </span>
      </div>
    </div>
  );
};
