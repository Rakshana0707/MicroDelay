import React from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  CheckCircle2, 
  ArrowRight
} from 'lucide-react';
import type { DriftAnalysisData } from '../types/intelligence';

interface DriftAlertBannerProps {
  driftAnalysis: DriftAnalysisData | null;
  onAcknowledge?: (eventId: string) => void;
  onViewDetails?: () => void;
}

export const DriftAlertBanner: React.FC<DriftAlertBannerProps> = ({
  driftAnalysis,
  onAcknowledge,
  onViewDetails
}) => {
  if (!driftAnalysis || !driftAnalysis.events_detected || driftAnalysis.events_detected.length === 0) {
    return null;
  }

  // Filter for active significant events
  const activeEvents = driftAnalysis.events_detected.filter(e => 
    e.status !== 'dismissed' && e.status !== 'acknowledged' && e.is_statistically_significant
  );

  if (activeEvents.length === 0) {
    return null;
  }

  const primaryEvent = activeEvents[0];
  const isPositive = primaryEvent.valence === 'positive' || primaryEvent.drift_type === 'behavioral_improvement';
  const isDeterioration = primaryEvent.drift_type === 'behavioral_deterioration' || primaryEvent.valence === 'negative';

  const borderColor = isPositive ? 'rgba(16, 185, 129, 0.4)' : isDeterioration ? 'rgba(239, 68, 68, 0.4)' : 'rgba(139, 92, 246, 0.4)';
  const bgColor = isPositive 
    ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(59, 130, 246, 0.04) 100%)'
    : isDeterioration 
    ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(245, 158, 11, 0.04) 100%)'
    : 'linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(59, 130, 246, 0.04) 100%)';

  const iconColor = isPositive ? 'var(--accent-success)' : isDeterioration ? 'var(--accent-danger)' : '#8b5cf6';

  return (
    <div style={{
      padding: '0.85rem 1.15rem',
      borderRadius: 'var(--radius-md)',
      border: `1px solid ${borderColor}`,
      background: bgColor,
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      flexWrap: 'wrap',
      gap: '0.75rem',
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', maxWidth: '75%' }}>
        <div style={{
          padding: '0.45rem',
          borderRadius: '8px',
          backgroundColor: isPositive ? 'rgba(16, 185, 129, 0.2)' : isDeterioration ? 'rgba(239, 68, 68, 0.2)' : 'rgba(139, 92, 246, 0.2)',
          color: iconColor,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0
        }}>
          {isPositive ? <TrendingDown size={20} /> : <TrendingUp size={20} />}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
            <strong style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>
              {isPositive 
                ? 'Habit Improvement: Punctuality Progress Detected' 
                : isDeterioration 
                ? 'Routine Friction: Increased Preparation Overhead Detected' 
                : 'Routine Evolution: Habit Distribution Shift'}
            </strong>
            <span className="status-badge" style={{
              fontSize: '0.68rem',
              backgroundColor: isPositive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              color: isPositive ? 'var(--accent-success)' : 'var(--accent-danger)'
            }}>
              Significant (p &lt; 0.05)
            </span>
          </div>

          <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            {primaryEvent.humanized_summary}
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {onAcknowledge && (
          <button
            onClick={() => onAcknowledge(primaryEvent.drift_event_id)}
            style={{
              background: 'none',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.35rem 0.65rem',
              fontSize: '0.75rem',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem'
            }}
          >
            <CheckCircle2 size={12} /> Got It
          </button>
        )}

        {onViewDetails && (
          <button
            onClick={onViewDetails}
            className="btn-primary"
            style={{
              fontSize: '0.75rem',
              padding: '0.35rem 0.75rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem'
            }}
          >
            View Routine Analysis <ArrowRight size={12} />
          </button>
        )}
      </div>
    </div>
  );
};
