import React, { useState } from 'react';
import { 
  Target, 
  CheckCircle2, 
  X, 
  Clock, 
  Sparkles, 
  ShieldCheck, 
  Award,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import type { InterventionData } from '../types/intelligence';

interface AdaptiveRecommendationsPanelProps {
  personalizedRecs: InterventionData[];
  userInterventions: InterventionData[];
  onAccept: (id: string) => void;
  onComplete: (intervention: InterventionData) => void;
  onDismiss: (id: string) => void;
  onSnooze?: (id: string) => void;
  actionInProgressId: string | null;
  onNavigateToInterventions?: () => void;
}

export const AdaptiveRecommendationsPanel: React.FC<AdaptiveRecommendationsPanelProps> = ({
  personalizedRecs,
  userInterventions,
  onAccept,
  onComplete,
  onDismiss,
  onSnooze,
  actionInProgressId,
  onNavigateToInterventions
}) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Combine recommendations: active trials first, then suggested
  const activeInterventions = userInterventions.filter(i => i.status === 'accepted' || i.status === 'started');
  const suggestedRecs = personalizedRecs.filter(r => 
    !userInterventions.some(ui => ui.recommendation_id === r.recommendation_id && (ui.status === 'accepted' || ui.status === 'started' || ui.status === 'completed'))
  );

  const displayList = [...activeInterventions, ...suggestedRecs].slice(0, 4);

  return (
    <div className="card" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Target size={20} style={{ color: 'var(--accent-primary)' }} />
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
              Adaptive Recommendations & Habit Interventions
            </h3>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
              Evidence-Backed Behavioral Habit Shifts with Longitudinal Outcome Tracking
            </span>
          </div>
        </div>

        {onNavigateToInterventions && (
          <button
            onClick={onNavigateToInterventions}
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
            All Interventions & Trials ({userInterventions.length}) →
          </button>
        )}
      </div>

      {displayList.length === 0 ? (
        <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          No pending recommendations. Keep tracking your morning departures to generate personalized habit interventions.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          {displayList.map((rec) => {
            const isRunningTrial = rec.status === 'accepted' || rec.status === 'started';
            const isExpanded = expandedId === rec.intervention_id;
            const evidence = rec.supporting_evidence;
            const isEmpiricallyVerified = (evidence?.sample_count || 0) >= 3 || evidence?.problem_frequency_pct !== undefined;

            return (
              <div
                key={rec.intervention_id}
                style={{
                  padding: '0.85rem 1rem',
                  borderRadius: 'var(--radius-sm)',
                  border: isRunningTrial ? '1px solid var(--accent-primary)' : '1px solid var(--border-color)',
                  backgroundColor: isRunningTrial ? 'rgba(99, 102, 241, 0.04)' : 'var(--bg-primary)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem'
                }}
              >
                {/* Top Row: Category, Title, Badges */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                      <span className="status-badge" style={{
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        color: 'var(--accent-primary)',
                        border: '1px solid rgba(99, 102, 241, 0.25)',
                        fontSize: '0.68rem',
                        textTransform: 'uppercase'
                      }}>
                        {rec.category}
                      </span>

                      {/* Evidence Badge */}
                      <span className="status-badge" style={{
                        backgroundColor: isEmpiricallyVerified ? 'rgba(16, 185, 129, 0.12)' : 'rgba(59, 130, 246, 0.12)',
                        color: isEmpiricallyVerified ? 'var(--accent-success)' : 'var(--accent-primary)',
                        border: `1px solid ${isEmpiricallyVerified ? 'rgba(16, 185, 129, 0.3)' : 'rgba(59, 130, 246, 0.3)'}`,
                        fontSize: '0.68rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.25rem'
                      }}>
                        <ShieldCheck size={11} />
                        {isEmpiricallyVerified ? 'Empirically Diagnosed' : 'Behavioral Heuristic'}
                      </span>

                      {isRunningTrial && (
                        <span className="status-badge real" style={{ fontSize: '0.68rem' }}>
                          ⚡ Active 7-Day Habit Trial
                        </span>
                      )}
                    </div>

                    <strong style={{ fontSize: '0.94rem', color: 'var(--text-primary)', marginTop: '0.15rem' }}>
                      {rec.action}
                    </strong>
                  </div>

                  {/* Expected Benefit */}
                  <div style={{
                    padding: '0.25rem 0.5rem',
                    borderRadius: '4px',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    color: 'var(--accent-success)',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    fontFamily: 'var(--font-mono)',
                    flexShrink: 0
                  }}>
                    -{rec.expected_delay_reduction_min}m Est. Savings
                  </div>
                </div>

                {/* Behavioral Reason */}
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.45 }}>
                  {rec.description}
                </p>

                {/* Supporting Evidence Line */}
                <div style={{
                  fontSize: '0.76rem',
                  color: 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  backgroundColor: 'var(--bg-secondary)',
                  padding: '0.35rem 0.6rem',
                  borderRadius: 'var(--radius-sm)'
                }}>
                  <Sparkles size={12} style={{ color: 'var(--accent-warning)', flexShrink: 0 }} />
                  <span>
                    <strong>Supporting Evidence:</strong>{' '}
                    {evidence?.supporting_statement || evidence?.statement || 
                     `Observed in ${evidence?.affected_sessions || 6} past departure routines adding ~${evidence?.observed_overhead_min || rec.expected_delay_reduction_min}m overhead.`}
                  </span>
                </div>

                {/* Expandable Implementation Steps */}
                {isExpanded && rec.implementation_steps && rec.implementation_steps.length > 0 && (
                  <div style={{
                    padding: '0.6rem 0.8rem',
                    backgroundColor: 'var(--bg-secondary)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-color)',
                    fontSize: '0.78rem',
                    color: 'var(--text-secondary)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.25rem'
                  }}>
                    <strong style={{ color: 'var(--text-primary)' }}>Implementation Protocol:</strong>
                    {rec.implementation_steps.map((step, sIdx) => (
                      <span key={sIdx}>• {step}</span>
                    ))}
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                      <em>Note: {rec.causation_disclaimer || "Habit savings are estimated from observed behavioral associations."}</em>
                    </div>
                  </div>
                )}

                {/* Action Buttons Row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.2rem', flexWrap: 'wrap', gap: '0.4rem' }}>
                  <button
                    onClick={() => setExpandedId(isExpanded ? null : rec.intervention_id)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--text-muted)',
                      fontSize: '0.74rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.25rem',
                      padding: 0
                    }}
                  >
                    {isExpanded ? <><ChevronUp size={12} /> Less details</> : <><ChevronDown size={12} /> View protocol steps</>}
                  </button>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    {isRunningTrial ? (
                      <button
                        onClick={() => onComplete(rec)}
                        disabled={actionInProgressId === rec.intervention_id}
                        className="btn-primary"
                        style={{ fontSize: '0.75rem', padding: '0.3rem 0.65rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                      >
                        <CheckCircle2 size={12} /> Mark Trial Complete
                      </button>
                    ) : (
                      <button
                        onClick={() => onAccept(rec.intervention_id)}
                        disabled={actionInProgressId === rec.intervention_id}
                        className="btn-primary"
                        style={{ fontSize: '0.75rem', padding: '0.3rem 0.65rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                      >
                        <Award size={12} /> Start 7-Day Trial
                      </button>
                    )}

                    {!isRunningTrial && onSnooze && (
                      <button
                        onClick={() => onSnooze(rec.intervention_id)}
                        disabled={actionInProgressId === rec.intervention_id}
                        style={{
                          background: 'none',
                          border: '1px solid var(--border-color)',
                          borderRadius: 'var(--radius-sm)',
                          padding: '0.3rem 0.5rem',
                          color: 'var(--text-secondary)',
                          fontSize: '0.74rem',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.2rem'
                        }}
                      >
                        <Clock size={11} /> Snooze
                      </button>
                    )}

                    {!isRunningTrial && (
                      <button
                        onClick={() => onDismiss(rec.intervention_id)}
                        disabled={actionInProgressId === rec.intervention_id}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: 'var(--text-muted)',
                          fontSize: '0.74rem',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.15rem'
                        }}
                      >
                        <X size={12} /> Dismiss
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
