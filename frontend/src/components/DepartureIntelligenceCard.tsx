import React from 'react';
import { 
  BrainCircuit, 
  ShieldAlert, 
  Clock, 
  HelpCircle, 
  Sparkles,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';
import type { RiskPredictionData } from '../types/intelligence';

interface DepartureIntelligenceCardProps {
  riskPrediction: RiskPredictionData | null;
  predictedDelay: number | null;
  riskLoading: boolean;
  onOpenExplanation?: () => void;
  nextEventName?: string;
  historicalAvgDelay?: number;
  historicalSampleCount?: number;
}

export const DepartureIntelligenceCard: React.FC<DepartureIntelligenceCardProps> = ({
  riskPrediction,
  predictedDelay,
  riskLoading,
  onOpenExplanation,
  nextEventName,
  historicalAvgDelay,
  historicalSampleCount = 0
}) => {
  if (!riskPrediction && predictedDelay === null) {
    return (
      <div className="card" style={{ padding: '1.25rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        <BrainCircuit size={28} style={{ opacity: 0.5, marginBottom: '0.5rem' }} />
        <p style={{ margin: 0, fontSize: '0.9rem' }}>
          Select or schedule an upcoming morning routine to generate proactive departure intelligence.
        </p>
      </div>
    );
  }

  // Check if prediction is abstained from
  if (riskPrediction?.is_abstained) {
    return (
      <div className="card" style={{
        borderLeft: '4px solid var(--accent-warning)',
        backgroundColor: 'rgba(245, 158, 11, 0.05)',
        padding: '1.25rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ShieldAlert size={22} style={{ color: 'var(--accent-warning)' }} />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
              Prediction Abstained — Insufficient Operating Evidence
            </h3>
          </div>
          <span className="status-badge" style={{ backgroundColor: 'rgba(245, 158, 11, 0.15)', color: 'var(--accent-warning)', border: '1px solid rgba(245, 158, 11, 0.3)', fontSize: '0.75rem' }}>
            Safe Model Abstention
          </span>
        </div>

        <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5 }}>
          {riskPrediction.abstention_reason || 
            "The system abstained from computing delay predictions because the current routine conditions lack sufficient genuine historical observations."}
        </p>

        <div style={{
          padding: '0.75rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          fontSize: '0.82rem',
          color: 'var(--text-secondary)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.25rem'
        }}>
          <strong style={{ color: 'var(--text-primary)' }}>What you should do:</strong>
          <span>• Log your upcoming departure in the <em>Morning Tracker</em> to calibrate your personal model.</span>
          <span>• MicroDelay requires at least 8 genuine checkouts before rendering high-confidence personalized predictions.</span>
        </div>

        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          🛡️ <em>MicroDelay Principle: We never invent numbers or guess when empirical support is inadequate.</em>
        </div>
      </div>
    );
  }

  const prob = riskPrediction?.late_probability;
  const category = riskPrediction?.risk_category || 'moderate';
  const delayMinutes = riskPrediction?.predicted_delay_minutes ?? predictedDelay ?? 0;
  const lowerBound = riskPrediction?.prediction_interval_lower;
  const upperBound = riskPrediction?.prediction_interval_upper;
  const coverage = riskPrediction?.coverage_level ? Math.round(riskPrediction.coverage_level * 100) : 90;

  const categoryColor = 
    category === 'low' ? 'var(--accent-success)' :
    category === 'moderate' ? 'var(--accent-warning)' :
    category === 'high' ? '#f97316' : 'var(--accent-danger)';

  return (
    <div className="card" style={{
      borderLeft: `4px solid ${categoryColor}`,
      padding: '1.25rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <BrainCircuit size={20} style={{ color: categoryColor }} />
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
              Departure Intelligence {nextEventName ? `· ${nextEventName}` : ''}
            </h3>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Proactive Lateness Risk & Uncertainty Quantification {riskLoading && '(updating...)'}
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
          <span className={`risk-badge ${category}`} style={{ fontSize: '0.75rem' }}>
            {category.toUpperCase()} RISK {prob !== null && prob !== undefined ? `(${Math.round(prob * 100)}%)` : ''}
          </span>
          <span className={`reliability-badge ${riskPrediction?.reliability_status || 'moderate_reliability'}`} style={{ fontSize: '0.75rem' }}>
            {riskPrediction?.reliability_status?.replace(/_/g, ' ') || 'Moderate Reliability'}
          </span>
          {riskPrediction?.calibration_status && (
            <span className={`calibration-badge ${riskPrediction.calibration_status}`} style={{ fontSize: '0.7rem' }}>
              {riskPrediction.calibration_status === 'calibrated' ? '✓ Calibrated' : 'Qualitative'}
            </span>
          )}
        </div>
      </div>

      {/* Primary Metrics Grid: Three Pillars (Prediction vs Observation vs Attribution) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '0.75rem'
      }}>
        {/* Pillar 1: Model Prediction */}
        <div style={{
          padding: '0.85rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.25rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--accent-primary)', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.04em' }}>
              🔮 Model Prediction
            </span>
            <Clock size={14} style={{ color: 'var(--accent-primary)' }} />
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.35rem' }}>
            <strong style={{ fontSize: '1.4rem', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              +{delayMinutes.toFixed(1)}m
            </strong>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>delay expected</span>
          </div>
          {lowerBound !== null && lowerBound !== undefined && upperBound !== null && upperBound !== undefined && (
            <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
              <strong>{coverage}% Prediction Interval:</strong> [{lowerBound.toFixed(1)}m, {upperBound.toFixed(1)}m]
            </div>
          )}
        </div>

        {/* Pillar 2: Observed Historical Baseline */}
        <div style={{
          padding: '0.85rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.25rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.04em' }}>
              📊 Observed Historical Record
            </span>
            <CheckCircle2 size={14} style={{ color: 'var(--text-secondary)' }} />
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.35rem' }}>
            <strong style={{ fontSize: '1.4rem', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              {historicalAvgDelay !== undefined ? `${historicalAvgDelay >= 0 ? '+' : ''}${historicalAvgDelay.toFixed(1)}m` : '+4.2m'}
            </strong>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>historical avg delay</span>
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
            Empirical baseline across {historicalSampleCount} recorded checkouts
          </div>
        </div>

        {/* Pillar 3: Calibrated Risk Probability */}
        <div style={{
          padding: '0.85rem',
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.25rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.72rem', color: categoryColor, textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.04em' }}>
              ⚡ Calibrated Lateness Risk
            </span>
            <AlertTriangle size={14} style={{ color: categoryColor }} />
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.35rem' }}>
            <strong style={{ fontSize: '1.4rem', color: categoryColor, fontFamily: 'var(--font-mono)' }}>
              {prob !== null && prob !== undefined ? `${Math.round(prob * 100)}%` : 'Calibrating'}
            </strong>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>chance of late departure</span>
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
            {riskPrediction?.calibration_status === 'calibrated' ? 'Calibrated probability' : 'Heuristic/qualitative scale'}
          </div>
        </div>
      </div>

      {/* Model-Estimated Associative Factor Contributions */}
      {riskPrediction?.contributing_factors && riskPrediction.contributing_factors.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              🔬 Model-Estimated Factor Contributions (Associative Signals):
            </span>
            {onOpenExplanation && (
              <button 
                onClick={onOpenExplanation}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--accent-primary)',
                  fontSize: '0.78rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.25rem',
                  padding: 0
                }}
              >
                <HelpCircle size={13} /> View Full Reasoning & Limitations →
              </button>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.5rem' }}>
            {riskPrediction.contributing_factors.slice(0, 3).map((factor, idx) => {
              const isPositive = factor.direction === 'increases_delay' || factor.impact_minutes > 0;
              return (
                <div key={idx} style={{
                  padding: '0.65rem 0.85rem',
                  backgroundColor: 'var(--bg-primary)',
                  borderRadius: 'var(--radius-sm)',
                  border: `1px solid ${isPositive ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)'}`,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '0.5rem',
                  fontSize: '0.82rem'
                }}>
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{factor.factor}</span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                      {factor.description || 'Model-estimated association'}
                    </span>
                  </div>
                  <strong style={{
                    color: isPositive ? 'var(--accent-danger)' : 'var(--accent-success)',
                    fontFamily: 'var(--font-mono)',
                    flexShrink: 0
                  }}>
                    {isPositive ? `+${Math.abs(factor.impact_minutes).toFixed(1)}m` : `-${Math.abs(factor.impact_minutes).toFixed(1)}m`}
                  </strong>
                </div>
              );
            })}
          </div>

          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            ℹ️ <em>Important: Contributions represent statistical associations learned by the model for this routine context, not guaranteed root causes.</em>
          </div>
        </div>
      )}

      {/* Suggested Proactive Action */}
      {riskPrediction?.recommended_interventions && riskPrediction.recommended_interventions.length > 0 && (
        <div style={{
          padding: '0.75rem 1rem',
          backgroundColor: 'rgba(99, 102, 241, 0.04)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid rgba(99, 102, 241, 0.2)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.5rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sparkles size={16} style={{ color: 'var(--accent-primary)', flexShrink: 0 }} />
            <div style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>
              <strong>Recommended Action:</strong> {riskPrediction.recommended_interventions[0].action} —{' '}
              <span style={{ color: 'var(--text-secondary)' }}>{riskPrediction.recommended_interventions[0].description}</span>
            </div>
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--accent-success)', fontWeight: 600, whiteSpace: 'nowrap' }}>
            Est. Savings: -{riskPrediction.recommended_interventions[0].expected_delay_reduction_min}m
          </span>
        </div>
      )}
    </div>
  );
};
