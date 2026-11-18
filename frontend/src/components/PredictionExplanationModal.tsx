import React from 'react';
import { 
  X, 
  BrainCircuit, 
  Clock, 
  CheckCircle2, 
  AlertTriangle 
} from 'lucide-react';
import type { PredictionExplanationData, RiskPredictionData } from '../types/intelligence';

interface PredictionExplanationModalProps {
  isOpen: boolean;
  onClose: () => void;
  explanation: PredictionExplanationData | null;
  riskPrediction: RiskPredictionData | null;
}

export const PredictionExplanationModal: React.FC<PredictionExplanationModalProps> = ({
  isOpen,
  onClose,
  explanation,
  riskPrediction
}) => {
  if (!isOpen) return null;

  const topPos = explanation?.top_positive_contributors || [];
  const topMit = explanation?.top_mitigating_factors || [];
  const histComps = explanation?.historical_comparisons || [];
  const limitations = explanation?.limitations || [
    'Predictions reflect associative statistical patterns and do not prove direct causation.',
    'Model accuracy depends on consistent logging of morning departure routines.',
    'Prediction interval represents a 90% coverage level under observed conditions.'
  ];

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.65)',
      backdropFilter: 'blur(3px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '1rem'
    }}>
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border-color)',
        width: '100%',
        maxWidth: '720px',
        maxHeight: '90vh',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.3)'
      }}>
        {/* Modal Header */}
        <div style={{
          padding: '1.25rem 1.5rem',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          position: 'sticky',
          top: 0,
          backgroundColor: 'var(--bg-secondary)',
          zIndex: 10
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <BrainCircuit size={22} style={{ color: 'var(--accent-primary)' }} />
            <div>
              <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
                Prediction Explainability & Behavioral Reasoning
              </h2>
              <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                Model: {explanation?.model_version || riskPrediction?.model_version || 'v2.4-rf'} · Method: {explanation?.explanation_method || 'Tree Feature Contributions'}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '0.25rem'
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Overview Banner */}
          <div style={{
            padding: '1rem',
            backgroundColor: 'var(--bg-primary)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-color)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '0.75rem'
          }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Predicted Departure Delay
              </span>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                +{explanation?.predicted_delay_minutes?.toFixed(1) || riskPrediction?.predicted_delay_minutes?.toFixed(1) || '0.0'}m
              </div>
            </div>

            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Risk Classification
              </span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-warning)', textTransform: 'capitalize' }}>
                {riskPrediction?.risk_category || 'Moderate'} Risk
              </div>
            </div>

            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                90% Conformal Interval
              </span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                [{riskPrediction?.prediction_interval_lower?.toFixed(1) || '0.0'}m, {riskPrediction?.prediction_interval_upper?.toFixed(1) || '10.0'}m]
              </div>
            </div>
          </div>

          {/* Non-Causal Associative Disclaimer */}
          <div style={{
            padding: '0.75rem 1rem',
            backgroundColor: 'rgba(99, 102, 241, 0.05)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            lineHeight: 1.45
          }}>
            <strong style={{ color: 'var(--accent-primary)' }}>Important Associative Framing:</strong>{' '}
            {explanation?.causation_disclaimer ||
              'Model-estimated contributions represent statistical associations learned across observed routines, not proven causal levers. Delays arise from interconnected morning dynamics.'}
          </div>

          {/* Section 1: Factors Increasing Delay */}
          <div>
            <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--accent-danger)', margin: '0 0 0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <AlertTriangle size={16} /> Factors Increasing Delay (+Minutes)
            </h4>
            {topPos.length === 0 ? (
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', padding: '0.5rem', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)' }}>
                No significant delay-increasing factors diagnosed for this routine.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {topPos.map((item, idx) => (
                  <div key={idx} style={{
                    padding: '0.65rem 0.85rem',
                    backgroundColor: 'var(--bg-primary)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid rgba(239, 68, 68, 0.2)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: '0.5rem'
                  }}>
                    <div>
                      <strong style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>{item.factor}</strong>
                      <p style={{ margin: '0.15rem 0 0', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        {item.description}
                      </p>
                    </div>
                    <strong style={{ color: 'var(--accent-danger)', fontFamily: 'var(--font-mono)', fontSize: '0.9rem', flexShrink: 0 }}>
                      +{Math.abs(item.impact_value).toFixed(1)}m
                    </strong>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Section 2: Mitigating Factors */}
          <div>
            <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--accent-success)', margin: '0 0 0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <CheckCircle2 size={16} /> Mitigating Factors Protecting Departure (-Minutes)
            </h4>
            {topMit.length === 0 ? (
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', padding: '0.5rem', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)' }}>
                No mitigating factors active. Preparing bag or simplifying schedule will reduce risk.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {topMit.map((item, idx) => (
                  <div key={idx} style={{
                    padding: '0.65rem 0.85rem',
                    backgroundColor: 'var(--bg-primary)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid rgba(16, 185, 129, 0.2)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: '0.5rem'
                  }}>
                    <div>
                      <strong style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>{item.factor}</strong>
                      <p style={{ margin: '0.15rem 0 0', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        {item.description}
                      </p>
                    </div>
                    <strong style={{ color: 'var(--accent-success)', fontFamily: 'var(--font-mono)', fontSize: '0.9rem', flexShrink: 0 }}>
                      -{Math.abs(item.impact_value).toFixed(1)}m
                    </strong>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Section 3: Historical Baseline Comparisons */}
          {histComps.length > 0 && (
            <div>
              <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-primary)', margin: '0 0 0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Clock size={16} style={{ color: 'var(--accent-primary)' }} /> Comparison Against Your Normal Baseline
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {histComps.map((comp, idx) => (
                  <div key={idx} style={{
                    padding: '0.55rem 0.85rem',
                    backgroundColor: 'var(--bg-primary)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-color)',
                    fontSize: '0.8rem',
                    color: 'var(--text-secondary)'
                  }}>
                    <span>{comp.statement}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Section 4: Model Limitations & Caveats */}
          <div style={{
            padding: '0.75rem 1rem',
            backgroundColor: 'var(--bg-primary)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-color)',
            fontSize: '0.76rem',
            color: 'var(--text-muted)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem'
          }}>
            <strong style={{ color: 'var(--text-secondary)' }}>System Limitations & Governance:</strong>
            {limitations.map((lim, lIdx) => (
              <span key={lIdx}>• {lim}</span>
            ))}
          </div>
        </div>

        {/* Modal Footer */}
        <div style={{
          padding: '1rem 1.5rem',
          borderTop: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'flex-end',
          backgroundColor: 'var(--bg-secondary)'
        }}>
          <button onClick={onClose} className="btn-primary" style={{ fontSize: '0.85rem', padding: '0.4rem 1.25rem' }}>
            Close Explanation
          </button>
        </div>
      </div>
    </div>
  );
};
