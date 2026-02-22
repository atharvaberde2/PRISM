import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldCheck,
  ShieldAlert,
  ArrowRight,
  TrendingUp,
  AlertTriangle,
  Lock,
  Unlock,
} from 'lucide-react';
import { usePipelineStore } from '../../store/pipeline';
import Card from '../shared/Card';
import { saveCleanedCSV } from '../../lib/mockEngine';

export default function Stage3Gate() {
  const {
    gateResult,
    setGateResult,
    setStage,
    unlockStage,
    currentBiasScore,
    baselineBias,
    cleaningSessionId,
  } = usePipelineStore();

  const [showOverride, setShowOverride] = useState(false);
  const [justification, setJustification] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [revealed, setRevealed] = useState(false);

  // Trigger the reveal animation on mount
  useEffect(() => {
    const timer = setTimeout(() => setRevealed(true), 300);
    return () => clearTimeout(timer);
  }, []);

  if (!gateResult) return null;

  const allPassed = gateResult.dimensions.every((d) => d.passed);
  const overallDelta =
    (currentBiasScore) - (baselineBias?.overallScore ?? 0);

  const handleProceed = async () => {
    if (cleaningSessionId) {
      setIsSaving(true);
      try {
        await saveCleanedCSV(cleaningSessionId);
      } catch {
        // Non-blocking — training can still use in-memory data
      } finally {
        setIsSaving(false);
      }
    }
    unlockStage(4);
    setStage(4);
  };

  const handleOverride = () => {
    if (!justification.trim()) return;
    setGateResult({
      ...gateResult,
      passed: true,
      overriddenBy: 'user',
      overrideJustification: justification,
    });
    setShowOverride(false);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Gate Status Banner */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Card
          glow
          className={`text-center py-10 space-y-4 ${
            allPassed || gateResult.overriddenBy
              ? 'border-positive/30'
              : 'border-negative/30'
          }`}
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', delay: 0.2 }}
            className="flex justify-center"
          >
            <div
              className={`w-20 h-20 rounded-2xl flex items-center justify-center ${
                allPassed || gateResult.overriddenBy
                  ? 'bg-positive-bg'
                  : 'bg-negative-bg'
              }`}
            >
              {allPassed || gateResult.overriddenBy ? (
                <Unlock size={36} className="text-positive" />
              ) : (
                <Lock size={36} className="text-negative" />
              )}
            </div>
          </motion.div>

          <div>
            <h2
              className={`text-2xl font-bold ${
                allPassed || gateResult.overriddenBy
                  ? 'text-positive'
                  : 'text-negative'
              }`}
            >
              {allPassed
                ? 'Bias Gate Passed'
                : gateResult.overriddenBy
                  ? 'Gate Overridden'
                  : 'Bias Gate Failed'}
            </h2>
            <p className="text-sm text-text-muted mt-2 max-w-md mx-auto">
              {allPassed
                ? 'Your cleaning improved bias across all dimensions. The pipeline is cleared for training.'
                : gateResult.overriddenBy
                  ? `Gate overridden with justification. This override is logged in the audit trail.`
                  : 'One or more bias dimensions worsened after cleaning. Training is blocked until resolved.'}
            </p>
          </div>

          {/* Overall delta */}
          <div className="flex flex-wrap items-center justify-center gap-6 pt-2">
            <div className="text-center">
              <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1">
                Raw Dataset
              </p>
              <p className="text-3xl font-bold font-mono text-negative">
                {Number(baselineBias?.overallScore ?? 0).toFixed(1)}
              </p>
            </div>
            <TrendingUp size={24} className="text-positive" />
            <div className="text-center">
              <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1">
                Cleaned Dataset
              </p>
              <p className="text-3xl font-bold font-mono text-positive">
                {Number(currentBiasScore).toFixed(1)}
              </p>
            </div>
            <div className="text-center border-l border-border pl-6">
              <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1">
                Improvement
              </p>
              <p className="text-3xl font-bold font-mono text-positive">
                +{Number(overallDelta).toFixed(1)}
              </p>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Dimension Breakdown */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-text-secondary">
          Bias Scorecard — All 5 Dimensions
        </h3>

        <div className="grid gap-3">
          {gateResult.dimensions.map((dim, idx) => (
            <motion.div
              key={dim.name}
              initial={{ opacity: 0, x: -20 }}
              animate={revealed ? { opacity: 1, x: 0 } : {}}
              transition={{ delay: 0.4 + idx * 0.15 }}
            >
              <Card hover>
                <div className="flex items-center gap-4">
                  {/* Pass/fail icon */}
                  <div
                    className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                      dim.passed
                        ? 'bg-positive-bg text-positive'
                        : 'bg-negative-bg text-negative'
                    }`}
                  >
                    {dim.passed ? (
                      <ShieldCheck size={20} />
                    ) : (
                      <ShieldAlert size={20} />
                    )}
                  </div>

                  {/* Label */}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-text-primary">
                      {dim.label}
                    </p>
                  </div>

                  {/* Before/After */}
                  <div className="flex flex-wrap items-center gap-4 md:gap-6">
                    <div className="text-center">
                      <p className="text-[9px] uppercase tracking-wider text-text-muted">
                        Before
                      </p>
                      <p className="text-lg font-bold font-mono text-negative">
                        {dim.before}
                      </p>
                    </div>

                    <div className="text-text-muted">&rarr;</div>

                    <div className="text-center">
                      <p className="text-[9px] uppercase tracking-wider text-text-muted">
                        After
                      </p>
                      <p
                        className={`text-lg font-bold font-mono ${
                          dim.passed ? 'text-positive' : 'text-negative'
                        }`}
                      >
                        {dim.after}
                      </p>
                    </div>

                    <div
                      className={`px-2 py-1 rounded-md text-xs font-mono font-semibold ${
                        dim.delta > 0
                          ? 'bg-positive-bg text-positive'
                          : 'bg-negative-bg text-negative'
                      }`}
                    >
                      {dim.delta > 0 ? '+' : ''}
                      {dim.delta}
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Override Section (for failed gates) */}
      {!allPassed && !gateResult.overriddenBy && (
        <Card className="border-negative/20">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle size={16} className="text-warning" />
            <span className="text-sm font-medium text-warning">
              Override Required
            </span>
          </div>
          {!showOverride ? (
            <button
              onClick={() => setShowOverride(true)}
              className="text-xs text-text-muted hover:text-warning transition-colors"
            >
              I understand the risks — let me override with justification
            </button>
          ) : (
            <div className="space-y-3">
              <textarea
                value={justification}
                onChange={(e) => setJustification(e.target.value)}
                placeholder="Provide justification for overriding the bias gate..."
                className="w-full bg-bg-primary border border-border rounded-lg px-3 py-2 text-sm text-text-primary resize-none h-20 focus:outline-none focus-visible:ring-1 focus-visible:ring-warning focus:border-warning"
              />
              <button
                onClick={handleOverride}
                disabled={!justification.trim()}
                className="px-4 py-2 rounded-lg bg-warning/20 hover:bg-warning/30 text-warning text-xs font-medium transition-colors disabled:opacity-50"
              >
                Override & Log Justification
              </button>
            </div>
          )}
        </Card>
      )}

      {/* Proceed button */}
      {(allPassed || gateResult.overriddenBy) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2 }}
        >
          <Card glow className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-text-primary">
                Pipeline cleared for training
              </p>
              <p className="text-xs text-text-muted">
                Your cleaning improved bias by {Number(overallDelta).toFixed(1)} points across all
                dimensions.
              </p>
            </div>
            <button
              onClick={handleProceed}
              disabled={isSaving}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-light text-white text-sm font-medium transition-colors disabled:opacity-60"
            >
              {isSaving ? 'Saving data...' : 'Begin Training'}
              <ArrowRight size={16} />
            </button>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
