import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ArrowRight,
  Wrench,
  Hash,
  Type,
  Loader2,
  Undo2,
  ChevronRight,
  Shield,
  Star,
  SkipForward,
  CircleCheck,
  TriangleAlert,
  CircleAlert,
  FlaskConical,
  Eye,
  TrendingUp,
  TrendingDown,
  Zap,
  ArrowDown,
  ArrowUp,
  Minus,
  Info,
  Play,
  Download,
} from 'lucide-react';
import { usePipelineStore } from '../../store/pipeline';
import Card from '../shared/Card';
import BiasScoreBadge from '../shared/BiasScoreBadge';
import {
  initCleaningSession,
  suggestTechnique,
  previewTechnique,
  commitTechnique,
  revertFeature,
  computeGateResult,
} from '../../lib/mockEngine';
import type {
  TechniqueSuggestion,
  SuggestResult,
  CleaningPreview,
  FeatureMetrics,
} from '../../lib/types';

// ---------------------------------------------------------------------------
// Metric helpers
// ---------------------------------------------------------------------------

const METRIC_KEYS = ['p_value', 'js_divergence', 'chi_squared', 'missingness_rate', 'skewness'] as const;

const METRIC_LABELS: Record<string, string> = {
  p_value: 'P-Value',
  js_divergence: 'JS Divergence',
  chi_squared: 'Chi-Squared',
  missingness_rate: 'Missingness',
  skewness: 'Skewness',
};

const METRIC_TOOLTIPS: Record<string, string> = {
  p_value: 'How confident we are that groups are actually different (lower p = more different)',
  js_divergence: 'How similar the data distributions look across groups (lower = more similar)',
  chi_squared: 'How strongly this feature separates groups (lower = less separation)',
  missingness_rate: 'Whether missing data is unevenly spread across groups (lower = more even)',
  skewness: 'Whether data shapes differ across groups (lower = more similar shapes)',
};

function normValue(fm: FeatureMetrics, key: string): number {
  switch (key) {
    case 'p_value': return fm.norm_pval;
    case 'js_divergence': return fm.norm_jsd;
    case 'chi_squared': return fm.norm_effect;
    case 'missingness_rate': return fm.norm_miss;
    case 'skewness': return fm.norm_skew;
    default: return 0;
  }
}

function rawValue(fm: FeatureMetrics, key: string): string {
  switch (key) {
    case 'p_value': return fm.p_value < 0.001 ? '< 0.001' : fm.p_value.toFixed(4);
    case 'js_divergence': return fm.jsd.toFixed(4);
    case 'chi_squared': return fm.effect_size.toFixed(3);
    case 'missingness_rate': return `${(fm.missingness_gap * 100).toFixed(1)}%`;
    case 'skewness': return fm.skew_diff.toFixed(3);
    default: return '—';
  }
}

function verdict(norm: number) {
  if (norm < 0.15) return { label: 'Low', color: 'text-positive', bg: 'bg-positive/10', ring: 'ring-positive/20', bar: 'bg-positive' };
  if (norm < 0.4) return { label: 'Medium', color: 'text-warning', bg: 'bg-warning/10', ring: 'ring-warning/20', bar: 'bg-warning' };
  return { label: 'High', color: 'text-negative', bg: 'bg-negative/10', ring: 'ring-negative/20', bar: 'bg-negative' };
}

function featureBiasScore(fm: FeatureMetrics): number {
  return Math.round((1 - fm.feature_bias) * 100);
}

function featureHealthLabel(score: number): { label: string; color: string } {
  if (score >= 80) return { label: 'Healthy', color: 'text-positive' };
  if (score >= 60) return { label: 'Fair', color: 'text-warning' };
  return { label: 'Needs Work', color: 'text-negative' };
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Step indicator showing where the user is in the flow */
function StepIndicator({ step }: { step: 1 | 2 | 3 }) {
  const steps = [
    { num: 1, label: 'Pick a Feature' },
    { num: 2, label: 'Review Fix' },
    { num: 3, label: 'Approve or Reject' },
  ];
  return (
    <div className="flex items-center justify-center gap-1">
      {steps.map((s, i) => (
        <div key={s.num} className="flex items-center gap-1">
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium transition-all ${
            s.num === step
              ? 'bg-accent/15 text-accent-light ring-1 ring-accent/30'
              : s.num < step
                ? 'bg-positive/10 text-positive'
                : 'bg-bg-primary text-text-muted'
          }`}>
            {s.num < step ? <CircleCheck size={10} /> : <span className="w-3.5 h-3.5 rounded-full border border-current text-center text-[8px] leading-[14px]">{s.num}</span>}
            <span className="hidden sm:inline">{s.label}</span>
          </div>
          {i < steps.length - 1 && <ChevronRight size={10} className="text-text-muted/40 mx-0.5" />}
        </div>
      ))}
    </div>
  );
}

/** Small metric pill for the before/after diff table */
function DeltaArrow({ delta }: { delta: number }) {
  if (Math.abs(delta) < 0.005) return <Minus size={10} className="text-text-muted" />;
  if (delta < 0) return <ArrowDown size={10} className="text-positive" />;
  return <ArrowUp size={10} className="text-negative" />;
}

/** Progress ring for overall cleaning progress */
function ProgressRing({ value, max, size = 40 }: { value: number; max: number; size?: number }) {
  const pct = max > 0 ? value / max : 0;
  const r = (size - 4) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - pct);
  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="currentColor" strokeWidth={3} className="text-bg-secondary" />
      <motion.circle
        cx={size / 2} cy={size / 2} r={r} fill="none"
        stroke="currentColor" strokeWidth={3} strokeLinecap="round"
        className="text-accent-light"
        initial={{ strokeDashoffset: circ }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        style={{ strokeDasharray: circ }}
      />
      <text x={size / 2} y={size / 2 + 1} textAnchor="middle" dominantBaseline="middle"
        className="fill-text-primary text-[10px] font-bold font-mono" transform={`rotate(90, ${size / 2}, ${size / 2})`}>
        {value}/{max}
      </text>
    </svg>
  );
}


// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function Stage2Clean() {
  const {
    uploadedFile,
    targetColumn,
    baselineBias,
    currentBiasScore,
    setCurrentBiasScore,
    cleaningSessionId,
    featureStates,
    activeFeature,
    setCleaningSessionId,
    setFeatureStates,
    setActiveFeature,
    updateFeatureState,
    setStage,
    unlockStage,
    setGateResult,
  } = usePipelineStore();

  const [isInitializing, setIsInitializing] = useState(!cleaningSessionId);
  const [initError, setInitError] = useState<string | null>(null);

  const [suggestions, setSuggestions] = useState<SuggestResult | null>(null);
  const [suggestionIdx, setSuggestionIdx] = useState(0);
  const [isSuggesting, setIsSuggesting] = useState(false);

  const [preview, setPreview] = useState<CleaningPreview | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);

  const [isCommitting, setIsCommitting] = useState(false);
  const [isReverting, setIsReverting] = useState(false);

  // Tooltip hover state
  const [hoveredMetric, setHoveredMetric] = useState<string | null>(null);

  const baselineScore = baselineBias?.overallScore ?? 0;

  // ── Initialize session on mount ────────────────────────────────
  useEffect(() => {
    if (cleaningSessionId || !uploadedFile || !targetColumn) return;
    let cancelled = false;

    (async () => {
      try {
        setIsInitializing(true);
        const result = await initCleaningSession(uploadedFile, targetColumn);
        if (cancelled) return;
        setCleaningSessionId(result.sessionId);
        setFeatureStates(result.features);
        setCurrentBiasScore(result.overallScore);
        if (result.features.length > 0) {
          setActiveFeature(result.features[0].feature);
        }
      } catch (e) {
        if (!cancelled) setInitError(String(e));
      } finally {
        if (!cancelled) setIsInitializing(false);
      }
    })();

    return () => { cancelled = true; };
  }, []);

  // ── Fetch suggestions when activeFeature changes ───────────────
  const loadSuggestions = useCallback(async (feature: string) => {
    if (!cleaningSessionId) return;
    setIsSuggesting(true);
    setSuggestions(null);
    setSuggestionIdx(0);
    setPreview(null);
    try {
      const result = await suggestTechnique(cleaningSessionId, feature);
      setSuggestions(result);
    } catch {
      /* ignore */
    } finally {
      setIsSuggesting(false);
    }
  }, [cleaningSessionId]);

  useEffect(() => {
    if (activeFeature && cleaningSessionId) {
      // Don't auto-load suggestions if we just approved (show choice UI instead)
      if (!justApproved) {
        loadSuggestions(activeFeature);
      }
    }
  }, [activeFeature, cleaningSessionId]);

  // ── Auto-preview: when suggestions arrive, auto-run preview for the top suggestion
  useEffect(() => {
    if (!suggestions || suggestions.suggestions.length === 0 || !cleaningSessionId || !activeFeature) return;
    const tech = suggestions.suggestions[0];
    if (!tech) return;

    let cancelled = false;
    setIsPreviewing(true);
    (async () => {
      try {
        const result = await previewTechnique(cleaningSessionId, activeFeature, tech.technique);
        if (!cancelled) setPreview(result);
      } catch { /* ignore */ }
      finally { if (!cancelled) setIsPreviewing(false); }
    })();

    return () => { cancelled = true; };
  }, [suggestions, cleaningSessionId, activeFeature]);

  // ── Actions ────────────────────────────────────────────────────
  const handlePreviewTechnique = async (idx: number) => {
    if (!cleaningSessionId || !activeFeature || !suggestions) return;
    const tech = suggestions.suggestions[idx];
    if (!tech) return;
    setSuggestionIdx(idx);
    setIsPreviewing(true);
    try {
      const result = await previewTechnique(cleaningSessionId, activeFeature, tech.technique);
      setPreview(result);
    } catch { /* ignore */ }
    finally { setIsPreviewing(false); }
  };

  // After approval: show choice to clean again or move on
  const [justApproved, setJustApproved] = useState(false);

  const handleApprove = async () => {
    if (!cleaningSessionId || !activeFeature || !suggestions) return;
    const tech = suggestions.suggestions[suggestionIdx];
    if (!tech) return;
    setIsCommitting(true);
    try {
      const result = await commitTechnique(cleaningSessionId, activeFeature, tech.technique);
      const currentFs = featureStates.find(f => f.feature === activeFeature);
      const history = [...(currentFs?.cleaningHistory ?? []), { technique: tech.technique, label: tech.label }];
      updateFeatureState(activeFeature, {
        status: 'cleaned',
        metrics: result.metrics,
        committedTechnique: tech.technique,
        committedTechniqueLabel: tech.label,
        cleaningHistory: history,
      });
      setCurrentBiasScore(result.overallScore);
      setPreview(null);
      setSuggestions(null);
      setJustApproved(true);
    } catch { /* ignore */ }
    finally { setIsCommitting(false); }
  };

  const handleCleanAgain = () => {
    setJustApproved(false);
    if (activeFeature && cleaningSessionId) {
      loadSuggestions(activeFeature);
    }
  };

  const handleNextFeature = () => {
    setJustApproved(false);
    setPreview(null);
    setSuggestions(null);
    const nextUncleaned = featureStates.find(f => f.feature !== activeFeature && f.status !== 'cleaned');
    if (nextUncleaned) {
      setActiveFeature(nextUncleaned.feature);
    } else {
      // All cleaned — pick the first feature that isn't the current one
      const other = featureStates.find(f => f.feature !== activeFeature);
      if (other) setActiveFeature(other.feature);
    }
  };

  const handleSkip = () => {
    setPreview(null);
    setSuggestions(null);
    setJustApproved(false);
    const nextUncleaned = featureStates.find(f => f.feature !== activeFeature && f.status !== 'cleaned');
    if (nextUncleaned) {
      setActiveFeature(nextUncleaned.feature);
    }
  };

  const handleRevert = async (feature: string) => {
    if (!cleaningSessionId) return;
    setIsReverting(true);
    try {
      const result = await revertFeature(cleaningSessionId, feature);
      updateFeatureState(feature, {
        status: 'uncleaned',
        metrics: result.metrics,
        committedTechnique: undefined,
        committedTechniqueLabel: undefined,
        cleaningHistory: [],
      });
      setCurrentBiasScore(result.overallScore);
      setJustApproved(false);
      if (activeFeature === feature) {
        loadSuggestions(feature);
      }
    } catch { /* ignore */ }
    finally { setIsReverting(false); }
  };

  const handleSelectFeature = (feature: string) => {
    setActiveFeature(feature);
    setPreview(null);
    setSuggestionIdx(0);
    setJustApproved(false);
  };

  const handleProceed = () => {
    // Compute current sub-scores from per-feature metrics
    const fms = featureStates.map(f => f.metrics);
    const n = fms.length || 1;
    const currentSubScores = {
      js_divergence: fms.reduce((s, m) => s + m.norm_jsd, 0) / n,
      chi2_effect: fms.reduce((s, m) => s + m.norm_effect, 0) / n,
      skew_diff: fms.reduce((s, m) => s + m.norm_skew, 0) / n,
      missingness_gap: fms.reduce((s, m) => s + m.norm_miss, 0) / n,
      p_value: fms.reduce((s, m) => s + m.norm_pval, 0) / n,
    };
    setGateResult(computeGateResult(baselineBias?.sub_scores, currentSubScores));
    unlockStage(3);
    setStage(3);
  };

  const activeFs = featureStates.find(f => f.feature === activeFeature);
  const currentSuggestion: TechniqueSuggestion | null =
    suggestions?.suggestions?.[suggestionIdx] ?? null;
  const cleanedCount = featureStates.filter(f => f.status === 'cleaned').length;
  const allProcessed = featureStates.length > 0 && featureStates.every(f => f.status === 'cleaned');
  const improvementDelta = Math.round((currentBiasScore - baselineScore) * 10) / 10;

  // Determine current step for step indicator
  const currentStep: 1 | 2 | 3 = !activeFs ? 1
    : justApproved ? 3
    : preview ? 3
    : isSuggesting || isPreviewing ? 2
    : currentSuggestion ? 2
    : 1;

  // ── Loading state ──────────────────────────────────────────────
  if (isInitializing) {
    return (
      <div className="max-w-5xl mx-auto space-y-6">
        <Card className="flex flex-col items-center justify-center py-20 gap-5">
          <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}>
            <Loader2 size={36} className="text-accent" />
          </motion.div>
          <div className="text-center">
            <p className="text-sm font-semibold text-text-primary">Setting up your cleaning workspace</p>
            <p className="text-xs text-text-muted mt-1.5 max-w-sm mx-auto">
              Analyzing every feature for potential bias issues. This uses real statistical tests — not simulated data.
            </p>
          </div>
        </Card>
      </div>
    );
  }

  if (initError) {
    return (
      <div className="max-w-5xl mx-auto">
        <Card className="text-center py-12">
          <AlertTriangle size={32} className="text-negative mx-auto mb-3" />
          <p className="text-sm text-negative font-medium">Failed to initialize cleaning session</p>
          <p className="text-xs text-text-muted mt-1">{initError}</p>
        </Card>
      </div>
    );
  }

  // ── Main UI ────────────────────────────────────────────────────
  return (
    <div className="max-w-6xl mx-auto space-y-4">

      {/* ── Header: How it works + Progress ── */}
      <Card className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-accent/10 flex items-center justify-center">
              <Sparkles size={18} className="text-accent-light" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-text-primary">Clean Your Data</h2>
              <p className="text-[11px] text-text-muted">
                Review each feature. We suggest the best fix — you approve or reject it.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <ProgressRing value={cleanedCount} max={featureStates.length} />
            <div className="text-right hidden sm:block">
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-text-muted font-mono">{baselineScore}</span>
                <ArrowRight size={10} className="text-text-muted" />
                <span className={`text-sm font-bold font-mono ${
                  improvementDelta > 0 ? 'text-positive' : improvementDelta < 0 ? 'text-negative' : 'text-text-primary'
                }`}>{Number(currentBiasScore).toFixed(1)}</span>
              </div>
              {improvementDelta !== 0 && (
                <p className={`text-[10px] font-mono ${improvementDelta > 0 ? 'text-positive' : 'text-negative'}`}>
                  {improvementDelta > 0 ? '+' : ''}{improvementDelta} pts improvement
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Step Indicator */}
        <StepIndicator step={currentStep} />
      </Card>

      {/* ── Two-Column Layout ── */}
      <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr] gap-4">

        {/* ── Feature List (Left) ── */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between px-1 mb-1">
            <h3 className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">Features</h3>
            <span className="text-[10px] text-text-muted">{cleanedCount}/{featureStates.length}</span>
          </div>
          {featureStates.map((fs, i) => {
            const isActive = fs.feature === activeFeature;
            const score = featureBiasScore(fs.metrics);
            const health = featureHealthLabel(score);
            return (
              <motion.button
                key={fs.feature}
                onClick={() => handleSelectFeature(fs.feature)}
                className={`
                  w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-left transition-all
                  ${isActive
                    ? 'bg-accent/8 ring-1 ring-accent/30 shadow-[0_0_12px_rgba(99,102,241,0.06)]'
                    : 'hover:bg-bg-card-hover'}
                `}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.03 }}
              >
                {/* Status dot */}
                <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${
                  fs.status === 'cleaned' ? 'bg-positive/15' : isActive ? 'bg-accent/10' : 'bg-bg-primary'
                }`}>
                  {fs.status === 'cleaned' ? (
                    <CheckCircle2 size={14} className="text-positive" />
                  ) : fs.dtype === 'numeric' ? (
                    <Hash size={11} className={isActive ? 'text-accent-light' : 'text-text-muted'} />
                  ) : (
                    <Type size={11} className={isActive ? 'text-accent-light' : 'text-text-muted'} />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <p className={`text-xs font-medium truncate ${isActive ? 'text-accent-light' : 'text-text-primary'}`}>
                    {fs.feature}
                  </p>
                  <p className={`text-[10px] truncate ${
                    fs.status === 'cleaned' ? 'text-positive' : `${health.color} opacity-70`
                  }`}>
                    {fs.status === 'cleaned'
                      ? (fs.cleaningHistory.length > 1
                        ? `${fs.cleaningHistory.length} techniques applied`
                        : fs.committedTechniqueLabel ?? 'Cleaned')
                      : health.label}
                  </p>
                </div>

                {/* Score badge */}
                <div className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded flex-shrink-0 ${
                  score >= 80 ? 'text-positive bg-positive/10' : score >= 60 ? 'text-warning bg-warning/10' : 'text-negative bg-negative/10'
                }`}>
                  {score}
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* ── Main Panel (Right) ── */}
        <div className="space-y-4">
          <AnimatePresence mode="wait">
            {activeFs && (
              <motion.div
                key={activeFeature}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
                className="space-y-4"
              >
                {/* ── Feature Header ── */}
                <Card>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                        activeFs.dtype === 'numeric' ? 'bg-accent/10' : 'bg-warning/10'
                      }`}>
                        {activeFs.dtype === 'numeric'
                          ? <Hash size={18} className="text-accent-light" />
                          : <Type size={18} className="text-warning" />}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h2 className="text-base font-semibold text-text-primary">{activeFs.feature}</h2>
                          {activeFs.status === 'cleaned' && (
                            <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-positive/10 text-positive ring-1 ring-positive/20">
                              <CheckCircle2 size={10} /> Approved
                            </span>
                          )}
                        </div>
                        <p className="text-[11px] text-text-muted mt-0.5">
                          {activeFs.dtype === 'numeric' ? 'Number column' : 'Category column'} — fairness score: {featureBiasScore(activeFs.metrics)}/100
                        </p>
                      </div>
                    </div>
                    {activeFs.status === 'cleaned' && (
                      <button
                        onClick={() => handleRevert(activeFs.feature)}
                        disabled={isReverting}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-bg-primary hover:bg-negative/10 text-text-muted hover:text-negative text-xs font-medium transition-colors disabled:opacity-50 border border-border"
                      >
                        <Undo2 size={11} />
                        {isReverting ? 'Undoing...' : 'Undo'}
                      </button>
                    )}
                  </div>
                </Card>

                {/* ── Current Metrics Table ── */}
                <Card className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-semibold text-text-secondary flex items-center gap-2">
                      <FlaskConical size={12} className="text-accent-light" />
                      Bias Metrics
                    </h3>
                    {suggestions && activeFs.status !== 'cleaned' && (
                      <span className="text-[10px] text-accent-light bg-accent/8 px-2 py-0.5 rounded-full font-medium">
                        Worst: {METRIC_LABELS[suggestions.worstMetric] ?? suggestions.worstMetric}
                      </span>
                    )}
                  </div>

                  <div className="grid gap-2">
                    {METRIC_KEYS.map((key) => {
                      const norm = normValue(activeFs.metrics, key);
                      const v = verdict(norm);
                      const isWorst = suggestions?.worstMetric === key;
                      return (
                        <div
                          key={key}
                          className={`relative flex items-center gap-3 px-3 py-2 rounded-lg transition-all ${
                            isWorst ? 'bg-accent/5 ring-1 ring-accent/20' : 'bg-bg-primary/60'
                          }`}
                          onMouseEnter={() => setHoveredMetric(key)}
                          onMouseLeave={() => setHoveredMetric(null)}
                        >
                          {/* Bias level bar (background) */}
                          <div className="absolute inset-0 rounded-lg overflow-hidden">
                            <motion.div
                              className={`h-full ${v.bar} opacity-[0.04]`}
                              initial={{ width: 0 }}
                              animate={{ width: `${Math.max(norm * 100, 1)}%` }}
                              transition={{ duration: 0.6, ease: 'easeOut' }}
                            />
                          </div>

                          {/* Content */}
                          <div className="relative flex items-center gap-3 w-full">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-1.5">
                                <span className="text-[11px] font-medium text-text-secondary">{METRIC_LABELS[key]}</span>
                                {isWorst && <Zap size={9} className="text-accent-light" />}
                                <Info size={9} className="text-text-muted/40 cursor-help" />
                              </div>
                              {/* Tooltip */}
                              <AnimatePresence>
                                {hoveredMetric === key && (
                                  <motion.p
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="text-[10px] text-text-muted leading-relaxed mt-0.5"
                                  >
                                    {METRIC_TOOLTIPS[key]}
                                  </motion.p>
                                )}
                              </AnimatePresence>
                            </div>
                            <span className="text-[11px] font-mono text-text-primary flex-shrink-0">{rawValue(activeFs.metrics, key)}</span>
                            <span className={`text-[9px] font-semibold px-2 py-0.5 rounded-full ${v.color} ${v.bg} ring-1 ${v.ring} flex-shrink-0`}>
                              {v.label}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Card>

                {/* ── Just approved: show Clean Again / Next Feature choice ── */}
                {justApproved && activeFs.status === 'cleaned' && (
                  <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
                    <Card className="space-y-4 ring-1 ring-positive/20">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-positive/10 flex items-center justify-center ring-1 ring-positive/20">
                          <CheckCircle2 size={18} className="text-positive" />
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-semibold text-positive">Cleaning Applied</p>
                          <p className="text-xs text-text-muted mt-0.5">
                            <span className="text-text-secondary font-medium">{activeFs.committedTechniqueLabel ?? 'technique'}</span> committed to training data.
                          </p>
                        </div>
                      </div>

                      {/* Cleaning history */}
                      {activeFs.cleaningHistory.length > 0 && (
                        <div className="space-y-1">
                          <span className="text-[10px] text-text-muted font-semibold uppercase tracking-wider">Applied so far:</span>
                          <div className="flex flex-wrap gap-1.5">
                            {activeFs.cleaningHistory.map((h, i) => (
                              <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-positive/8 text-positive ring-1 ring-positive/20 font-medium">
                                {i + 1}. {h.label}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Choice buttons */}
                      <div className="flex items-center gap-3 pt-1">
                        <button
                          onClick={handleCleanAgain}
                          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent/10 hover:bg-accent/20 text-accent-light text-sm font-semibold transition-all ring-1 ring-accent/20 hover:ring-accent/40"
                        >
                          <Wrench size={14} />
                          Clean Again
                        </button>
                        <button
                          onClick={handleNextFeature}
                          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-bg-primary hover:bg-bg-card-hover text-text-secondary text-sm font-medium transition-all border border-border"
                        >
                          Next Feature
                          <ChevronRight size={14} />
                        </button>
                        <button
                          onClick={() => handleRevert(activeFs.feature)}
                          disabled={isReverting}
                          className="ml-auto flex items-center gap-1.5 px-3 py-2.5 text-text-muted hover:text-negative text-xs transition-colors"
                        >
                          <Undo2 size={11} />
                          {isReverting ? 'Undoing...' : 'Undo All'}
                        </button>
                      </div>
                    </Card>
                  </motion.div>
                )}

                {/* ── Cleaning suggestion panel ── */}
                {!justApproved && (
                  <>
                    {/* Loading */}
                    {(isSuggesting) && (
                      <Card className="flex items-center justify-center py-10 gap-3">
                        <Loader2 size={20} className="text-accent animate-spin" />
                        <span className="text-xs text-text-muted">Finding the best cleaning method...</span>
                      </Card>
                    )}

                    {/* Suggested technique card */}
                    {currentSuggestion && !isSuggesting && (
                      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
                        <Card className="space-y-4">
                          {/* Technique header */}
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex items-start gap-3">
                              <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${
                                currentSuggestion.isHero ? 'bg-positive/10' : 'bg-negative/10'
                              }`}>
                                {currentSuggestion.isHero
                                  ? <Star size={16} className="text-positive" />
                                  : <AlertTriangle size={16} className="text-negative" />}
                              </div>
                              <div>
                                <div className="flex items-center gap-2 flex-wrap">
                                  <h3 className="text-sm font-semibold text-text-primary">{currentSuggestion.label}</h3>
                                  <span className={`text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded-full ${
                                    currentSuggestion.isHero ? 'bg-positive/10 text-positive' : 'bg-negative/10 text-negative'
                                  }`}>
                                    {currentSuggestion.isHero ? 'Recommended' : 'Compare'}
                                  </span>
                                  <span className="text-[9px] text-text-muted bg-bg-primary px-2 py-0.5 rounded-full">
                                    {currentSuggestion.category}
                                  </span>
                                </div>
                                <p className="text-xs text-text-muted mt-1 leading-relaxed max-w-lg">{currentSuggestion.description}</p>
                              </div>
                            </div>
                          </div>

                          {/* Alternative techniques (if available) */}
                          {suggestions && suggestions.suggestions.length > 1 && (
                            <div className="space-y-1.5">
                              <span className="text-[10px] text-text-muted">Try a different method:</span>
                              <div className="flex items-center gap-1.5 flex-wrap">
                                {suggestions.suggestions.map((s, idx) => (
                                  <button
                                    key={s.technique}
                                    onClick={() => handlePreviewTechnique(idx)}
                                    disabled={isPreviewing}
                                    className={`text-[10px] px-2.5 py-1 rounded-lg font-medium transition-all inline-flex items-center gap-1.5 ${
                                      idx === suggestionIdx
                                        ? 'bg-accent/15 text-accent-light ring-1 ring-accent/30'
                                        : 'bg-bg-primary text-text-muted hover:text-text-secondary hover:bg-bg-card-hover border border-border/50'
                                    }`}
                                  >
                                    {idx === 0 ? '★ ' : ''}{s.label}
                                    {s.expectedDelta != null && (
                                      <span className={`font-mono text-[9px] ${s.expectedDelta > 0 ? 'text-positive' : s.expectedDelta < 0 ? 'text-negative' : 'text-text-muted'}`}>
                                        {s.expectedDelta > 0 ? '+' : ''}{s.expectedDelta}
                                      </span>
                                    )}
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Preview loading */}
                          {isPreviewing && (
                            <div className="flex items-center justify-center py-6 gap-2">
                              <Loader2 size={16} className="text-accent animate-spin" />
                              <span className="text-xs text-text-muted">Running {currentSuggestion.label} on your data...</span>
                            </div>
                          )}

                          {/* ── DIFF VIEW: Before → After (like Cursor code review) ── */}
                          {preview && !isPreviewing && (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              className="space-y-3"
                            >
                              {/* Impact banner */}
                              <div className={`flex items-center justify-between px-4 py-2.5 rounded-lg ${
                                preview.delta > 0
                                  ? 'bg-positive/8 ring-1 ring-positive/20'
                                  : preview.delta < 0
                                    ? 'bg-negative/8 ring-1 ring-negative/20'
                                    : 'bg-bg-primary ring-1 ring-border/50'
                              }`}>
                                <div className="flex items-center gap-2">
                                  {preview.delta > 0 ? <TrendingUp size={14} className="text-positive" /> :
                                   preview.delta < 0 ? <TrendingDown size={14} className="text-negative" /> :
                                   <Minus size={14} className="text-text-muted" />}
                                  <span className="text-xs font-medium text-text-primary">
                                    {preview.delta > 0 ? 'This fix improves your data' :
                                     preview.delta < 0 ? 'This may make things slightly worse' :
                                     'No significant change detected'}
                                  </span>
                                </div>
                                <div className="flex items-center gap-3">
                                  <span className="text-xs text-text-muted font-mono">{preview.overallBefore}</span>
                                  <ArrowRight size={10} className="text-text-muted" />
                                  <span className={`text-sm font-bold font-mono ${
                                    preview.delta > 0 ? 'text-positive' : preview.delta < 0 ? 'text-negative' : 'text-text-primary'
                                  }`}>{preview.overallAfter}</span>
                                  <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                                    preview.delta > 0 ? 'bg-positive/15 text-positive' : preview.delta < 0 ? 'bg-negative/15 text-negative' : 'bg-bg-primary text-text-muted'
                                  }`}>
                                    {preview.delta > 0 ? '+' : ''}{preview.delta}
                                  </span>
                                </div>
                              </div>

                              {/* Diff table */}
                              <div className="rounded-lg border border-border overflow-hidden">
                                <table className="w-full">
                                  <thead>
                                    <tr className="bg-bg-secondary/50 text-[10px] text-text-muted uppercase tracking-wider">
                                      <th className="text-left px-4 py-2 font-semibold">Metric</th>
                                      <th className="text-right px-4 py-2 font-semibold">Before</th>
                                      <th className="text-right px-4 py-2 font-semibold">After</th>
                                      <th className="text-right px-4 py-2 font-semibold">Change</th>
                                    </tr>
                                  </thead>
                                  <tbody className="text-xs">
                                    {METRIC_KEYS.map((key) => {
                                      const bNorm = normValue(preview.before as FeatureMetrics, key);
                                      const aNorm = normValue(preview.after as FeatureMetrics, key);
                                      const delta = aNorm - bNorm;
                                      const improved = delta < -0.005;
                                      const worsened = delta > 0.005;
                                      const bRaw = rawValue(preview.before as FeatureMetrics, key);
                                      const aRaw = rawValue(preview.after as FeatureMetrics, key);
                                      return (
                                        <tr key={key} className={`border-t border-border/30 transition-colors ${
                                          improved ? 'bg-positive/[0.03]' : worsened ? 'bg-negative/[0.03]' : ''
                                        }`}>
                                          <td className="px-4 py-2.5 font-medium text-text-secondary">
                                            {METRIC_LABELS[key]}
                                          </td>
                                          <td className="px-4 py-2.5 text-right font-mono text-text-muted">{bRaw}</td>
                                          <td className={`px-4 py-2.5 text-right font-mono font-medium ${
                                            improved ? 'text-positive' : worsened ? 'text-negative' : 'text-text-primary'
                                          }`}>{aRaw}</td>
                                          <td className="px-4 py-2.5 text-right">
                                            <span className={`inline-flex items-center gap-1 font-mono text-[11px] font-semibold ${
                                              improved ? 'text-positive' : worsened ? 'text-negative' : 'text-text-muted'
                                            }`}>
                                              <DeltaArrow delta={delta} />
                                              {Math.abs(delta * 100).toFixed(1)}%
                                            </span>
                                          </td>
                                        </tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                              </div>

                              {/* ── Approve / Reject (like Cursor) ── */}
                              <div className="flex items-center gap-2 pt-1">
                                <button
                                  onClick={handleApprove}
                                  disabled={isCommitting}
                                  className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-positive/15 hover:bg-positive/25 text-positive text-sm font-semibold transition-all ring-1 ring-positive/20 hover:ring-positive/40 disabled:opacity-50"
                                >
                                  {isCommitting ? (
                                    <><Loader2 size={14} className="animate-spin" /> Applying...</>
                                  ) : (
                                    <><CheckCircle2 size={14} /> Accept</>
                                  )}
                                </button>
                                <button
                                  onClick={() => setPreview(null)}
                                  className="flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-bg-primary hover:bg-negative/8 text-text-muted hover:text-negative text-sm font-medium transition-all border border-border hover:border-negative/30"
                                >
                                  <XCircle size={14} />
                                  Reject
                                </button>
                                <button
                                  onClick={handleSkip}
                                  className="flex items-center gap-1.5 px-3 py-2.5 text-text-muted hover:text-text-secondary text-xs transition-colors ml-auto"
                                >
                                  <SkipForward size={12} />
                                  Skip
                                </button>
                              </div>
                            </motion.div>
                          )}
                        </Card>
                      </motion.div>
                    )}

                    {/* No suggestions */}
                    {suggestions && suggestions.suggestions.length === 0 && !isSuggesting && (
                      <Card className="text-center py-8">
                        <CheckCircle2 size={24} className="text-positive mx-auto mb-2 opacity-60" />
                        <p className="text-xs text-text-muted">
                          {activeFs.status === 'cleaned'
                            ? 'No additional cleaning methods available for this feature.'
                            : 'This feature looks good! No cleaning needed.'}
                        </p>
                        <button onClick={handleSkip} className="mt-3 text-xs text-accent hover:text-accent-light transition-colors">
                          Next feature →
                        </button>
                      </Card>
                    )}
                  </>
                )}

                {/* ── Cleaned feature info (when browsing, not just approved) ── */}
                {activeFs.status === 'cleaned' && !justApproved && !suggestions && !isSuggesting && (
                  <Card className="ring-1 ring-positive/15">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-positive/10 flex items-center justify-center ring-1 ring-positive/20">
                        <CheckCircle2 size={18} className="text-positive" />
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-semibold text-positive">Cleaning Applied</p>
                        <p className="text-xs text-text-muted mt-0.5">
                          {activeFs.cleaningHistory.length > 0
                            ? `${activeFs.cleaningHistory.length} technique${activeFs.cleaningHistory.length > 1 ? 's' : ''} applied`
                            : `Applied ${activeFs.committedTechniqueLabel ?? 'technique'}`
                          } — changes committed to training data.
                        </p>
                      </div>
                      <button
                        onClick={() => handleRevert(activeFs.feature)}
                        disabled={isReverting}
                        className="text-[10px] text-text-muted hover:text-negative transition-colors"
                      >
                        Undo
                      </button>
                    </div>
                  </Card>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Proceed to Stage 3 ── */}
      {(allProcessed || cleanedCount > 0) && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <Card className="ring-1 ring-accent/15 shadow-[0_0_24px_rgba(99,102,241,0.06)]">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-accent/10 flex items-center justify-center">
                  <Shield size={18} className="text-accent-light" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-text-primary">
                    {allProcessed ? 'All features reviewed' : `${cleanedCount} of ${featureStates.length} features cleaned`}
                  </p>
                  <p className="text-xs text-text-muted mt-0.5">
                    {allProcessed
                      ? 'Your cleaned training data is ready for the bias gate check.'
                      : 'You can proceed now or continue cleaning more features.'}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {cleaningSessionId && (
                  <a
                    href={`/api/clean/download/${cleaningSessionId}`}
                    download="cleaned_training_data.csv"
                    className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-bg-primary hover:bg-bg-card-hover text-text-secondary text-sm font-medium transition-all border border-border"
                  >
                    <Download size={14} />
                    Export CSV
                  </a>
                )}
                <button
                  onClick={handleProceed}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent hover:bg-accent-light text-white text-sm font-semibold transition-all shadow-[0_0_16px_rgba(99,102,241,0.2)] hover:shadow-[0_0_24px_rgba(99,102,241,0.3)]"
                >
                  Continue
                  <ArrowRight size={14} />
                </button>
              </div>
            </div>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
