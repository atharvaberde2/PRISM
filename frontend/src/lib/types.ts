export interface ColumnProfile {
  name: string;
  dtype: 'numeric' | 'categorical' | 'boolean' | 'datetime' | 'text';
  totalCount: number;
  nullCount: number;
  nullPercent: number;
  uniqueCount: number;
  uniquePercent: number;
  // Numeric stats
  mean?: number;
  median?: number;
  std?: number;
  min?: number;
  max?: number;
  q1?: number;
  q3?: number;
  iqr?: number;
  skewness?: number;
  p5?: number;
  p95?: number;

  // Categorical stats
  topValues?: { value: string; count: number; percent: number }[];
  mode?: string;
  modeCount?: number;
  modePercent?: number;
  categoryCount?: number;
  entropy?: number;

  // Boolean stats
  trueCount?: number;
  falseCount?: number;
  truePercent?: number;
}

export interface BiasMetric {
  name: string;
  label: string;
  score: number;        // 0–100
  description: string;
}

export interface FeatureJSD {
  mean_jsd: number;
  pairwise: Record<string, number>;
}

export interface MissingnessRecord {
  class: string;
  feature: string;
  missing_count: number;
  missing_rate: number;
}

/** Per-feature bias metrics returned by /api/bias/baseline */
export interface FeatureMetrics {
  dtype: 'numeric' | 'categorical';
  jsd: number;
  test: 'ttest' | 'anova' | 'chi2';
  test_stat: number;
  p_value: number;
  effect_size: number;
  effect_label: 'cohens_d' | 'cramers_v';
  skew_by_class: Record<string, number>;
  kurt_by_class: Record<string, number>;
  skew_diff: number;
  kurt_diff: number;
  missingness_by_class: Record<string, number>;
  missingness_gap: number;
  norm_jsd: number;
  norm_effect: number;
  norm_skew: number;
  norm_miss: number;
  norm_pval: number;
  feature_bias: number;   // 0–1, higher = more biased
}

export interface SubScores {
  js_divergence: number;
  chi2_effect: number;
  skew_diff: number;
  missingness_gap: number;
  p_value: number;
}

export interface BiasReport {
  overallScore: number; // 0–100  (higher = less bias)
  metrics?: BiasMetric[];
  class_counts?: Record<string, number>;
  class_probabilities?: Record<string, number>;
  imbalance_ratio?: number;
  feature_metrics?: Record<string, FeatureMetrics>;
  sub_scores?: SubScores;
  missingness_by_class?: MissingnessRecord[];
  // Deprecated (kept for backward compat)
  feature_js_divergence?: Record<string, FeatureJSD>;
}

export interface CleaningStep {
  id: string;
  column: string;
  type: 'imputation' | 'outlier' | 'duplicate' | 'encoding' | 'rebalancing' | 'feature_cleaning' | 'missingness_indicator';
  strategy: string;
  description: string;
  qualityImpact: number;   // 0–100
  confidence: number;      // 0–100
  biasDelta: number;       // negative = worse, positive = better
  afterScore?: number;     // bias score after this technique
  afterSubScores?: SubScores;
  rowsAffected?: number;
  alternative?: {
    strategy: string;
    description: string;
    biasDelta: number;
    afterScore?: number;
    afterSubScores?: SubScores;
  };
  status: 'pending' | 'approved' | 'rejected' | 'alternative_selected';
}

export interface GateResult {
  passed: boolean;
  dimensions: {
    name: string;
    label: string;
    before: number;
    after: number;
    delta: number;
    passed: boolean;
  }[];
  overriddenBy?: string;
  overrideJustification?: string;
}

export interface ModelResult {
  name: string;
  type: 'extra_trees' | 'random_forest' | 'gradient_boosted';
  metrics: Record<string, number>;
  bestParams?: Record<string, string | number | boolean | null>;
  cvScore?: number;
  trainingProgress: number; // 0–100
  isWinner: boolean;
}

export interface ShapFeature {
  feature: string;
  importance: number;
  direction: 'positive' | 'negative';
}

export interface SubgroupPerformance {
  group: string;
  count: number;
  percentOfTotal: number;
  // Classification fields
  accuracy?: number;
  f1?: number;
  // Regression fields
  mae?: number;
  r2?: number;
}

export interface ExplainResult {
  shapFeatures: ShapFeature[];
  subgroupPerformance: SubgroupPerformance[];
  complianceReady: boolean;
}

// ---------------------------------------------------------------------------
// Stage 2 – Iterative Feature-by-Feature Cleaning
// ---------------------------------------------------------------------------

export interface TechniqueSuggestion {
  technique: string;
  label: string;
  category: string;
  description: string;
  isHero: boolean;
  targetMetric: string;
  targetMetricLabel: string;
  expectedDelta?: number;
}

export interface CommittedStep {
  technique: string;
  label: string;
}

export interface FeatureCleanState {
  feature: string;
  dtype: 'numeric' | 'categorical';
  metrics: FeatureMetrics;
  status: 'uncleaned' | 'previewing' | 'cleaned';
  committedTechnique?: string;
  committedTechniqueLabel?: string;
  cleaningHistory: CommittedStep[];
}

export interface CleaningPreview {
  feature: string;
  technique: string;
  before: FeatureMetrics;
  after: FeatureMetrics;
  overallBefore: number;
  overallAfter: number;
  delta: number;
}

export interface SuggestResult {
  feature: string;
  dtype: string;
  worstMetric: string;
  worstMetricValue: number;
  currentMetrics: FeatureMetrics;
  norms: Record<string, number>;
  suggestions: TechniqueSuggestion[];
}

export type Stage = 1 | 2 | 3 | 4 | 5;

export interface SplitInfo {
  trainSize: number;
  testSize: number;
  splitRatio: string;
  stratified: boolean;
  targetType: string;
  nClasses?: number;
  cvFolds: number;
}

export interface PipelineState {
  currentStage: Stage;
  maxUnlockedStage: Stage;

  // Stage 1
  fileName: string | null;
  columns: ColumnProfile[];
  selectedColumns: string[];
  targetColumn: string | null;
  sensitiveColumn: string | null;
  baselineBias: BiasReport | null;
  rawData: Record<string, unknown>[] | null;
  uploadedFile: File | null;

  // Stage 2 (legacy — kept for backward compat)
  cleaningSteps: CleaningStep[];
  currentBiasScore: number;
  isGeneratingSteps: boolean;

  // Stage 2 – iterative session-based cleaning
  cleaningSessionId: string | null;
  featureStates: FeatureCleanState[];
  activeFeature: string | null;

  // Stage 3
  gateResult: GateResult | null;

  // Stage 4
  models: ModelResult[];
  isTraining: boolean;
  splitInfo: SplitInfo | null;

  // Stage 5
  explainResult: ExplainResult | null;
}
