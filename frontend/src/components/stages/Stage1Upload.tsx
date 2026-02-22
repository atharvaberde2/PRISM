import { useCallback, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  FileSpreadsheet,
  Target,
  ArrowRight,
  Hash,
  Type,
  ToggleLeft,
  AlertCircle,
  CheckCircle2,
  Columns,
  Check,
  CircleCheck,
  TriangleAlert,
  CircleAlert,
  Info,
} from 'lucide-react';
import type { ColumnProfile, FeatureMetrics } from '../../lib/types';
import Papa from 'papaparse';
import { usePipelineStore } from '../../store/pipeline';
import { profileColumnsAPI, computeBaselineBiasAPI } from '../../lib/mockEngine';
import Card from '../shared/Card';

const dtypeIcons: Record<string, typeof Hash> = {
  numeric: Hash,
  categorical: Type,
  boolean: ToggleLeft,
  text: Type,
  datetime: Hash,
};

const dtypeColors: Record<string, string> = {
  numeric: 'text-accent-light bg-accent/10',
  categorical: 'text-warning bg-warning-bg',
  boolean: 'text-positive bg-positive-bg',
  text: 'text-text-muted bg-bg-card',
  datetime: 'text-accent-light bg-accent/10',
};

/* ---------- Stat row helper ---------- */
function Stat({ label, value, mono = false }: { label: string; value: string | number; mono?: boolean }) {
  return (
    <div className="flex justify-between">
      <span className="text-text-muted">{label}</span>
      <span className={`text-text-secondary ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  );
}

/* ---------- Five-number box-plot indicator ---------- */
function BoxPlot({ min, q1, median, q3, max }: { min: number; q1: number; median: number; q3: number; max: number }) {
  const range = max - min || 1;
  const pct = (v: number) => ((v - min) / range) * 100;
  return (
    <div className="relative h-4 mt-1 mb-0.5">
      {/* whisker line */}
      <div
        className="absolute top-1/2 h-px bg-text-muted/40"
        style={{ left: `${pct(min)}%`, width: `${pct(max) - pct(min)}%` }}
      />
      {/* box */}
      <div
        className="absolute top-0.5 h-3 rounded-sm bg-accent/20 border border-accent/40"
        style={{ left: `${pct(q1)}%`, width: `${Math.max(pct(q3) - pct(q1), 1)}%` }}
      />
      {/* median line */}
      <div
        className="absolute top-0 h-4 w-0.5 bg-accent rounded-full"
        style={{ left: `${pct(median)}%` }}
      />
    </div>
  );
}

/* ---------- Frequency bar for categorical values ---------- */
function FreqBar({ value, percent, maxPercent }: { value: string; percent: number; maxPercent: number }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-text-secondary text-[10px] truncate w-16 text-right flex-shrink-0">{value}</span>
      <div className="flex-1 h-2 rounded-full bg-bg-primary overflow-hidden">
        <div
          className="h-full rounded-full bg-warning/60"
          style={{ width: `${(percent / maxPercent) * 100}%` }}
        />
      </div>
      <span className="text-text-muted text-[10px] w-10 text-right flex-shrink-0">{percent}%</span>
    </div>
  );
}

/* ---------- Column Card ---------- */
function ColumnCard({ col, selected, onToggle }: { col: ColumnProfile; selected?: boolean; onToggle?: () => void }) {
  const Icon = dtypeIcons[col.dtype] || Type;
  const selectable = selected !== undefined;

  const header = (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2 min-w-0">
        {selectable && (
          <div
            className={`w-5 h-5 rounded border flex items-center justify-center flex-shrink-0 transition-colors ${
              selected
                ? 'bg-accent border-accent text-white'
                : 'border-border bg-bg-primary text-transparent'
            }`}
          >
            <Check size={10} />
          </div>
        )}
        <div className={`w-6 h-6 rounded flex items-center justify-center flex-shrink-0 ${dtypeColors[col.dtype]}`}>
          <Icon size={12} />
        </div>
        <span className="text-sm font-medium text-text-primary truncate">{col.name}</span>
      </div>
      <span className="text-[10px] uppercase text-text-muted flex-shrink-0 ml-1">{col.dtype}</span>
    </div>
  );

  const nullBar = (
    <div className="h-1 rounded-full bg-bg-primary overflow-hidden">
      <div
        className={`h-full rounded-full transition-all ${
          col.nullPercent > 20 ? 'bg-negative' : col.nullPercent > 5 ? 'bg-warning' : 'bg-positive'
        }`}
        style={{ width: `${100 - col.nullPercent}%` }}
      />
    </div>
  );

  const cardClass = `space-y-2 ${selectable ? 'cursor-pointer' : ''} ${selectable && !selected ? 'opacity-40' : ''}`;

  /* ---- Numeric card ---- */
  if (col.dtype === 'numeric' && col.mean !== undefined) {
    return (
      <Card hover className={cardClass} onClick={onToggle}>
        {header}
        {/* Five-number summary */}
        <div className="grid grid-cols-5 text-center text-[10px] text-text-muted">
          <span>{col.min}</span><span>{col.q1}</span><span className="font-medium text-text-secondary">{col.median}</span><span>{col.q3}</span><span>{col.max}</span>
        </div>
        <div className="grid grid-cols-5 text-center text-[9px] text-text-muted/60">
          <span>min</span><span>Q1</span><span>med</span><span>Q3</span><span>max</span>
        </div>
        <BoxPlot min={col.min!} q1={col.q1!} median={col.median!} q3={col.q3!} max={col.max!} />
        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
          <Stat label="Mean" value={col.mean.toLocaleString()} mono />
          <Stat label="Std" value={col.std?.toLocaleString() ?? '—'} mono />
          <Stat label="Skew" value={col.skewness ?? '—'} mono />
          <Stat label="Nulls" value={`${col.nullPercent}%`} />
        </div>
        {nullBar}
      </Card>
    );
  }

  /* ---- Categorical card ---- */
  if (col.dtype === 'categorical' || col.dtype === 'text') {
    const topThree = col.topValues?.slice(0, 3) ?? [];
    const maxPct = topThree.length ? topThree[0].percent : 1;
    return (
      <Card hover className={cardClass} onClick={onToggle}>
        {header}
        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
          <Stat label="Mode" value={col.mode ?? '—'} />
          <Stat label="Cats" value={col.categoryCount ?? '—'} />
          <Stat label="Entropy" value={col.entropy ?? '—'} mono />
          <Stat label="Nulls" value={`${col.nullPercent}%`} />
        </div>
        {topThree.length > 0 && (
          <div className="space-y-1 pt-0.5">
            {topThree.map((tv) => (
              <FreqBar key={tv.value} value={tv.value} percent={tv.percent} maxPercent={maxPct} />
            ))}
          </div>
        )}
        {nullBar}
      </Card>
    );
  }

  /* ---- Boolean card ---- */
  if (col.dtype === 'boolean') {
    const truePct = col.truePercent ?? 0;
    const falsePct = Math.round((100 - truePct) * 10) / 10;
    return (
      <Card hover className={cardClass} onClick={onToggle}>
        {header}
        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
          <Stat label="True" value={`${col.trueCount?.toLocaleString()} (${truePct}%)`} />
          <Stat label="False" value={`${col.falseCount?.toLocaleString()} (${falsePct}%)`} />
          <Stat label="Nulls" value={`${col.nullPercent}%`} />
          <Stat label="Total" value={col.totalCount.toLocaleString()} />
        </div>
        {/* Split bar */}
        <div className="h-3 rounded-full bg-bg-primary overflow-hidden flex">
          <div className="h-full bg-positive/60 transition-all" style={{ width: `${truePct}%` }} />
          <div className="h-full bg-negative/40 transition-all" style={{ width: `${falsePct}%` }} />
        </div>
        <div className="flex justify-between text-[10px] text-text-muted">
          <span>True {truePct}%</span>
          <span>False {falsePct}%</span>
        </div>
      </Card>
    );
  }

  /* ---- Fallback (datetime / other) ---- */
  return (
    <Card hover className={cardClass} onClick={onToggle}>
      {header}
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
        <Stat label="Nulls" value={`${col.nullPercent}%`} />
        <Stat label="Unique" value={col.uniqueCount.toLocaleString()} />
      </div>
      {nullBar}
    </Card>
  );
}

export default function Stage1Upload() {
  const {
    columns,
    selectedColumns,
    targetColumn,
    baselineBias,
    rawData,
    setFileData,
    setSelectedColumns,
    setTargetColumn,
    setBaselineBias,
    setStage,
    unlockStage,
    setUploadedFile: storeSetUploadedFile,
  } = usePipelineStore();

  const selectedSet = new Set(selectedColumns);

  const toggleColumn = (name: string) => {
    // Never deselect the target column
    if (name === targetColumn && selectedSet.has(name)) return;
    const next = new Set(selectedSet);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setSelectedColumns([...next]);
  };

  const handleSetTarget = (col: string) => {
    setTargetColumn(col);
    if (col && !selectedSet.has(col)) {
      setSelectedColumns([...selectedColumns, col]);
    }
  };

  const selectAll = () => setSelectedColumns(columns.map((c) => c.name));
  const selectNone = () => {
    // Keep target column selected
    setSelectedColumns(targetColumn ? [targetColumn] : []);
  };

  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<'upload' | 'profile' | 'configure' | 'ready'>(
    columns.length ? (baselineBias ? 'ready' : 'configure') : 'upload',
  );

  const handleFile = useCallback(
    (file: File) => {
      if (!file.name.endsWith('.csv')) return;
      setIsProcessing(true);
      setUploadedFile(file);
      storeSetUploadedFile(file);

      // Profile via backend (falls back to JS automatically)
      const profilePromise = profileColumnsAPI(file);

      // Parse CSV in-browser so rawData is available for later stages
      const parsePromise = new Promise<Record<string, unknown>[]>((resolve, reject) => {
        Papa.parse(file, {
          header: true,
          skipEmptyLines: true,
          complete: (result) => resolve(result.data as Record<string, unknown>[]),
          error: (err: Error) => reject(err),
        });
      });

      Promise.all([profilePromise, parsePromise])
        .then(([cols, data]) => {
          setFileData(file.name, cols, data);
          setIsProcessing(false);
          setPhase('profile');

          // Auto-advance to configure after brief delay
          setTimeout(() => setPhase('configure'), 1500);
        })
        .catch(() => setIsProcessing(false));
    },
    [setFileData],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleAnalyze = async () => {
    if (!targetColumn || !uploadedFile) return;
    setIsAnalyzing(true);
    const report = await computeBaselineBiasAPI(uploadedFile, targetColumn);
    setBaselineBias(report);
    setIsAnalyzing(false);
    setPhase('ready');
  };

  const handleProceed = () => {
    unlockStage(2);
    setStage(2);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <AnimatePresence mode="wait">
        {/* UPLOAD ZONE */}
        {phase === 'upload' && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`
                relative border-2 border-dashed rounded-2xl p-16 text-center
                transition-all duration-300 cursor-pointer
                ${
                  isDragging
                    ? 'border-accent bg-accent-glow scale-[1.02]'
                    : 'border-border hover:border-border-light hover:bg-bg-card/50'
                }
              `}
              onClick={() =>
                document.getElementById('csv-input')?.click()
              }
            >
              <input
                id="csv-input"
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                }}
              />

              <motion.div
                animate={isDragging ? { scale: 1.1 } : { scale: 1 }}
                className="flex flex-col items-center gap-4"
              >
                {isProcessing ? (
                  <>
                    <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center">
                      <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                    </div>
                    <p className="text-text-secondary text-lg">
                      Profiling your data...
                    </p>
                  </>
                ) : (
                  <>
                    <div className="w-16 h-16 rounded-2xl bg-bg-card border border-border flex items-center justify-center">
                      <Upload
                        size={28}
                        className={isDragging ? 'text-accent' : 'text-text-muted'}
                      />
                    </div>
                    <div>
                      <p className="text-text-primary text-lg font-medium">
                        Drop your CSV here
                      </p>
                      <p className="text-text-muted text-sm mt-1">
                        or click to browse. PRISM will profile every column
                        automatically.
                      </p>
                    </div>
                  </>
                )}
              </motion.div>
            </div>
          </motion.div>
        )}

        {/* COLUMN PROFILES */}
        {(phase === 'profile' || phase === 'configure' || phase === 'ready') && (
          <motion.div
            key="profiles"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* File header */}
            <Card className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
                <FileSpreadsheet size={20} className="text-accent-light" />
              </div>
              <div>
                <p className="text-sm font-medium text-text-primary">
                  {usePipelineStore.getState().fileName}
                </p>
                <p className="text-xs text-text-muted">
                  {columns.length} columns &middot;{' '}
                  {columns[0]?.totalCount.toLocaleString()} rows
                </p>
              </div>
            </Card>

            {/* Column grid */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-text-secondary">
                  Column Profiles
                </h3>
                {(phase === 'configure' || phase === 'ready') && (
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-text-muted">
                      {selectedColumns.length}/{columns.length} selected
                    </span>
                    <button
                      onClick={selectAll}
                      className="text-xs text-accent hover:text-accent-light transition-colors"
                    >
                      All
                    </button>
                    <button
                      onClick={selectNone}
                      className="text-xs text-text-muted hover:text-text-secondary transition-colors"
                    >
                      None
                    </button>
                  </div>
                )}
              </div>
              {(phase === 'configure' || phase === 'ready') && (
                <p className="text-xs text-text-muted mb-3">
                  Click columns to include or exclude them from cleaning and training.
                </p>
              )}
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                {columns.map((col, i) => (
                  <motion.div
                    key={col.name}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                  >
                    <ColumnCard
                      col={col}
                      {...(phase === 'configure' || phase === 'ready'
                        ? { selected: selectedSet.has(col.name), onToggle: () => toggleColumn(col.name) }
                        : {})}
                    />
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Column selection */}
            {(phase === 'configure' || phase === 'ready') && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <Card glow className="space-y-5">
                  <h3 className="text-sm font-semibold text-text-primary">
                    Configure Pipeline
                  </h3>

                  {/* Selected columns summary */}
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 text-xs text-text-secondary font-medium">
                      <Columns size={14} className="text-accent-light" />
                      Columns for Cleaning &amp; Training ({selectedColumns.length} of {columns.length})
                    </label>
                    <div className="flex flex-wrap gap-1.5">
                      {columns.map((c) => {
                        const active = selectedSet.has(c.name);
                        return (
                          <button
                            key={c.name}
                            onClick={() => toggleColumn(c.name)}
                            className={`
                              px-2 py-1 rounded-md text-xs font-medium border transition-all
                              ${active
                                ? 'bg-accent/10 border-accent/40 text-accent-light'
                                : 'bg-bg-primary border-border text-text-muted opacity-50'}
                            `}
                          >
                            {c.name}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Target column */}
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 text-xs text-text-secondary font-medium">
                      <Target size={14} className="text-accent-light" />
                      Target Column (what to predict)
                    </label>
                    <select
                      value={targetColumn || ''}
                      onChange={(e) => handleSetTarget(e.target.value)}
                      className="w-full bg-bg-primary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus-visible:ring-1 focus-visible:ring-accent focus:border-accent"
                    >
                      <option value="">Select target...</option>
                      {columns.filter((c) => selectedSet.has(c.name)).map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name} ({c.dtype})
                        </option>
                      ))}
                    </select>
                  </div>

                  {targetColumn && !baselineBias && (
                    <motion.button
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      onClick={handleAnalyze}
                      disabled={isAnalyzing}
                      className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-light text-white text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      {isAnalyzing ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                          Analyzing…
                        </>
                      ) : (
                        <>
                          <AlertCircle size={16} />
                          Compute Baseline Bias Analysis
                        </>
                      )}
                    </motion.button>
                  )}
                </Card>
              </motion.div>
            )}

            {/* ============================================================ */}
            {/* DATASET SUMMARY — Feature-by-Feature Analysis                 */}
            {/* ============================================================ */}
            {phase === 'ready' && baselineBias && (() => {
              // Derived values
              const totalRows = columns[0]?.totalCount ?? 0;
              const totalNulls = columns.reduce((s, c) => s + c.nullCount, 0);
              const totalCells = totalRows * columns.length;
              const completeness = totalCells > 0 ? ((totalCells - totalNulls) / totalCells) * 100 : 100;
              const numericCount = columns.filter(c => c.dtype === 'numeric').length;
              const catCount = columns.filter(c => c.dtype === 'categorical').length;
              const boolCount = columns.filter(c => c.dtype === 'boolean').length;
              const textCount = columns.filter(c => c.dtype === 'text').length;

              const classCounts = baselineBias.class_counts ?? {};
              const classProbs = baselineBias.class_probabilities ?? {};
              const classes = Object.keys(classCounts);
              const imbalance = baselineBias.imbalance_ratio ?? 1;
              const fmMap = baselineBias.feature_metrics ?? {};

              // Feature list sorted alphabetically for consistent reading
              const featureEntries = Object.entries(fmMap).sort(([a], [b]) => a.localeCompare(b));

              // Per-class feature means for distribution bars
              const numericFeatureCols = columns
                .filter(c => c.dtype === 'numeric' && c.name !== targetColumn && selectedSet.has(c.name))
                .map(c => c.name);

              const classFeatureStats: Record<string, Record<string, { mean: number; missingPct: number }>> = {};
              if (rawData && rawData.length > 0) {
                for (const col of numericFeatureCols) {
                  classFeatureStats[col] = {};
                  for (const cls of classes) {
                    const rows = rawData.filter(r => String(r[targetColumn!]) === String(cls));
                    const nums = rows.map(r => Number(r[col])).filter(v => !isNaN(v));
                    const missing = rows.length - nums.length;
                    classFeatureStats[col][cls] = {
                      mean: nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0,
                      missingPct: rows.length > 0 ? (missing / rows.length) * 100 : 0,
                    };
                  }
                }
              }

              // Verdict helper: returns icon, label, color, and a one-line explanation
              const verdict = (norm: number): { icon: typeof CircleCheck; label: string; color: string; bg: string; barBg: string } => {
                if (norm < 0.15) return { icon: CircleCheck, label: 'Looks Good', color: 'text-positive', bg: 'bg-positive-bg', barBg: 'bg-positive/60' };
                if (norm < 0.4) return { icon: TriangleAlert, label: 'Worth Watching', color: 'text-warning', bg: 'bg-warning-bg', barBg: 'bg-warning/60' };
                return { icon: CircleAlert, label: 'Potential Concern', color: 'text-negative', bg: 'bg-negative-bg', barBg: 'bg-negative/60' };
              };

              // Plain-English metric descriptions
              const metricInfo = (key: string, fm: FeatureMetrics): { name: string; value: string; norm: number; whatItMeans: string; goodExplain: string; badExplain: string } => {
                switch (key) {
                  case 'jsd':
                    return {
                      name: 'Distribution Similarity',
                      value: fm.jsd.toFixed(4),
                      norm: fm.norm_jsd,
                      whatItMeans: 'Checks whether this feature\'s values are spread similarly across all groups. Think of it like comparing two histograms side by side.',
                      goodExplain: 'The groups have very similar distributions — the data looks consistent.',
                      badExplain: 'The groups have noticeably different distributions — one group\'s data looks quite different from another.',
                    };
                  case 'effect':
                    return {
                      name: fm.test === 'chi2' ? 'Association Strength (Cramér\'s V)' : 'Group Difference Size (Cohen\'s D)',
                      value: fm.effect_size.toFixed(3),
                      norm: fm.norm_effect,
                      whatItMeans: fm.test === 'chi2'
                        ? 'Measures how strongly this categorical feature is linked to the target. A high value means the categories cluster differently across groups.'
                        : 'Measures how far apart the average values are between groups, relative to the spread of the data. Like asking "how many standard deviations apart are the group averages?"',
                      goodExplain: fm.test === 'chi2'
                        ? 'Weak association — the categories are spread fairly evenly across groups.'
                        : 'Small difference — the group averages are close together.',
                      badExplain: fm.test === 'chi2'
                        ? 'Strong association — certain categories heavily favor specific groups, which could introduce bias.'
                        : 'Large difference — the group averages are far apart, which could skew predictions.',
                    };
                  case 'skew':
                    return {
                      name: 'Shape Difference',
                      value: fm.skew_diff.toFixed(2),
                      norm: fm.norm_skew,
                      whatItMeans: 'Compares the "shape" of data across groups. If one group\'s data is bunched to the left while another is bunched to the right, that\'s a shape difference.',
                      goodExplain: 'The data has a similar shape in every group — no lopsided distributions.',
                      badExplain: 'The data is shaped very differently between groups — one group may have outliers or skewed values that the others don\'t.',
                    };
                  case 'miss':
                    return {
                      name: 'Missing Data Gap',
                      value: `${(fm.missingness_gap * 100).toFixed(1)}%`,
                      norm: fm.norm_miss,
                      whatItMeans: 'Checks whether missing values are spread evenly across groups, or if one group has significantly more gaps than another.',
                      goodExplain: 'Missing values are spread fairly evenly — no group is disproportionately affected.',
                      badExplain: 'One group has significantly more missing data than another — this could mean the model learns less about that group.',
                    };
                  case 'pval':
                    return {
                      name: 'P-Value (Significance)',
                      value: fm.p_value < 0.001 ? 'p < 0.001' : `p = ${fm.p_value.toFixed(4)}`,
                      norm: fm.norm_pval,
                      whatItMeans: 'Tests whether the observed difference between groups could have happened by chance. A low p-value means the difference is statistically significant.',
                      goodExplain: 'The difference is not statistically significant — likely due to random variation.',
                      badExplain: 'The difference between groups is statistically significant — unlikely to be random noise.',
                    };
                  default:
                    return { name: key, value: '—', norm: 0, whatItMeans: '', goodExplain: '', badExplain: '' };
                }
              };

              // Class colour palette
              const classColors = ['bg-accent/60', 'bg-warning/60', 'bg-positive/60', 'bg-negative/60', 'bg-purple-400/60'];
              const classTextColors = ['text-accent-light', 'text-warning', 'text-positive', 'text-negative', 'text-purple-400'];

              return (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="space-y-4"
                >
                  {/* ──────────────── Dataset Overview ──────────────── */}
                  <Card className="space-y-3">
                    <h3 className="text-sm font-semibold text-text-primary">Dataset Summary</h3>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="p-3 rounded-lg bg-bg-primary text-center">
                        <p className="text-2xl font-bold font-mono text-text-primary">{totalRows.toLocaleString()}</p>
                        <p className="text-[10px] text-text-muted uppercase tracking-wider">Rows</p>
                      </div>
                      <div className="p-3 rounded-lg bg-bg-primary text-center">
                        <p className="text-2xl font-bold font-mono text-text-primary">{selectedColumns.length}</p>
                        <p className="text-[10px] text-text-muted uppercase tracking-wider">Selected Columns</p>
                      </div>
                      <div className="p-3 rounded-lg bg-bg-primary text-center">
                        <p className={`text-2xl font-bold font-mono ${completeness >= 95 ? 'text-positive' : completeness >= 80 ? 'text-warning' : 'text-negative'}`}>
                          {completeness.toFixed(1)}%
                        </p>
                        <p className="text-[10px] text-text-muted uppercase tracking-wider">Completeness</p>
                      </div>
                      <div className="p-3 rounded-lg bg-bg-primary text-center">
                        <div className="flex flex-wrap justify-center gap-1 text-[10px]">
                          {numericCount > 0 && <span className="px-1.5 py-0.5 rounded bg-accent/10 text-accent-light">{numericCount} num</span>}
                          {catCount > 0 && <span className="px-1.5 py-0.5 rounded bg-warning-bg text-warning">{catCount} cat</span>}
                          {boolCount > 0 && <span className="px-1.5 py-0.5 rounded bg-positive-bg text-positive">{boolCount} bool</span>}
                          {textCount > 0 && <span className="px-1.5 py-0.5 rounded bg-bg-card text-text-muted">{textCount} text</span>}
                        </div>
                        <p className="text-[10px] text-text-muted uppercase tracking-wider mt-1">Types</p>
                      </div>
                    </div>

                    {/* Imbalance callout */}
                    <div className="flex items-center gap-2 p-2.5 rounded-lg bg-bg-primary text-xs">
                      {imbalance <= 1.5 ? <CircleCheck size={14} className="text-positive flex-shrink-0" /> :
                       imbalance <= 3 ? <TriangleAlert size={14} className="text-warning flex-shrink-0" /> :
                       <CircleAlert size={14} className="text-negative flex-shrink-0" />}
                      <div>
                        <span className="text-text-secondary font-medium">Class Balance: </span>
                        <span className={`font-mono font-bold ${
                          imbalance <= 1.5 ? 'text-positive' : imbalance <= 3 ? 'text-warning' : 'text-negative'
                        }`}>
                          {imbalance === Infinity ? '∞' : `${imbalance}:1`} ratio
                        </span>
                        <span className="text-text-muted ml-1.5">
                          {imbalance <= 1.5
                            ? '— Groups are roughly equal in size. This is ideal for fair training.'
                            : imbalance <= 3
                            ? '— Some groups are larger than others. The model may learn more about the bigger group.'
                            : '— One group is much larger than the rest. The model could become biased toward the majority group.'}
                        </span>
                      </div>
                    </div>
                  </Card>

                  {/* ──────────────── Target Distribution ──────────────── */}
                  <Card className="space-y-3">
                    <h3 className="text-sm font-semibold text-text-primary">Target Distribution — "{targetColumn}"</h3>
                    <p className="text-xs text-text-muted">
                      This is the column the model will learn to predict. Below shows how many rows belong to each group.
                    </p>
                    <div className="space-y-1.5">
                      {classes.map((cls, i) => {
                        const prob = classProbs[cls] ?? 0;
                        return (
                          <div key={cls} className="flex items-center gap-3">
                            <span className="text-xs font-mono text-text-secondary w-16 text-right flex-shrink-0 truncate">{cls}</span>
                            <div className="flex-1 h-6 rounded bg-bg-primary overflow-hidden">
                              <div
                                className={`h-full rounded ${classColors[i % classColors.length]} flex items-center pl-2`}
                                style={{ width: `${Math.max(prob * 100, 3)}%` }}
                              >
                                <span className="text-[11px] font-mono text-white/90 whitespace-nowrap">
                                  {classCounts[cls]?.toLocaleString()} rows ({(prob * 100).toFixed(1)}%)
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    {/* Legend */}
                    <div className="flex flex-wrap gap-3 pt-1">
                      {classes.map((cls, i) => (
                        <span key={cls} className={`flex items-center gap-1.5 text-xs ${classTextColors[i % classTextColors.length]}`}>
                          <span className={`w-3 h-3 rounded-sm ${classColors[i % classColors.length]}`} />
                          {cls}
                        </span>
                      ))}
                    </div>
                  </Card>

                  {/* ──────────────── How to Read This Section ──────────────── */}
                  <Card className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Info size={14} className="text-accent-light flex-shrink-0" />
                      <h3 className="text-sm font-semibold text-text-primary">How to Read the Feature Analysis Below</h3>
                    </div>
                    <p className="text-xs text-text-muted leading-relaxed">
                      For each feature (column) in your dataset, we run <strong className="text-text-secondary">5 checks</strong> to see if the data behaves
                      differently across your target groups. This helps spot potential unfairness <em>before</em> training.
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
                      <div className="flex items-center gap-2 p-2 rounded-lg bg-positive-bg/50">
                        <CircleCheck size={14} className="text-positive flex-shrink-0" />
                        <span className="text-text-secondary"><strong className="text-positive">Looks Good</strong> — No notable differences between groups.</span>
                      </div>
                      <div className="flex items-center gap-2 p-2 rounded-lg bg-warning-bg/50">
                        <TriangleAlert size={14} className="text-warning flex-shrink-0" />
                        <span className="text-text-secondary"><strong className="text-warning">Worth Watching</strong> — Some differences; may or may not be a problem.</span>
                      </div>
                      <div className="flex items-center gap-2 p-2 rounded-lg bg-negative-bg/50">
                        <CircleAlert size={14} className="text-negative flex-shrink-0" />
                        <span className="text-text-secondary"><strong className="text-negative">Potential Concern</strong> — Large differences that could affect fairness.</span>
                      </div>
                    </div>
                  </Card>

                  {/* ──────────────── Feature-by-Feature Analysis ──────────────── */}
                  {featureEntries.map(([col, fm], featureIdx) => {
                    const stats = classFeatureStats[col];
                    const maxMean = stats ? Math.max(...Object.values(stats).map(s => Math.abs(s.mean)), 1) : 0;

                    const metrics = [
                      metricInfo('jsd', fm),
                      metricInfo('effect', fm),
                      metricInfo('skew', fm),
                      metricInfo('miss', fm),
                      metricInfo('pval', fm),
                    ];

                    return (
                      <motion.div
                        key={col}
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 + featureIdx * 0.05 }}
                      >
                        <Card className="space-y-4">
                          {/* Feature header */}
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center">
                              {fm.dtype === 'numeric'
                                ? <Hash size={16} className="text-accent-light" />
                                : <Type size={16} className="text-warning" />}
                            </div>
                            <div>
                              <h3 className="text-sm font-semibold text-text-primary">{col}</h3>
                              <p className="text-[11px] text-text-muted">
                                {fm.dtype === 'numeric' ? 'Numeric column' : 'Categorical column'} — analyzed against "{targetColumn}"
                              </p>
                            </div>
                          </div>

                          {/* Distribution by group */}
                          {stats && (
                            <div className="space-y-1.5">
                              <p className="text-xs font-medium text-text-secondary">Average value by group</p>
                              <p className="text-[11px] text-text-muted">
                                If the bars below are roughly the same length, this feature behaves similarly across groups. Big differences may signal a disparity.
                              </p>
                              <div className="space-y-1 pt-1">
                                {classes.map((cls, i) => {
                                  const s = stats[cls];
                                  if (!s) return null;
                                  const barW = maxMean > 0 ? (Math.abs(s.mean) / maxMean) * 100 : 0;
                                  return (
                                    <div key={cls} className="flex items-center gap-2">
                                      <span className="text-[11px] font-mono text-text-muted w-16 text-right flex-shrink-0 truncate">{cls}</span>
                                      <div className="flex-1 h-4 rounded bg-bg-primary overflow-hidden">
                                        <div
                                          className={`h-full rounded ${classColors[i % classColors.length]} flex items-center pl-1.5`}
                                          style={{ width: `${barW}%`, minWidth: barW > 0 ? '2.5rem' : 0 }}
                                        >
                                          <span className="text-[10px] font-mono text-white/90 whitespace-nowrap">
                                            {s.mean >= 1000 ? `${(s.mean / 1000).toFixed(1)}k` : s.mean.toFixed(1)}
                                          </span>
                                        </div>
                                      </div>
                                      {s.missingPct > 0 && (
                                        <span className="text-[10px] text-negative flex-shrink-0">
                                          {s.missingPct.toFixed(1)}% missing
                                        </span>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}

                          {/* 5 metrics */}
                          <div className="space-y-3">
                            <p className="text-xs font-medium text-text-secondary">5 Fairness Checks</p>
                            {metrics.map((m) => {
                              const v = verdict(m.norm);
                              const VIcon = v.icon;
                              return (
                                <div key={m.name} className="p-3 rounded-lg bg-bg-primary space-y-2">
                                  {/* Metric header row */}
                                  <div className="flex items-center justify-between gap-2">
                                    <div className="flex items-center gap-2 min-w-0">
                                      <VIcon size={14} className={`${v.color} flex-shrink-0`} />
                                      <span className="text-xs font-medium text-text-primary">{m.name}</span>
                                    </div>
                                    <div className="flex items-center gap-2 flex-shrink-0">
                                      <span className="text-xs font-mono text-text-secondary">{m.value}</span>
                                      <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${v.color} ${v.bg}`}>
                                        {v.label}
                                      </span>
                                    </div>
                                  </div>

                                  {/* Progress bar */}
                                  <div className="h-2 rounded-full bg-bg-secondary overflow-hidden">
                                    <div
                                      className={`h-full rounded-full transition-all ${v.barBg}`}
                                      style={{ width: `${Math.max(m.norm * 100, 2)}%` }}
                                    />
                                  </div>

                                  {/* Plain English explanation */}
                                  <p className="text-[11px] text-text-muted leading-relaxed">
                                    <strong className="text-text-secondary">What this checks: </strong>{m.whatItMeans}
                                  </p>
                                  <p className={`text-[11px] leading-relaxed ${v.color}`}>
                                    <strong>Result: </strong>{m.norm < 0.15 ? m.goodExplain : m.norm < 0.4 ? m.badExplain : m.badExplain}
                                  </p>
                                </div>
                              );
                            })}
                          </div>
                        </Card>
                      </motion.div>
                    );
                  })}

                  {/* Proceed button */}
                  <Card>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs text-text-muted">
                        <CheckCircle2 size={14} className="text-accent" />
                        Analysis complete. Ready to begin bias-aware cleaning.
                      </div>
                      <button
                        onClick={handleProceed}
                        className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-light text-white text-sm font-medium transition-colors"
                      >
                        Proceed to Cleaning
                        <ArrowRight size={16} />
                      </button>
                    </div>
                  </Card>
                </motion.div>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
