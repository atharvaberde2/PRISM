import { motion } from 'framer-motion';
import {
  BarChart3,
  Users,
  CheckCircle2,
  AlertTriangle,
  Award,
} from 'lucide-react';
import { usePipelineStore } from '../../store/pipeline';
import Card from '../shared/Card';

export default function Stage5Explain() {
  const { explainResult, models } =
    usePipelineStore();

  if (!explainResult) return null;

  const winner = models.find((m) => m.isWinner);
  const maxImportance = Math.max(
    ...explainResult.shapFeatures.map((f) => Math.abs(f.importance)),
  );

  // Compute actual accuracy gap from real data
  const accs = explainResult.subgroupPerformance.map((s) => s.accuracy);
  const accGap = accs.length >= 2 ? Math.max(...accs) - Math.min(...accs) : 0;
  const isEquitable = accGap < 0.10;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <Card className="text-center py-6">
        <Award size={32} className="text-accent mx-auto" />
        <h2 className="text-lg font-semibold text-text-primary mt-3">
          Model Explanation & Certification
        </h2>
        <p className="text-xs text-text-muted mt-1">
          {winner?.name} — AUC-ROC:{' '}
          {((winner?.metrics.aucRoc ?? 0) * 100).toFixed(1)}%
        </p>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* SHAP Feature Importance */}
        <Card className="space-y-4">
          <div className="flex items-center gap-2">
            <BarChart3 size={16} className="text-accent-light" />
            <h3 className="text-sm font-semibold text-text-primary">
              Feature Importance (Permutation)
            </h3>
          </div>

          <div className="space-y-2">
            {explainResult.shapFeatures.map((feature, idx) => {
              const width = (Math.abs(feature.importance) / maxImportance) * 100;
              return (
                <motion.div
                  key={feature.feature}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.08 }}
                  className="flex items-center gap-3"
                >
                  <span className="text-xs text-text-secondary w-32 truncate text-right">
                    {feature.feature}
                  </span>
                  <div className="flex-1 h-6 bg-bg-primary rounded overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${width}%` }}
                      transition={{ delay: 0.3 + idx * 0.08, duration: 0.6 }}
                      className={`h-full rounded ${
                        feature.direction === 'positive'
                          ? 'bg-accent/60'
                          : 'bg-negative/40'
                      }`}
                    />
                  </div>
                  <span className="text-xs font-mono text-text-muted w-10">
                    {(feature.importance * 100).toFixed(0)}%
                  </span>
                </motion.div>
              );
            })}
          </div>

          <div className="flex items-center gap-4 pt-2 text-[10px] text-text-muted">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-sm bg-accent/60" /> Positive impact
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-sm bg-negative/40" /> Negative impact
            </span>
          </div>
        </Card>

        {/* Subgroup Performance */}
        <Card className="space-y-4">
          <div className="flex items-center gap-2">
            <Users size={16} className="text-warning" />
            <h3 className="text-sm font-semibold text-text-primary">
              Subgroup Performance (Per Class)
            </h3>
          </div>

          <div className="space-y-3">
            {explainResult.subgroupPerformance.map((group, idx) => {
              const accuracyDiff = Math.abs(
                group.accuracy -
                  explainResult.subgroupPerformance[0].accuracy,
              );
              return (
                <motion.div
                  key={group.group}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + idx * 0.15 }}
                  className="p-3 rounded-lg bg-bg-primary"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-text-primary">
                      {group.group}
                    </span>
                    <span className="text-[10px] text-text-muted">
                      n={group.count.toLocaleString()} ({group.percentOfTotal}%)
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-[10px] text-text-muted uppercase">
                        Accuracy
                      </p>
                      <p className="text-lg font-bold font-mono text-text-primary">
                        {(group.accuracy * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-[10px] text-text-muted uppercase">
                        F1 Score
                      </p>
                      <p className="text-lg font-bold font-mono text-text-primary">
                        {(group.f1 * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  {idx > 0 && accuracyDiff > 0.001 && (
                    <div
                      className={`mt-2 text-[10px] font-medium ${
                        accuracyDiff < 0.05
                          ? 'text-positive'
                          : accuracyDiff < 0.10
                            ? 'text-warning'
                            : 'text-negative'
                      }`}
                    >
                      {accuracyDiff < 0.05
                        ? `Within 5% of reference group`
                        : accuracyDiff < 0.10
                          ? `${(accuracyDiff * 100).toFixed(1)}% gap — moderate disparity`
                          : `${(accuracyDiff * 100).toFixed(1)}% gap — significant disparity`}
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>

          {/* Equity check — computed from real data */}
          <div className={`p-3 rounded-lg border ${
            isEquitable
              ? 'bg-positive-bg border-positive/20'
              : 'bg-warning-bg border-warning/20'
          }`}>
            <div className="flex items-center gap-2">
              {isEquitable ? (
                <CheckCircle2 size={14} className="text-positive" />
              ) : (
                <AlertTriangle size={14} className="text-warning" />
              )}
              <span className={`text-xs font-medium ${isEquitable ? 'text-positive' : 'text-warning'}`}>
                {isEquitable
                  ? `Performance is consistent across subgroups (${(accGap * 100).toFixed(1)}% max accuracy gap)`
                  : `Performance gap of ${(accGap * 100).toFixed(1)}% detected across subgroups`}
              </span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
