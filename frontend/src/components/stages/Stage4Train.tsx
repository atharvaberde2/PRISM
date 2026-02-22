import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  ArrowRight,
  Zap,
  TreeDeciduous,
  TrendingUp,
  Loader2,
  AlertTriangle,
  SplitSquareVertical,
  Settings2,
  CheckCircle2,
  CircleDot,
} from 'lucide-react';
import { usePipelineStore } from '../../store/pipeline';
import Card from '../shared/Card';
import ProgressBar from '../shared/ProgressBar';
import { trainModelsAPI, explainModelAPI, simulateTraining } from '../../lib/mockEngine';
import type { ModelResult, SplitInfo } from '../../lib/types';

const MODEL_OPTIONS = [
  {
    name: 'Extra Trees',
    type: 'extra_trees',
    icon: TrendingUp,
    color: 'text-accent-light bg-accent/10',
    ring: 'ring-accent/40',
    description: 'Extremely randomized trees. Fast, robust, and reduces variance.',
  },
  {
    name: 'Random Forest',
    type: 'random_forest',
    icon: TreeDeciduous,
    color: 'text-positive bg-positive-bg',
    ring: 'ring-positive/40',
    description: 'Ensemble of decision trees. Robust to outliers and non-linear patterns.',
  },
  {
    name: 'Gradient Boosted Trees',
    type: 'gradient_boosted',
    icon: Zap,
    color: 'text-warning bg-warning-bg',
    ring: 'ring-warning/40',
    description: 'Sequential boosting for high accuracy. Often wins competitions.',
  },
] as const;

type ModelType = (typeof MODEL_OPTIONS)[number]['type'];

interface TrainedResult {
  model: ModelResult;
  splitInfo: SplitInfo | null;
}

export default function Stage4Train() {
  const {
    setModels,
    updateModel,
    setIsTraining,
    setSplitInfo,
    setStage,
    unlockStage,
    setExplainResult,
    cleaningSessionId,
  } = usePipelineStore();

  // --- Multi-train state ---
  const [trainedResults, setTrainedResults] = useState<Record<string, TrainedResult>>({});
  const [selectedModelType, setSelectedModelType] = useState<string | null>(null);
  const [trainingType, setTrainingType] = useState<string | null>(null);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [chosenForExplain, setChosenForExplain] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [latestSplitInfo, setLatestSplitInfo] = useState<SplitInfo | null>(null);

  const trainedCount = Object.keys(trainedResults).length;
  const isIdle = trainingType === null && !explaining;

  const handleStartTraining = useCallback(async () => {
    if (!selectedModelType || trainingType) return;

    const selected = MODEL_OPTIONS.find((m) => m.type === selectedModelType)!;
    setTrainingType(selectedModelType);
    setTrainingProgress(0);
    setError(null);

    // Set up a temporary model in the store so ProgressBar can render
    setModels([
      {
        name: selected.name,
        type: selected.type as ModelResult['type'],
        metrics: {},
        trainingProgress: 0,
        isWinner: false,
      },
    ]);
    setIsTraining(true);

    if (cleaningSessionId) {
      try {
        const { models: results, splitInfo: info } = await trainModelsAPI(cleaningSessionId, selectedModelType);
        const result = results[0];
        if (result) {
          const withWinner = { ...result, isWinner: true, trainingProgress: 100 };
          setTrainedResults((prev) => ({
            ...prev,
            [selectedModelType]: { model: withWinner, splitInfo: info ?? null },
          }));
          setModels([withWinner]);
          if (info) {
            setSplitInfo(info);
            setLatestSplitInfo(info);
          }
          // Check for per-model errors
          if ((result as Record<string, unknown>).error) {
            setError(String((result as Record<string, unknown>).error));
          }
        }
        setTrainingProgress(100);
      } catch (e) {
        setError(String(e));
        // Fallback to simulation
        simulateTraining(
          selectedModelType,
          (name, progress) => {
            setTrainingProgress(progress);
            updateModel(name, { trainingProgress: progress });
          },
          (finalModels) => {
            const result = finalModels[0];
            if (result) {
              setTrainedResults((prev) => ({
                ...prev,
                [selectedModelType]: { model: result, splitInfo: null },
              }));
              setModels(finalModels);
            }
            setTrainingProgress(100);
            setTrainingType(null);
            setIsTraining(false);
          },
        );
        return; // simulateTraining handles cleanup via callback
      }
    } else {
      // No session — simulate
      await new Promise<void>((resolve) => {
        simulateTraining(
          selectedModelType,
          (name, progress) => {
            setTrainingProgress(progress);
            updateModel(name, { trainingProgress: progress });
          },
          (finalModels) => {
            const result = finalModels[0];
            if (result) {
              setTrainedResults((prev) => ({
                ...prev,
                [selectedModelType]: { model: result, splitInfo: null },
              }));
              setModels(finalModels);
            }
            setTrainingProgress(100);
            resolve();
          },
        );
      });
    }

    setTrainingType(null);
    setIsTraining(false);
    setSelectedModelType(null);
  }, [selectedModelType, trainingType, cleaningSessionId, setModels, updateModel, setIsTraining, setSplitInfo]);

  const handleProceed = useCallback(async () => {
    if (!chosenForExplain || !cleaningSessionId) return;
    setExplaining(true);
    setError(null);

    try {
      // Re-train the chosen model so the backend session has the right _trained_model
      await trainModelsAPI(cleaningSessionId, chosenForExplain);
      const result = await explainModelAPI(cleaningSessionId);
      // Also update the store models to the chosen one
      const chosen = trainedResults[chosenForExplain];
      if (chosen) {
        setModels([chosen.model]);
        if (chosen.splitInfo) setSplitInfo(chosen.splitInfo);
      }
      setExplainResult(result);
      unlockStage(5);
      setStage(5);
    } catch (e) {
      setError(`Explain failed: ${e}`);
    } finally {
      setExplaining(false);
    }
  }, [chosenForExplain, cleaningSessionId, trainedResults, setModels, setSplitInfo, setExplainResult, unlockStage, setStage]);

  // --- Derived ---
  const chosenModel = chosenForExplain ? trainedResults[chosenForExplain]?.model : null;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <Card className="text-center py-6">
        <motion.div
          animate={trainingType ? { rotate: 360 } : { rotate: 0 }}
          transition={trainingType ? { duration: 2, repeat: Infinity, ease: 'linear' } : {}}
          className="inline-flex"
        >
          {trainingType ? (
            <Loader2 size={32} className="text-accent" />
          ) : explaining ? (
            <Loader2 size={32} className="text-accent animate-spin" />
          ) : trainedCount > 0 ? (
            <CheckCircle2 size={32} className="text-positive" />
          ) : (
            <Brain size={32} className="text-text-muted" />
          )}
        </motion.div>
        <h2 className="text-lg font-semibold text-text-primary mt-3">
          {trainingType
            ? `Training ${MODEL_OPTIONS.find((m) => m.type === trainingType)?.name}...`
            : explaining
              ? 'Preparing model for explanation...'
              : trainedCount > 0 && chosenForExplain
                ? 'Ready to Explain & Certify'
                : trainedCount > 0
                  ? `${trainedCount} model${trainedCount > 1 ? 's' : ''} trained — select one or train another`
                  : 'Select a Model to Train'}
        </h2>
        <p className="text-xs text-text-muted mt-1">
          {trainingType
            ? 'Running GridSearchCV with stratified 80/20 split for maximum accuracy'
            : explaining
              ? 'Re-training chosen model and computing SHAP explanations...'
              : trainedCount > 0
                ? 'Train multiple models to compare, then pick the best one to explain'
                : 'Choose one model, then click Train to start hyperparameter-tuned training'}
        </p>
        {cleaningSessionId && trainedCount === 0 && !trainingType && (
          <p className="text-[10px] text-positive mt-2 bg-positive/8 inline-block px-3 py-1 rounded-full">
            Using your cleaned training data (saved to disk)
          </p>
        )}
      </Card>

      {/* Model Picker Grid — always visible */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {MODEL_OPTIONS.map((option, idx) => {
          const Icon = option.icon;
          const isTrained = !!trainedResults[option.type];
          const isCurrentlyTraining = trainingType === option.type;
          const isSelected = selectedModelType === option.type;
          const canSelect = !isTrained && !trainingType && !explaining;

          return (
            <motion.div
              key={option.type}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <Card
                hover={canSelect}
                className={`relative transition-all ${
                  canSelect ? 'cursor-pointer' : ''
                } ${
                  isCurrentlyTraining
                    ? `ring-2 ${option.ring} shadow-lg`
                    : isTrained
                      ? 'ring-1 ring-positive/30'
                      : isSelected
                        ? `ring-2 ${option.ring} shadow-lg`
                        : canSelect
                          ? 'hover:border-border-hover'
                          : 'opacity-60'
                }`}
                onClick={() => canSelect && !isTrained && setSelectedModelType(option.type)}
              >
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${option.color}`}>
                      <Icon size={18} />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-text-primary">{option.name}</p>
                      {isTrained && (
                        <p className="text-[10px] text-positive font-medium">
                          {(trainedResults[option.type].model.metrics.accuracy * 100).toFixed(1)}% accuracy
                        </p>
                      )}
                    </div>
                    {isTrained ? (
                      <div className="w-5 h-5 rounded-full bg-positive flex items-center justify-center">
                        <CheckCircle2 size={14} className="text-white" />
                      </div>
                    ) : isSelected && !isCurrentlyTraining ? (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="w-5 h-5 rounded-full bg-accent flex items-center justify-center"
                      >
                        <CheckCircle2 size={14} className="text-white" />
                      </motion.div>
                    ) : null}
                  </div>

                  {/* Show progress bar when training */}
                  {isCurrentlyTraining && (
                    <ProgressBar
                      value={trainingProgress}
                      color={trainingProgress >= 100 ? 'positive' : 'accent'}
                      height="md"
                      showLabel
                    />
                  )}

                  {/* Description for untrained models */}
                  {!isTrained && !isCurrentlyTraining && (
                    <p className="text-[11px] text-text-muted leading-relaxed">
                      {option.description}
                    </p>
                  )}

                  {/* Trained badge */}
                  {isTrained && !isCurrentlyTraining && (
                    <p className="text-[10px] text-positive bg-positive/8 inline-block px-2 py-0.5 rounded-full">
                      Trained
                    </p>
                  )}
                </div>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Train button — visible when a model is selected and not currently training */}
      <AnimatePresence>
        {selectedModelType && !trainedResults[selectedModelType] && !trainingType && !explaining && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex justify-center"
          >
            <button
              onClick={handleStartTraining}
              className="flex items-center gap-2 px-6 py-3 rounded-lg bg-accent hover:bg-accent-light text-white text-sm font-medium transition-colors"
            >
              <Brain size={16} />
              Train {MODEL_OPTIONS.find((m) => m.type === selectedModelType)?.name}
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error banner */}
      {error && (
        <Card className="flex items-center gap-3 text-warning">
          <AlertTriangle size={16} />
          <span className="text-xs">{error}</span>
        </Card>
      )}

      {/* Split Info Banner — show from most recent training */}
      {latestSplitInfo && trainedCount > 0 && !trainingType && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center shrink-0">
              <SplitSquareVertical size={18} className="text-accent" />
            </div>
            <div className="flex-1 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <p className="text-text-muted">Split</p>
                <p className="text-text-primary font-medium">{latestSplitInfo.splitRatio} (Train/Test)</p>
              </div>
              <div>
                <p className="text-text-muted">Train / Test</p>
                <p className="text-text-primary font-medium">{latestSplitInfo.trainSize} / {latestSplitInfo.testSize} rows</p>
              </div>
              <div>
                <p className="text-text-muted">Stratified</p>
                <p className="text-text-primary font-medium">{latestSplitInfo.stratified ? 'Yes' : 'No'} ({latestSplitInfo.targetType})</p>
              </div>
              <div>
                <p className="text-text-muted">CV Folds</p>
                <p className="text-text-primary font-medium">{latestSplitInfo.cvFolds}-fold{latestSplitInfo.nClasses ? ` (${latestSplitInfo.nClasses} classes)` : ''}</p>
              </div>
            </div>
          </Card>
        </motion.div>
      )}

      {/* Trained Results Comparison Section */}
      <AnimatePresence>
        {trainedCount > 0 && !trainingType && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <h3 className="text-sm font-semibold text-text-primary">
              Trained Models — Select one for Explanation
            </h3>

            {Object.entries(trainedResults).map(([type, { model }]) => {
              const opt = MODEL_OPTIONS.find((o) => o.type === type);
              const Icon = opt?.icon ?? Brain;
              const colorClass = opt?.color ?? 'text-text-muted bg-bg-card';
              const isChosen = chosenForExplain === type;

              return (
                <motion.div
                  key={type}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <Card
                    className={`relative overflow-hidden cursor-pointer transition-all ${
                      isChosen
                        ? 'ring-2 ring-accent/60 shadow-[0_0_20px_rgba(99,102,241,0.12)]'
                        : 'hover:border-border-hover'
                    }`}
                    onClick={() => setChosenForExplain(type)}
                  >
                    <div className="space-y-3">
                      {/* Model header with radio */}
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${colorClass}`}>
                          <Icon size={18} />
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-medium text-text-primary">{model.name}</p>
                        <p className="text-[10px] text-positive">
                            {model.metrics.accuracy != null
                              ? `Accuracy: ${(model.metrics.accuracy * 100).toFixed(1)}%`
                              : model.metrics.r2 != null
                                ? `R²: ${(model.metrics.r2 * 100).toFixed(1)}%`
                                : ''}
                          </p>
                        </div>
                        {/* Radio indicator */}
                        <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                          isChosen ? 'border-accent bg-accent' : 'border-text-muted'
                        }`}>
                          {isChosen && <CircleDot size={12} className="text-white" />}
                        </div>
                      </div>

                      {/* Metrics grid */}
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 pt-2 border-t border-border">
                        {Object.entries(model.metrics).map(([key, value]) => {
                          const isPercent = ['accuracy', 'aucRoc', 'f1', 'precision', 'recall', 'r2'].includes(key);
                          const labelMap: Record<string, string> = {
                            aucRoc: 'AUC-ROC', accuracy: 'Accuracy', f1: 'F1 Score',
                            precision: 'Precision', recall: 'Recall',
                            rmse: 'RMSE', mae: 'MAE', r2: 'R²', mse: 'MSE',
                          };
                          return (
                            <div key={key} className="text-center">
                              <p className="text-[10px] text-text-muted">{labelMap[key] ?? key}</p>
                              <p className="text-xs font-mono font-semibold text-text-primary">
                                {isPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(4)}
                              </p>
                            </div>
                          );
                        })}
                      </div>

                      {/* CV Score + Hyperparameters row */}
                      <div className="flex flex-wrap gap-4 pt-2 border-t border-border/50">
                        {model.cvScore != null && model.cvScore !== 0 && (
                          <div>
                            <p className="text-[10px] text-text-muted">CV Score</p>
                            <p className="text-[10px] font-mono text-text-secondary">
                              {model.cvScore.toFixed(4)}
                            </p>
                          </div>
                        )}
                        {model.bestParams && Object.keys(model.bestParams).length > 0 && (
                          <div className="flex-1">
                            <div className="flex items-center gap-1 mb-1">
                              <Settings2 size={10} className="text-text-muted" />
                              <span className="text-[10px] font-medium text-text-muted">
                                Tuned Hyperparameters
                              </span>
                            </div>
                            <div className="flex flex-wrap gap-x-4 gap-y-0.5">
                              {Object.entries(model.bestParams).map(([key, val]) => (
                                <span key={key} className="text-[10px] font-mono text-text-muted">
                                  {key}: <span className="text-accent font-medium">{String(val)}</span>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* "Use This Model" label */}
                    {isChosen && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="absolute top-2 right-2"
                      >
                        <span className="text-[10px] font-medium text-accent bg-accent/10 px-2 py-0.5 rounded-full">
                          Selected
                        </span>
                      </motion.div>
                    )}
                  </Card>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Proceed bar — appears after a model is chosen */}
      <AnimatePresence>
        {chosenForExplain && chosenModel && !trainingType && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card glow className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-positive-bg flex items-center justify-center">
                  <CheckCircle2 size={24} className="text-positive" />
                </div>
                <div>
                  <p className="text-sm font-medium text-text-primary">
                    {chosenModel.name}
                  </p>
                  <p className="text-xs text-text-muted">
                    {chosenModel.metrics.accuracy != null
                      ? `Accuracy: ${(chosenModel.metrics.accuracy * 100).toFixed(1)}%`
                      : chosenModel.metrics.r2 != null
                        ? `R²: ${(chosenModel.metrics.r2 * 100).toFixed(1)}%`
                        : ''}
                    {chosenModel.metrics.aucRoc != null
                      ? ` · AUC-ROC: ${(chosenModel.metrics.aucRoc * 100).toFixed(1)}%`
                      : chosenModel.metrics.rmse != null
                        ? ` · RMSE: ${chosenModel.metrics.rmse.toFixed(4)}`
                        : ''}
                  </p>
                </div>
              </div>
              <button
                onClick={handleProceed}
                disabled={explaining}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-light text-white text-sm font-medium transition-colors disabled:opacity-60"
              >
                {explaining ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Preparing model...
                  </>
                ) : (
                  <>
                    Explain & Certify
                    <ArrowRight size={16} />
                  </>
                )}
              </button>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
