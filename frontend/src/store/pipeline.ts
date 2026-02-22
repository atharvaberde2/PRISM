import { create } from 'zustand';
import type { PipelineState, Stage, ColumnProfile, BiasReport, CleaningStep, GateResult, ModelResult, ExplainResult, FeatureCleanState, SplitInfo } from '../lib/types';

interface PipelineActions {
  setStage: (stage: Stage) => void;
  unlockStage: (stage: Stage) => void;

  // Stage 1
  setFileData: (fileName: string, columns: ColumnProfile[], rawData: Record<string, unknown>[]) => void;
  setSelectedColumns: (cols: string[]) => void;
  setTargetColumn: (col: string) => void;
  setSensitiveColumn: (col: string) => void;
  setBaselineBias: (report: BiasReport) => void;
  setUploadedFile: (file: File) => void;

  // Stage 2 (legacy)
  setCleaningSteps: (steps: CleaningStep[]) => void;
  updateCleaningStep: (id: string, update: Partial<CleaningStep>) => void;
  setCurrentBiasScore: (score: number) => void;
  setIsGeneratingSteps: (v: boolean) => void;

  // Stage 2 – iterative
  setCleaningSessionId: (id: string) => void;
  setFeatureStates: (states: FeatureCleanState[]) => void;
  setActiveFeature: (feature: string | null) => void;
  updateFeatureState: (feature: string, update: Partial<FeatureCleanState>) => void;

  // Stage 3
  setGateResult: (result: GateResult) => void;

  // Stage 4
  setModels: (models: ModelResult[]) => void;
  updateModel: (name: string, update: Partial<ModelResult>) => void;
  setIsTraining: (v: boolean) => void;
  setSplitInfo: (info: SplitInfo | null) => void;

  // Stage 5
  setExplainResult: (result: ExplainResult) => void;

  reset: () => void;
}

const initialState: PipelineState = {
  currentStage: 1,
  maxUnlockedStage: 1,
  fileName: null,
  columns: [],
  selectedColumns: [],
  targetColumn: null,
  sensitiveColumn: null,
  baselineBias: null,
  rawData: null,
  uploadedFile: null,
  cleaningSteps: [],
  currentBiasScore: 0,
  isGeneratingSteps: false,
  cleaningSessionId: null,
  featureStates: [],
  activeFeature: null,
  gateResult: null,
  models: [],
  isTraining: false,
  splitInfo: null,
  explainResult: null,
};

export const usePipelineStore = create<PipelineState & PipelineActions>((set) => ({
  ...initialState,

  setStage: (stage) => set({ currentStage: stage }),
  unlockStage: (stage) => set((s) => ({
    maxUnlockedStage: Math.max(s.maxUnlockedStage, stage) as Stage,
  })),

  setFileData: (fileName, columns, rawData) => set({
    fileName,
    columns,
    rawData,
    selectedColumns: columns.map((c) => c.name),
  }),
  setSelectedColumns: (cols) => set({ selectedColumns: cols }),
  setTargetColumn: (col) => set({ targetColumn: col }),
  setSensitiveColumn: (col) => set({ sensitiveColumn: col }),
  setBaselineBias: (report) => set({
    baselineBias: report,
    currentBiasScore: report.overallScore,
  }),
  setUploadedFile: (file) => set({ uploadedFile: file }),

  setCleaningSteps: (steps) => set({ cleaningSteps: steps }),
  updateCleaningStep: (id, update) => set((s) => ({
    cleaningSteps: s.cleaningSteps.map((step) =>
      step.id === id ? { ...step, ...update } : step
    ),
  })),
  setCurrentBiasScore: (score) => set({ currentBiasScore: score }),
  setIsGeneratingSteps: (v) => set({ isGeneratingSteps: v }),

  setCleaningSessionId: (id) => set({ cleaningSessionId: id }),
  setFeatureStates: (states) => set({ featureStates: states }),
  setActiveFeature: (feature) => set({ activeFeature: feature }),
  updateFeatureState: (feature, update) => set((s) => ({
    featureStates: s.featureStates.map((f) =>
      f.feature === feature ? { ...f, ...update } : f
    ),
  })),

  setGateResult: (result) => set({ gateResult: result }),

  setModels: (models) => set({ models }),
  updateModel: (name, update) => set((s) => ({
    models: s.models.map((m) =>
      m.name === name ? { ...m, ...update } : m
    ),
  })),
  setIsTraining: (v) => set({ isTraining: v }),
  setSplitInfo: (info) => set({ splitInfo: info }),

  setExplainResult: (result) => set({ explainResult: result }),

  reset: () => set(initialState),
}));
