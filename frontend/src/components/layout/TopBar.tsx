import { usePipelineStore } from '../../store/pipeline';
import { RotateCcw } from 'lucide-react';
import BiasScoreBadge from '../shared/BiasScoreBadge';

const stageLabels: Record<number, string> = {
  1: 'Upload & Understand Your Data',
  2: 'Clean With Bias Awareness',
  3: 'Bias Gate — Pre-Training Audit',
  4: 'Train & Select Models',
  5: 'Explain & Certify',
};

export default function TopBar() {
  const { currentStage, currentBiasScore, baselineBias, fileName, reset } =
    usePipelineStore();

  return (
    <header className="h-14 bg-bg-secondary border-b border-border flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <div>
          <h2 className="text-sm font-semibold text-text-primary">
            {stageLabels[currentStage]}
          </h2>
          {fileName && (
            <p className="text-[11px] text-text-muted">{fileName}</p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-4">
        {baselineBias && (
          <div className="flex items-center gap-3">
            <span className="text-[11px] text-text-muted uppercase tracking-wider">
              Bias Score
            </span>
            <BiasScoreBadge score={currentBiasScore} size="sm" />
          </div>
        )}
        {fileName && (
          <button
            onClick={reset}
            className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-secondary transition-colors px-2 py-1 rounded hover:bg-bg-card"
          >
            <RotateCcw size={12} />
            Reset
          </button>
        )}
      </div>
    </header>
  );
}
